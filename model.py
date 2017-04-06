import time
import random

import numpy as np
import tensorflow as tf
import sys
sys.path.append('../sol-env')
import sol_env

class Buffer(object):
    def __init__(self, size):
        self.size = size
        self.items = []
        self.index = 0

    def insert(self, item):
        if len(self.items) < self.size:
            self.items.append(item)
        else:
            self.items[self.index] = item
            self.index = (self.index + 1) % self.size

    def sample(self, k):
        return random.sample(self.items, k)


def q_func(visible_ph, suit_ph, rank_ph, pos_ph, seq_len_ph, type_ph, n_slots, max_stack_len, scope, reuse=False):
    emb_size = 32
    lstm_size = 32
    hidden_size = 64
    n_types = 8 # number of slot types

    with tf.variable_scope(scope, reuse=reuse):

        # lookup embeddings for card properties
        vis_emb_matrix = tf.get_variable('vis_emb', [3, emb_size], tf.float32, tf.random_normal_initializer())
        vis_emb = tf.nn.embedding_lookup(vis_emb_matrix, visible_ph)

        suit_emb_matrix = tf.get_variable('suit_emb', [6, emb_size], tf.float32, tf.random_normal_initializer())
        suit_emb = tf.nn.embedding_lookup(suit_emb_matrix, suit_ph)

        rank_emb_matrix = tf.get_variable('rank_emb', [16, emb_size], tf.float32, tf.random_normal_initializer())
        rank_emb = tf.nn.embedding_lookup(rank_emb_matrix, rank_ph)
        
        position_emb_matrix = tf.get_variable('pos_emb', [max_stack_len, emb_size], tf.float32, tf.random_normal_initializer())
        pos_emb = tf.nn.embedding_lookup(position_emb_matrix, pos_ph)

        card_reps = tf.concat([vis_emb, suit_emb, rank_emb, pos_emb],3)

        # run rnn over cards in each slot
        card_reps_reshaped = tf.reshape(card_reps, [-1, max_stack_len, emb_size*4])
        seq_len = tf.reshape(seq_len_ph, [-1])

        outputs, (c,h) = tf.nn.dynamic_rnn(tf.contrib.rnn.LSTMCell(lstm_size), card_reps_reshaped, seq_len, dtype=tf.float32, scope='card_lstm')
        # state is batch_size*n_slots by lstm_size

        card_slot_rep = tf.reshape(c, [-1, n_slots, lstm_size])

        # concat in slot properties

        type_emb_matrix = tf.get_variable('type_emb', [n_types, emb_size], tf.float32, tf.random_normal_initializer())
        type_emb = tf.nn.embedding_lookup(type_emb_matrix, type_ph)

        slot_reps = tf.concat([card_slot_rep, type_emb], 2)
        # slot_reps is batch_size by n_slots by emb_size+lstm_size

        # run rnn over all the slots

        outputs, (c,h) = tf.nn.dynamic_rnn(tf.contrib.rnn.LSTMCell(lstm_size), slot_reps, dtype=tf.float32, scope='slot_lstm')

        global_context = c # size batch_size by lstm_size

        context_repeated = tf.tile(tf.expand_dims(global_context, 1), (1, n_slots, 1))

        slot_hiddens = tf.layers.conv1d(tf.concat([slot_reps,context_repeated],2), hidden_size, 1)

        click_slot_q_values = tf.layers.conv1d(slot_hiddens, 1, 1) # size batch_size by n_slots
        
        # TODO: the drag and drop values need more context and a hidden layer
        drag_card_q_values = tf.layers.conv2d(card_reps, 1, 1) # size batch_size by n_slots by max_stack_len
        drop_card_q_values = tf.layers.conv1d(slot_hiddens, 1, 1)

        q_values = tf.concat([tf.squeeze(click_slot_q_values,axis=[2]),
                   tf.squeeze(drop_card_q_values,axis=[2]),
                   tf.reshape(drag_card_q_values, (-1,n_slots*max_stack_len))],1)

    return q_values

class Model(object):
    def __init__(self, n_slots, max_stack_len):

        gamma = .99
        grad_clip_val = 10

        self.n_slots = n_slots
        self.max_stack_len = max_stack_len

	# placeholders
        self.visible_ph = tf.placeholder(tf.int32, [None,n_slots,max_stack_len]) # values are pad,y,n
        self.suit_ph = tf.placeholder(tf.int32, [None,n_slots,max_stack_len]) # values are pad,?,S,C,H,D
        self.rank_ph = tf.placeholder(tf.int32, [None,n_slots, max_stack_len]) # pad,?,1,...,13
        self.pos_ph = tf.placeholder(tf.int32, [None,n_slots,max_stack_len])
        self.seq_len_ph = tf.placeholder(tf.int32, [None,n_slots])
        self.type_ph = tf.placeholder(tf.int32, [None,n_slots])

	# placeholders for next timestep
        self.next_visible_ph = tf.placeholder(tf.int32, [None,n_slots,max_stack_len]) # values are pad,y,n
        self.next_suit_ph = tf.placeholder(tf.int32, [None,n_slots,max_stack_len]) # values are pad,?,S,C,H,D
        self.next_rank_ph = tf.placeholder(tf.int32, [None,n_slots, max_stack_len]) # pad,?,1,...,13
        self.next_pos_ph = tf.placeholder(tf.int32, [None,n_slots,max_stack_len])
        self.next_seq_len_ph = tf.placeholder(tf.int32, [None,n_slots])
        self.next_type_ph = tf.placeholder(tf.int32, [None,n_slots])

	# other placeholders
        self.action_ph = tf.placeholder(tf.int32, [None])
        self.reward_ph = tf.placeholder(tf.float32, [None])
        self.done_mask_ph = tf.placeholder(tf.float32, [None])

	# compute q values
        self.q_values = q_func(self.visible_ph, self.suit_ph, self.rank_ph, self.pos_ph, self.seq_len_ph, self.type_ph, n_slots, max_stack_len, 'q_func', False)
        q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')

	# compute q and target q network values for next timestep
        next_q_values = q_func(self.next_visible_ph, self.next_suit_ph, self.next_rank_ph, self.next_pos_ph, self.next_seq_len_ph, self.next_type_ph, n_slots, max_stack_len, 'q_func', reuse=True)
        target_next_q_values = q_func(self.next_visible_ph, self.next_suit_ph, self.next_rank_ph, self.next_pos_ph, self.next_seq_len_ph, self.next_type_ph, n_slots, max_stack_len, 'target_q_func', reuse=False)
        target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_q_func')

        # select the q values for the selected actions
        cat_idx_t = tf.stack([tf.range(0, tf.shape(self.q_values)[0]), self.action_ph], axis=1)
        q_for_actions = tf.gather_nd(self.q_values, cat_idx_t)
        # select the best q values for the next action (double DQN style)
        best_next_action = tf.cast(tf.argmax(next_q_values,1), tf.int32)
        cat_idx_tp1 = tf.stack([tf.range(0, tf.shape(self.q_values)[0]), best_next_action], axis=1)
        best_q_next = tf.gather_nd(target_next_q_values, cat_idx_tp1)

        # error
        error = q_for_actions - (self.reward_ph + (1-self.done_mask_ph)*gamma*best_q_next)
        total_error = tf.reduce_sum(error * error)

        # construct optimization op (with gradient clipping)
        self.learning_rate = tf.placeholder(tf.float32, (), name="learning_rate")
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        gradients = optimizer.compute_gradients(total_error)
        for i, (grad, var) in enumerate(gradients):
            if grad is not None:
                gradients[i] = (tf.clip_by_norm(grad, grad_clip_val), var)
        self.train_fn = optimizer.apply_gradients(gradients)

        # update_target_fn will be called periodically to copy Q network to target Q network
        update_target_fn = []
        for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                sorted(target_q_func_vars, key=lambda v: v.name)):
            update_target_fn.append(var_target.assign(var))
        self.update_target_fn = tf.group(*update_target_fn)


    def evaluate_q_values(self, observation, session):
        return session.run(self.q_values, feed_dict=self.feed_dict_from_obs([observation]))

    def train_step(self, samples, session):
        learning_rate = .001

        # samples looks like (last_obs, action_id, obs, rew, done)
        obs_feed = self.feed_dict_from_obs([o for o, _, _, _, _ in samples])
        next_obs_feed = self.feed_dict_from_obs([no for _, _, no, _, _ in samples], True)

        n_samp = len(samples)
        actions = np.zeros((n_samp), dtype=np.int32)
        rewards = np.zeros((n_samp), dtype=np.float32)
        dones = np.zeros((n_samp), dtype=np.float32)
        for i, sample in enumerate(samples):
            _, act, _, rew, done = sample
            actions[i] = act
            rewards[i] = rew
            dones[i] = 1 if done else 0
            
        feed_dict = {self.learning_rate:learning_rate, self.action_ph:actions, self.reward_ph:rewards, self.done_mask_ph:dones}
        feed_dict.update(obs_feed)
        feed_dict.update(next_obs_feed)

        session.run(self.train_fn, feed_dict=feed_dict)

    def feed_dict_from_obs(self, observations, next_values=False):
        n_obs = len(observations)
        lengths = np.zeros((n_obs,self.n_slots), dtype=np.int32)
        types = np.zeros((n_obs,self.n_slots), dtype=np.int32)
        visible = np.zeros((n_obs,self.n_slots,self.max_stack_len), dtype=np.int32)
        rank = np.zeros((n_obs,self.n_slots,self.max_stack_len), dtype=np.int32)
        suit = np.zeros((n_obs,self.n_slots,self.max_stack_len), dtype=np.int32)
        pos = np.zeros((n_obs,self.n_slots,self.max_stack_len), dtype=np.int32)
        
        for obs_num in range(n_obs):
            observation = observations[obs_num]
            assert len(observation) == self.n_slots
            for slot_num in range(self.n_slots):
                slot_type, cards = observation[slot_num]

                types[obs_num,slot_num] = slot_type.value

                length = len(cards)
                lengths[obs_num,slot_num] = length
                pos[obs_num,slot_num,:length] = range(length,0,-1) # from stack_length to 1, leaving 0 for padding
                for card_num in range(length):
                    if card_num >= self.max_stack_len:
                        print('stack too large')
                        break
                    v, s, r = cards[card_num]
                    v = int(v) + 1
                    s += 1
                    r += 1
                    assert v > 0 and r > 0 and s > 0 # let 0 be the padding value
                    visible[obs_num,slot_num,card_num] = v
                    rank[obs_num,slot_num,card_num] = r
                    suit[obs_num,slot_num,card_num] = s
                
        if next_values:
            return {self.next_visible_ph:visible,self.next_suit_ph:suit, self.next_rank_ph:rank, self.next_pos_ph:pos, self.next_seq_len_ph:lengths, self.next_type_ph:types}
        else:
            return {self.visible_ph:visible,self.suit_ph:suit, self.rank_ph:rank, self.pos_ph:pos, self.seq_len_ph:lengths, self.type_ph:types}

    def action_id_to_action(self, action_id):
        n_slots = self.n_slots
        max_stack_len = self.max_stack_len
        if action_id < n_slots:
            return sol_env.ActionType.CLICK, action_id, 0
        elif action_id < 2*n_slots:
            return sol_env.ActionType.DROP, action_id-n_slots, 0
        elif action_id < n_slots*(2+max_stack_len):
            a = (action_id-2*n_slots)
            
            return sol_env.ActionType.DRAG, a//max_stack_len, a%max_stack_len
        else:
            assert False # out of range

    def n_actions(self):
        return self.n_slots*(2+self.max_stack_len)

    def update_target(self, session):
        session.run(self.update_target_fn)

def card_count(obs):
    n = 0
    for slot in obs:
        n += len(slot[1])
    return n

def main():
    update_freq = 4
    batch_size = 32
    learning_starts = 50000
    max_steps = 5000000
    max_steps_per_ep = 5000
    buffer_size = 100000
    init_eps = .95
    final_eps = .01
    final_eps_timestep = 500000
    target_update_freq = 10000

    env = sol_env.SolEnv()
    obs = env.reset()
    buff = Buffer(buffer_size)

    n_cards = card_count(obs)

    with tf.Session() as sess:
        model = Model(len(obs),10) # TODO: max stack len?
        tf.global_variables_initializer().run()

        episode_t = 0
        episode_reward = 0
        num_param_updates = 0
        finished_episode_rewards = []
        for t in range(1,max_steps+1):
            episode_t += 1

            if t >= final_eps_timestep:
                eps = final_eps
            else:
                eps = init_eps + (final_eps-init_eps)*t/final_eps_timestep

            if t <= learning_starts or random.random() < eps:
                action_id = random.randrange(model.n_actions())
            else:
                q_values = model.evaluate_q_values(obs, sess)
                action_id = q_values.argmax() # XXX should we mask invalid ones?

            last_obs = obs
            action = model.action_id_to_action(action_id)
            obs, rew, done, info = env.step(action)
            episode_reward += rew

            buff.insert((last_obs, action_id, obs, rew, done))

            if done or episode_t > max_steps_per_ep:
                obs = env.reset()
                finished_episode_rewards.append(episode_reward)
                episode_t = 0
                episode_reward = 0

            if t >= learning_starts and t % update_freq == 0:
                model.train_step(buff.sample(batch_size), sess)
                num_param_updates += 1

            if num_param_updates % target_update_freq == 0:
                model.update_target(sess)

            if t % 10000 == 0:
                print('iteration:', t)
                print('epsilon greedy:', eps)
                mean_rew = np.mean(finished_episode_rewards[-10:])
                print('mean reward:', mean_rew)

main()
