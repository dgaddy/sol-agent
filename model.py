try:
    import xdbg
    INTERACTIVE = True
except:
    INTERACTIVE = False

if INTERACTIVE:
    import os
    os.chdir(os.path.expanduser('~/dev/cards/sol-agent'))

# %%
import time
import random
import pickle
import os
import os.path as path
from collections import defaultdict

import numpy as np
import tensorflow as tf
import gym
import sys
sys.path.append('../sol-env')
import sol_env

# %%

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class Buffer(object):
    def __init__(self, size):
        self.size = size
        self.items = []
        self.item_scores = np.zeros(self.size)
        self.index = 0

    def insert(self, item, score=None):
        if score is None:
            score = item[3]

        if len(self.items) < self.size:
            self.items.append(item)
            self.item_scores[len(self.items) - 1] = score
        else:
            self.items[self.index] = item
            self.item_scores[self.index] = score
            self.index = (self.index + 1) % self.size

    def reprioritize(self, idxs, td_errors):
        return # disabled currently
        td_errors = np.abs(td_errors)
        for idx, td_error in zip(idxs, td_errors):
            self.item_scores[idx] = td_error

        self.typical_score = max(0.99 * self.typical_score, np.max(td_errors) * 1.2)

    def print_debug(self):
        p = self.item_scores[:len(self.items)]
        p = p / np.sum(p) # /= would mutate item scores in place!

        print('Buffer debug statistics:')
        print('  items:', len(self.items))
        print('  prob ratio:', np.max(p) / np.min(p))
        _, idxs, weights = self.sample(20)
        print('  weights:', weights)
        print('  scores:', [self.item_scores[i] for i in idxs])
        print('  rewards:', [self.items[i][3] for i in idxs])

    def sample(self, k):
        p = self.item_scores[:len(self.items)]
        p = p / np.sum(p) # /= would mutate item scores in place!
        idxs = np.random.choice(len(self.items), k, p=p)

        weights = 1 / (len(self.items) * np.array([p[i] for i in idxs]))
        weights = weights ** 0.6
        weights /= np.max(weights)
        return [self.items[i] for i in idxs], idxs, weights

class DebugHelper(object):
    def __init__(self, n_slots, max_stack_len):
        self.items = []
        self.index = 0
        self.n_slots = n_slots
        self.max_stack_len = max_stack_len

        self.random_items = []
        self.random_index = 0

    def insert(self, observation, q_values, action, reward):
        size = 1000
        item = (observation, q_values, action, reward)
        if len(self.items) < size:
            self.items.append(item)
        else:
            self.items[self.index] = item
            self.index = (self.index + 1) % size

    def insert_random(self, observation, action, reward):
        size = 1000
        item = (observation, action, reward)
        if len(self.random_items) < size:
            self.random_items.append(item)
        else:
            self.random_items[self.random_index] = item
            self.random_index = (self.random_index + 1) % size

    def reset(self):
        self.items = []
        self.index = 0

        self.random_items = []
        self.random_index = 0

    def cards_to_strings(self, cards):
        return ["?" if fd else "{}{}".format(rank, "♣♦♥♠"[suit]) for (fd, suit, rank) in cards]

    def print_analysis(self):
        counts = defaultdict(int)
        n_useless_drags = 0
        for obs, q_values, action, reward in self.items:
            action_type = action[0]
            valid = reward >= 0 or reward < -0.99 # -1 reward for dragging down from foundations
            dragging = len(obs[-1][-1])>0

            counts[(action_type, valid)] += 1
            if action_type == sol_env.ActionType.DRAG_DROP and valid and action[1] in [2,3,4,5]:
                n_useless_drags += 1

        if self.random_items:
            counts_random = defaultdict(int)
            n_random_useless_drags = 0
            for obs, action, reward in self.random_items:
                action_type = action[0]
                valid = reward >= 0 or reward < -0.99

                counts_random[(action_type, valid)] += 1
                if action_type == sol_env.ActionType.DRAG_DROP and valid and action[1] in [2,3,4,5]:
                    n_random_useless_drags += 1


        print('===========debug info===========')
        for at in sol_env.ActionType:
            print('invalid', at, counts[(at,False)], '/', counts[(at,True)] + counts[(at,False)])
        print('useless drags', n_useless_drags, '/', counts[(sol_env.ActionType.DRAG_DROP, True)])

        if self.random_items:
            for at in sol_env.ActionType:
                print('random invalid', at, counts_random[(at,False)], '/', counts_random[(at,True)] + counts_random[(at,False)])
            print('random useless drags', n_random_useless_drags, '/', counts_random[(sol_env.ActionType.DRAG_DROP, True)])

        if False:
            prev_obs = None
            for obs, q_values, action, reward in self.items[::10]:
                if (reward >= 0) or action[0] != sol_env.ActionType.DRAG_DROP:
                    continue
                if obs == prev_obs:
                    continue
                prev_obs = obs

                _, from_slot, card_id, dst_slot = action
                from_cards = self.cards_to_strings(obs[from_slot][1])
                dst_cards = self.cards_to_strings(obs[dst_slot][1])

                from_type = obs[from_slot][0].name.lower()
                dst_type = obs[dst_slot][0].name.lower()

                from_str = '{} [{}]'.format(' '.join(from_cards[-10:-card_id - 1]), ' '.join(from_cards[-card_id - 1:]))
                dst_str = ' '.join(dst_cards[-10:])
                print("illegal {} {}: {} -> {} {}: {}".format(from_slot, from_type, from_str, dst_slot, dst_type, dst_str))
                # for i, (slot_type, cards) in enumerate(obs):
                #     print("  {} {}: {}".format(
                #         i, slot_type.name.lower(), ' '.join(self.cards_to_strings(cards[-10:]))))

            prev_obs = None
            for obs, q_values, action, reward in self.items:
                if action[0] != sol_env.ActionType.CLICK:
                    continue

                if obs == prev_obs:
                    continue
                prev_obs = obs

                slot_type = obs[action[1]][0].name.lower()
                cards = self.cards_to_strings(obs[action[1]][1])
                if (reward >= 0):
                    print('legal click {} {}: {}'.format(action[1], slot_type, ' '.join(cards[-10:])))
                else:
                    print('illegal click {} {}: {}'.format(action[1], slot_type, ' '.join(cards[-10:])))
                    for i, (slot_type, cards) in enumerate(obs):
                        print("  {} {}: {}".format(
                            i, slot_type.name.lower(), ' '.join(self.cards_to_strings(cards[-10:]))))



def batch_gather(params, indices, validate_indices=None,
    batch_size=None,
    options_size=None):
    """
    Gather slices from `params` according to `indices`, separately for each
    example in a batch.

    output[b, i, ..., j, :, ..., :] = params[b, indices[b, i, ..., j], :, ..., :]

    The arguments `batch_size` and `options_size`, if provided, are used instead
    of looking up the shape from the inputs. This may help avoid redundant
    computation (TODO: figure out if tensorflow's optimizer can do this automatically)

    Args:
      params: A `Tensor`, [batch_size, options_size, ...]
      indices: A `Tensor`, [batch_size, ...]
      validate_indices: An optional `bool`. Defaults to `True`
      batch_size: (optional) an integer or scalar tensor representing the batch size
      options_size: (optional) an integer or scalar Tensor representing the number of options to choose from
    """
    if batch_size is None:
        batch_size = params.get_shape()[0].merge_with(indices.get_shape()[0]).value
        if batch_size is None:
            batch_size = tf.shape(indices)[0]

    if options_size is None:
        options_size = params.get_shape()[1].value
        if options_size is None:
            options_size = tf.shape(params)[1]

    batch_size_times_options_size = batch_size * options_size

    # TODO(nikita): consider using gather_nd. However as of 1/9/2017 gather_nd
    # has no gradients implemented.
    flat_params = tf.reshape(params, tf.concat([[batch_size_times_options_size], tf.shape(params)[2:]], 0))

    indices_offsets = tf.reshape(tf.range(batch_size) * options_size, [-1] + [1] * (len(indices.get_shape())-1))
    indices_into_flat = indices + tf.cast(indices_offsets, indices.dtype)

    return tf.gather(flat_params, indices_into_flat, validate_indices=validate_indices)


def q_func(visible_ph, suit_ph, rank_ph, pos_ph, seq_len_ph, type_ph, valid_ph, n_slots, max_stack_len, scope, reuse=False):
    emb_size = 64
    lstm_size = 128
    hidden_size = 128
    n_types = 8 # number of slot types.

    with tf.variable_scope(scope, reuse=reuse):

        # lookup embeddings for card properties
        vis_emb_matrix = tf.get_variable('vis_emb', [3, emb_size], tf.float32, tf.random_normal_initializer())
        vis_emb = tf.nn.embedding_lookup(vis_emb_matrix, visible_ph)

        suit_emb_matrix = tf.get_variable('suit_emb', [6, emb_size], tf.float32, tf.random_normal_initializer())
        suit_emb = tf.nn.embedding_lookup(suit_emb_matrix, suit_ph)

        rank_emb_matrix = tf.get_variable('rank_emb', [16, emb_size], tf.float32, tf.random_normal_initializer())
        rank_emb = tf.nn.embedding_lookup(rank_emb_matrix, rank_ph)

        card_reps = tf.concat([vis_emb, suit_emb, rank_emb],3)

        # run rnn over cards in each slot
        card_reps_reshaped = tf.reshape(card_reps, [-1, max_stack_len, emb_size*3])
        seq_len = tf.reshape(seq_len_ph, [-1])

        cell_fw = tf.contrib.rnn.LSTMCell(lstm_size, use_peepholes=True)
        cell_bw = tf.contrib.rnn.LSTMCell(lstm_size, use_peepholes=True)

        (o1, o2), ((c1, h1), (c2, h2)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, card_reps_reshaped, seq_len, dtype=tf.float32, scope='card_lstm')

        # reverse since the "from" action is parametrized such that the last
        # card in a slot is 0, while our observation has the first card at index 0.
        contextual_card_reps_reversed = tf.reshape(tf.reverse_sequence(tf.concat([o1, o2], -1),
            seq_len, seq_axis=1, batch_axis=0),
            [-1, n_slots, max_stack_len, lstm_size * 2])

        card_slot_rep = tf.reshape(tf.concat([c1, c2], -1), [-1, n_slots, lstm_size*2])

        # concat in slot properties
        type_emb_matrix = tf.get_variable('type_emb', [n_types, emb_size], tf.float32, tf.random_normal_initializer())
        type_emb = tf.nn.embedding_lookup(type_emb_matrix, type_ph)

        slot_reps = tf.concat([card_slot_rep, type_emb], 2)
        contextual_card_reps_reversed = tf.concat([contextual_card_reps_reversed, tf.tile(tf.expand_dims(type_emb, 2), (1,1,max_stack_len,1))], -1)

        # run rnn over all the slots
        # note: apparently the sequence length argument is required
        (o1, o2), ((c1,h1), (c2, h2)) = tf.nn.bidirectional_dynamic_rnn(
            tf.contrib.rnn.LSTMCell(lstm_size), tf.contrib.rnn.LSTMCell(lstm_size),
            slot_reps, tf.tile([n_slots], (tf.shape(slot_reps)[0],)), dtype=tf.float32, scope='slot_lstm')

        global_context = tf.concat([c1, c2], -1) # size batch_size by lstm_size
        contextual_slot_reps = tf.concat([o1, o2], -1)

        context_hidden = tf.layers.dense(global_context, hidden_size, activation=tf.nn.relu)
        value_func = tf.layers.dense(context_hidden, 1) # deuling network style value and advantage separation

        # advantages

        context_repeated = tf.tile(tf.expand_dims(global_context, 1), (1, n_slots, 1))

        slot_hiddens = tf.layers.conv1d(tf.concat([contextual_slot_reps,context_repeated],2), hidden_size, 1, activation=tf.nn.relu)

        click_slot_adv_values = tf.layers.conv1d(slot_hiddens, 1, 1) # size batch_size by n_slots

        # better-parametrized drag-and-drop
        to_slot_reps = tf.expand_dims(tf.expand_dims(contextual_slot_reps, 1), 3) # (?, 1, n_slots, 1, ...)
        from_reps = tf.expand_dims(contextual_card_reps_reversed, 2)

        drag_drop_reps = tf.concat([
            tf.tile(from_reps, (1, 1, n_slots, 1, 1)),
            tf.tile(to_slot_reps, (1, n_slots, 1, max_stack_len, 1))], -1)

        drag_drop_hiddens = tf.layers.dense(drag_drop_reps, hidden_size, activation=tf.nn.relu)

        drag_drop_adv_values = tf.squeeze(tf.layers.dense(drag_drop_hiddens, 1), -1)

        adv_values = tf.concat([tf.squeeze(click_slot_adv_values,axis=[2]),
                   tf.reshape(drag_drop_adv_values, (-1,n_slots*n_slots*max_stack_len))],1)

        adv_values = adv_values - tf.reduce_mean(adv_values, 1, keep_dims=True)

        adv_values = tf.where(valid_ph, adv_values, -1e8 * tf.ones_like(adv_values))

    return value_func + adv_values

class Model(object):
    def __init__(self, n_slots, max_stack_len):

        gamma = 0.97
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
        self.valid_ph = tf.placeholder(tf.bool, [None, self.n_actions()])

        # placeholders for next timestep
        self.next_visible_ph = tf.placeholder(tf.int32, [None,n_slots,max_stack_len]) # values are pad,y,n
        self.next_suit_ph = tf.placeholder(tf.int32, [None,n_slots,max_stack_len]) # values are pad,?,S,C,H,D
        self.next_rank_ph = tf.placeholder(tf.int32, [None,n_slots, max_stack_len]) # pad,?,1,...,13
        self.next_pos_ph = tf.placeholder(tf.int32, [None,n_slots,max_stack_len])
        self.next_seq_len_ph = tf.placeholder(tf.int32, [None,n_slots])
        self.next_type_ph = tf.placeholder(tf.int32, [None,n_slots])
        self.next_valid_ph = tf.placeholder(tf.bool, [None, self.n_actions()])

        # other placeholders
        self.action_ph = tf.placeholder(tf.int32, [None])
        self.reward_ph = tf.placeholder(tf.float32, [None])
        self.done_mask_ph = tf.placeholder(tf.float32, [None])

        # compute q values
        self.q_values = q_func(self.visible_ph, self.suit_ph, self.rank_ph, self.pos_ph, self.seq_len_ph, self.type_ph, self.valid_ph, n_slots, max_stack_len, 'q_func', False)
        q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')

        # compute q and target q network values for next timestep
        next_q_values = q_func(self.next_visible_ph, self.next_suit_ph, self.next_rank_ph, self.next_pos_ph, self.next_seq_len_ph, self.next_type_ph, self.next_valid_ph, n_slots, max_stack_len, 'q_func', reuse=True)
        target_next_q_values = q_func(self.next_visible_ph, self.next_suit_ph, self.next_rank_ph, self.next_pos_ph, self.next_seq_len_ph, self.next_type_ph, self.next_valid_ph, n_slots, max_stack_len, 'target_q_func', reuse=False)
        target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_q_func')

        # select the q values for the selected actions
        cat_idx_t = tf.stack([tf.range(0, tf.shape(self.q_values)[0]), self.action_ph], axis=1)
        q_for_actions = tf.gather_nd(self.q_values, cat_idx_t)
        # select the best q values for the next action (double DQN style)
        best_next_action = tf.cast(tf.argmax(next_q_values,1), tf.int32)
        cat_idx_tp1 = tf.stack([tf.range(0, tf.shape(self.q_values)[0]), best_next_action], axis=1)
        best_q_next = tf.gather_nd(target_next_q_values, cat_idx_tp1)

        # error
        self.error = q_for_actions - (self.reward_ph + (1-self.done_mask_ph)*gamma*best_q_next)

        self.weights_ph = tf.placeholder(tf.float32, [None])
        self.total_error = tf.reduce_mean(self.weights_ph * self.error * self.error)

        # construct optimization op (with gradient clipping)
        self.learning_rate = tf.placeholder(tf.float32, (), name="learning_rate")
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        gradients = optimizer.compute_gradients(self.total_error)
        for i, (grad, var) in enumerate(gradients):
            if grad is not None:
                gradients[i] = (tf.clip_by_norm(grad, grad_clip_val), var)
                gradients[i] = (tf.verify_tensor_all_finite(gradients[i][0], 'non-finite grad'), var)
        self.train_fn = optimizer.apply_gradients(gradients)

        # update_target_fn will be called periodically to copy Q network to target Q network
        update_target_fn = []
        for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                sorted(target_q_func_vars, key=lambda v: v.name)):
            update_target_fn.append(var_target.assign(var))
        self.update_target_fn = tf.group(*update_target_fn)

    def evaluate_q_values(self, observation, session):
        return session.run(self.q_values, feed_dict=self.feed_dict_from_obs([observation])).squeeze()

    def train_step(self, samples, weights, session):
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

        feed_dict = {self.learning_rate:learning_rate, self.action_ph:actions, self.reward_ph:rewards, self.done_mask_ph:dones,
            self.weights_ph: weights}
        feed_dict.update(obs_feed)
        feed_dict.update(next_obs_feed)

        total_error, example_errors, _ = session.run([self.total_error, self.error, self.train_fn], feed_dict=feed_dict)
        return total_error, example_errors

    def feed_dict_from_obs(self, observations, next_values=False):
        n_obs = len(observations)
        lengths = np.zeros((n_obs,self.n_slots), dtype=np.int32)
        types = np.zeros((n_obs,self.n_slots), dtype=np.int32)
        visible = np.zeros((n_obs,self.n_slots,self.max_stack_len), dtype=np.int32)
        rank = np.zeros((n_obs,self.n_slots,self.max_stack_len), dtype=np.int32)
        suit = np.zeros((n_obs,self.n_slots,self.max_stack_len), dtype=np.int32)
        pos = np.zeros((n_obs,self.n_slots,self.max_stack_len), dtype=np.int32)
        valid = np.zeros((n_obs, self.n_actions()), dtype=bool)

        for obs_num in range(n_obs):
            observation = observations[obs_num]
            assert len(observation) == self.n_slots

            valid[obs_num, :] = (self.valid_action_mask(observation) > 0)
            for slot_num in range(self.n_slots):
                slot_type, cards = observation[slot_num]

                # note that Python enums are 1-indexed
                types[obs_num,slot_num] = slot_type.value - 1

                if len(cards) > self.max_stack_len:
                    cards = cards[-self.max_stack_len:] # take the top cards of the stack if it is too long
                length = len(cards)
                lengths[obs_num,slot_num] = length
                pos[obs_num,slot_num,:length] = range(length,0,-1) # from stack_length to 1, leaving 0 for padding
                for card_num in range(length):
                    v, s, r = cards[card_num]
                    v = int(v) + 1
                    s += 1
                    r += 1
                    assert v > 0 and r > 0 and s > 0 # let 0 be the padding value
                    visible[obs_num,slot_num,card_num] = v
                    rank[obs_num,slot_num,card_num] = r
                    suit[obs_num,slot_num,card_num] = s

        if next_values:
            return {self.next_visible_ph:visible,self.next_suit_ph:suit, self.next_rank_ph:rank, self.next_pos_ph:pos, self.next_seq_len_ph:lengths, self.next_type_ph:types, self.next_valid_ph:valid}
        else:
            return {self.visible_ph:visible,self.suit_ph:suit, self.rank_ph:rank, self.pos_ph:pos, self.seq_len_ph:lengths, self.type_ph:types, self.valid_ph:valid}

    def action_id_to_action(self, action_id, obs):
        n_slots = self.n_slots
        max_stack_len = self.max_stack_len
        if action_id < n_slots:
            return sol_env.ActionType.CLICK, action_id
        elif action_id < n_slots+n_slots*n_slots*max_stack_len:
            a = (action_id-n_slots)

            card = a % max_stack_len
            a = a // max_stack_len
            to_slot = a % n_slots
            a = a // n_slots
            from_slot = a

            return sol_env.ActionType.DRAG_DROP, from_slot, card, to_slot
        else:
            assert False # out of range

    def action_to_action_id(self, action):
        n_slots = self.n_slots
        max_stack_len = self.max_stack_len
        action_type = action[0]
        if action_type == sol_env.ActionType.CLICK:
            return action[1]
        elif action_type == sol_env.ActionType.DRAG_DROP:
            from_slot, card, to_slot = action[1:]
            return from_slot * (max_stack_len * n_slots) + to_slot * (max_stack_len) + card + n_slots
        else:
            assert False # unsupported

    def n_actions(self):
        return self.n_slots+self.n_slots*self.n_slots*self.max_stack_len

    def update_target(self, session):
        session.run(self.update_target_fn)

    def valid_action_mask(self, obs):
        # only allow drags for up to the number of cards in the pile
        card_mask = np.zeros((self.n_slots, self.n_slots, self.max_stack_len))
        for slot in range(self.n_slots):
            card_mask[slot, :, :len(obs[slot][1])] = 1
        return np.concatenate((np.ones(self.n_slots), card_mask.flatten()))

def get_initial_sample(buffer_size, env, model, max_steps_per_ep, initial_samples=500000):
    sample_file = 'samples.pkl'
    if path.exists(sample_file):
        with open(sample_file, 'rb') as f:
            return pickle.load(f)
    else:
        print('gathering samples')
        buff = Buffer(buffer_size)
        counter = StateCounter()
        loop_preventer = LoopPreventer(model.n_actions())
        obs = env.reset()
        episode_t = 0
        for t in range(1,initial_samples+1):
            episode_t += 1
            action_mask = model.valid_action_mask(obs)
            loop_preventer_mask, obs_token = loop_preventer.visit(obs)

            action_id = (np.random.random(model.n_actions())*((action_mask > 0) & loop_preventer_mask)).argmax()
            last_obs = obs
            action = model.action_id_to_action(action_id, obs)
            obs, rew, done, info = env.step(action)

            counter_bonus, counter_weight = counter.visit(obs)
            if t > 50000:
                mod_rew = rew + counter_bonus # exploration bonus
            else:
                mod_rew = rew

            buff.insert((last_obs, action_id, obs, mod_rew, done), counter_weight)

            loop_preventer.record_action(obs_token, action_id, (obs == last_obs))

            if done or episode_t > max_steps_per_ep:
                obs = env.reset()
                counter.reset()
                loop_preventer.reset()
                episode_t = 0

        with open(sample_file, 'wb') as f:
            pickle.dump(buff, f)
        return buff

class StateCounter:
    def __init__(self):
        self.table = {}
        self.unique_visited = set()
        self.total = 0
        self.num_types = max(x.value for x in sol_env.SlotType)

    def hash_obs_permutation_invariant(self, obs):
        # this hash is based on number of cards in each stack, but considers
        # slots of the same type as interchangable. Perhaps this can address
        # the useless drags phenomenon
        val = [[] for _ in range(self.num_types)]
        for slot_type, cards in obs:
            face_down = sum([1 for card in cards if card[0]])
            # NOTE: python enums are 1-indexed
            val[slot_type.value - 1].append((len(cards), face_down))
        val = tuple([tuple(sorted(subval)) for subval in val])

        return hash(val)

    def hash_obs(self, obs):
        # this hash is based on number of cards in each stack
        val = []
        for slot_type, cards in obs:
            face_down = sum([1 for card in cards if card[0]])
            val.append((face_down, len(cards)))

        return hash(tuple(val))

    def unique_hash_obs(self, obs):
        return hash(tuple([(a, tuple(b)) for a,b in obs]))

    def visit(self, obs):
        """
        obs -> (bonus_reward, probability_weight)
        """
        h = self.hash_obs(obs)
        if h not in self.table:
            self.table[h] = 1
        else:
            self.table[h] += 1
        self.total += 1

        if self.total < 50000:
            p_weight = 50000
        else:
            p_weight = self.total / self.table[h]

        hu = self.unique_hash_obs(obs)
        if hu not in self.unique_visited:
            self.unique_visited.add(hu)
            # TODO: move multiplicative constant elsewhere
            return 0.25 * np.sqrt(np.log(self.total) / self.table[h]) - 1., p_weight
        else:
            return -1., p_weight # Only give bonus when visiting state for the first time

    def reset(self):
        self.unique_visited = set()
        # self.table = {}

class DebugUniqueCounter:
    def __init__(self):
        self.table = {}
        self.total = 0

    def hash_obs(self, obs):
        return hash(tuple([(a, tuple(b)) for a,b in obs]))

    def visit(self, obs):
        h = self.hash_obs(obs)
        if h not in self.table:
            self.table[h] = 1
        else:
            self.table[h] += 1
        self.total += 1

    def get_counts(self):
        return len(self.table), self.total

    def reset(self):
        self.table = {}
        self.total = 0

class LoopPreventer:
    def __init__(self, n_actions, illegal_per_repeat=50):
        self.count_table = {}
        self.illegal_table = {}
        self.n_actions = n_actions
        self.illegal_per_repeat = illegal_per_repeat

    def hash_obs(self, obs):
        return hash(tuple([(a, tuple(b)) for a,b in obs]))

    def visit(self, obs):
        """
        obs -> (mask, obs_token)
        """
        h = self.hash_obs(obs)
        if h not in self.count_table:
            self.count_table[h] = np.zeros(self.n_actions, dtype=int)
            self.illegal_table[h] = np.zeros(self.n_actions, dtype=bool)

        num_illegal = np.sum(self.illegal_table[h])
        res = self.count_table[h] <= (num_illegal / self.illegal_per_repeat)
        res &= ~self.illegal_table[h]

        return res, h

    def record_action(self, obs_token, action_id, illegal):
        assert obs_token in self.count_table
        self.count_table[obs_token][action_id] += 1
        if illegal:
            self.illegal_table[obs_token][action_id] = True


    def reset(self):
        self.count_table = {}
        self.illegal_table = {}

def main():
    debug = True
    update_freq = 4
    batch_size = 32
    max_steps = 5000000
    max_steps_per_ep = 10000
    buffer_size = 100000
    final_eps_timestep = 1000000
    target_update_freq = 5000
    max_stack_len = 10

    env = sol_env.SolEnv('toy-klondike')
    # NOTE(nikita): I'm enabling unlimited redeals because it makes the observations
    # more Markovian, just to make sure that running out of cards from the deck
    # isn't what's holding up learning
    env.change_options({'Unlimited redeals': True})
    obs = env.reset()
    model = Model(len(obs),max_stack_len)
    counter = StateCounter()
    buff = get_initial_sample(buffer_size, env, model, max_steps_per_ep, 150000)
    for buf_obs, _, _, _, _ in buff.items:
        counter.visit(obs)
    print('Done re-visiting from buffer')

    loop_preventer = LoopPreventer(model.n_actions())

    debug_helper = DebugHelper(len(obs), max_stack_len)
    debug_counter = DebugUniqueCounter()

    saver = tf.train.Saver()

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5))
    with tf.Session(config=config) as sess:
        raw_env = env
        env = gym.wrappers.Monitor(env, os.path.join('/tmp/sol_vid', "gym"), force=True,
            video_callable=lambda num: (num % 1000 < 10) or (num % 10) == 0)
        obs = env.reset()
        tf.global_variables_initializer().run()
        # saver.restore(sess, 'models/model-837000')

        episode_t = 0
        episode_reward = 0
        num_param_updates = 0
        finished_episode_rewards = []
        errors = []
        for t in range(1,max_steps+1):
            episode_t += 1

            action_mask = model.valid_action_mask(obs)

            q_values = model.evaluate_q_values(obs, sess)
            q_values -= (1-action_mask)*1e8
            loop_preventer_mask, obs_token = loop_preventer.visit(obs)

            action_id = (q_values - (1 - loop_preventer_mask) * 1e8).argmax()
            random_action = (not loop_preventer_mask[q_values.argmax()])

            last_obs = obs
            action = model.action_id_to_action(action_id, obs)
            obs, rew, done, info = env.step(action)
            episode_reward += rew
            counter_bonus, counter_weight = counter.visit(obs)
            mod_rew = rew + counter_bonus # exploration bonus

            if random.random() < 0.01:
                print(counter_bonus, counter_weight)

            loop_preventer.record_action(obs_token, action_id, (obs == last_obs))

            buff.insert((last_obs, action_id, obs, mod_rew, done), counter_weight)
            if not random_action:
                debug_helper.insert(last_obs, q_values, action, rew)
            else:
                debug_helper.insert_random(last_obs, action, rew)
            debug_counter.visit(last_obs)

            if done or episode_t > max_steps_per_ep:
                finished_episode_rewards.append(episode_reward)

                print('============================================')
                print('iteration:', t)
                print('num updates:', num_param_updates)
                mean_rew = np.mean(finished_episode_rewards[-10:])
                print('mean reward:', mean_rew)
                print('error:',np.mean(errors[-100:]))
                print('score:', info['score'])
                print('latest reward:', finished_episode_rewards[-1:])

                unique_states, total_states = debug_counter.get_counts()
                debug_counter.reset()
                print('unique states visited:', unique_states, '/', total_states)

                if debug:
                    debug_helper.print_analysis()
                    buff.print_debug()

                obs = env.reset()
                counter.reset()
                debug_helper.reset()
                loop_preventer.reset()
                episode_t = 0
                episode_reward = 0

                if not path.exists('models'):
                    os.mkdir('models')
                saver.save(sess, 'models/model', global_step=t)

            if t % update_freq == 0:
                examples, example_idxs, example_weights = buff.sample(batch_size)
                error, example_errors = model.train_step(examples, example_weights, sess)
                errors.append(error)
                # TODO(nikita): consider recalculating counter weights for these
                # examples
                # buff.reprioritize(example_idxs, example_errors)
                num_param_updates += 1

            if num_param_updates % target_update_freq == 0:
                model.update_target(sess)

# %%

if __name__ == '__main__':
    main()
