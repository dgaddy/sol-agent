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

class Buffer(object):
    def __init__(self, size):
        self.size = size
        self.items = []
        self.index = 0
        self.stats = []

    def insert(self, item):
        if len(self.items) < self.size:
            self.items.append(item)
        else:
            self.items[self.index] = item
            self.index = (self.index + 1) % self.size

    def sample(self, k):
        # return random.sample(self.items, k)
        res = random.sample(self.items, k)
        while np.mean([0. if x[3] < 0 else 1. for x in res]) < 0.5:
            for i in reversed(range(k)):
                obs, act, next_obs, rew, done = res[i]
                if rew < 0:
                    del res[i]
            res.extend(random.sample(self.items, k - len(res)))

        return res

        # try:
        #     self.stats
        # except AttributeError:
        #     self.stats = []
        #
        # self.stats.append([0. if x[3] < 0 else 1. for x in res])
        # if len(self.stats) >= 100:
        #     print('Percent of valid actions sampled:', np.mean(self.stats))
        #     self.stats.clear()
        # return res


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
            for obs, action, reward in self.random_items:
                action_type = action[0]
                valid = reward >= 0

                counts_random[(action_type, valid)] += 1


        print('===========debug info===========')
        for at in sol_env.ActionType:
            print('invalid', at, counts[(at,False)], '/', counts[(at,True)] + counts[(at,False)])
        print('useless drags', n_useless_drags, '/', counts[(sol_env.ActionType.DRAG_DROP, True)])

        if self.random_items:
            for at in sol_env.ActionType:
                print('random invalid', at, counts_random[(at,False)], '/', counts_random[(at,True)] + counts_random[(at,False)])

        if False:
            prev_obs = None
            for obs, q_values, action, reward in self.items:
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

                # from_str = '{} [{}]'.format(' '.join(from_cards[-10:-card_id - 1]), ' '.join(from_cards[-card_id - 1:]))
                # dst_str = ' '.join(dst_cards[-10:])
                # print("illegal {} {}: {} -> {} {}: {}".format(from_slot, from_type, from_str, dst_slot, dst_type, dst_str))
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

        # TODO(nikita): this should really use the contextual embeddings from the LSTM,
        # not just the original inputs and the lstm output
        slot_hiddens = tf.layers.conv1d(tf.concat([slot_reps,context_repeated],2), hidden_size, 1, activation=tf.nn.relu)

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

        gamma = 0.8
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
        error = q_for_actions - (self.reward_ph + (1-self.done_mask_ph)*gamma*best_q_next)
        self.total_error = tf.reduce_mean(error * error)

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

        error, _ = session.run([self.total_error,self.train_fn], feed_dict=feed_dict)
        return error

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
        obs = env.reset()
        episode_t = 0
        for t in range(1,initial_samples+1):
            episode_t += 1
            action_mask = model.valid_action_mask(obs)
            action_id = (np.random.random(model.n_actions())*action_mask).argmax()
            last_obs = obs
            action = model.action_id_to_action(action_id, obs)
            obs, rew, done, info = env.step(action)

            buff.insert((last_obs, action_id, obs, rew, done))

            if done or episode_t > max_steps_per_ep:
                obs = env.reset()
                episode_t = 0

        with open(sample_file, 'wb') as f:
            pickle.dump(buff, f)
        return buff

class Counter:
    def __init__(self, n_actions):
        self.table = {}
        self.n_actions = n_actions

    def hash_obs(self, obs):
        return hash(tuple([(a, tuple(b)) for a,b in obs]))

    def visit(self, obs):
        """
        obs -> (bonus, is_uniform, obs_token)
        """
        h = self.hash_obs(obs)
        if h not in self.table:
            self.table[h] = np.ones(self.n_actions, dtype=int)
            uniform = True
        else:
            uniform = False

        entry = self.table[h]
        res = np.sqrt(np.log(np.sum(entry)) / entry) # UCB
        return res, uniform, h

    def record_action(self, obs_token, action_id):
        assert obs_token in self.table
        self.table[obs_token][action_id] += 1

    def reset(self):
        self.table = {}

class HashCounter:
    def __init__(self, n_actions, max_stack_len):
        self.num_cards = 64 # TODO: actual number is less
        self.num_hashes = self.num_cards + (self.num_cards * self.num_cards) + 1
        self.count_table = np.ones((self.num_hashes, 2), dtype=int)
        self.n_actions = n_actions
        self.max_stack_len = max_stack_len

    def encode_card(self, card):
        if card is None:
            return 60 # should probably be 14 * 4 + 1= 57
        face_down, suit, rank  = card
        if face_down:
            return 61

        return rank + 14 * suit

    def visit(self, obs):
        """
        obs -> (bonus, obs_token)
        """
        idxs = np.ones([self.n_actions], dtype=int) * (self.num_hashes - 1)

        n_slots = len(obs)
        max_stack_len = self.max_stack_len

        # (from, to, max_stack_len)
        for to_slot, (to_slot_type, to_cards) in enumerate(obs):
            to_enc = self.encode_card(to_cards[-1] if to_cards else None)
            to_enc *= self.num_cards

            next_cards = obs[(to_slot + 1) % n_slots][1]
            idxs[to_slot] = self.encode_card(next_cards[-1] if next_cards else None)
            for from_slot, (from_slot_type, from_cards) in enumerate(obs):
                for stack_idx, card in enumerate(reversed(from_cards)):
                    idxs[from_slot * (max_stack_len * n_slots) + to_slot * (max_stack_len) + stack_idx
                        + n_slots] = self.num_cards + self.encode_card(card) + to_enc

        entries = self.count_table[idxs]
        res = np.sqrt(np.log(entries[:,1]) / entries[:, 0])

        self.count_table[np.unique(idxs)] += [0, 1]
        # self.count_table[idxs] += [0, 1]
        return res, idxs

    def record_action(self, obs_token, action_id):
        self.count_table[obs_token[action_id], 0] += 1

    def reset(self):
        assert np.all(self.count_table[:, 0] <= self.count_table[:, 1])
        pass
        # self.count_table[:,:] = 1


def main():
    debug = True
    update_freq = 4
    batch_size = 32
    max_steps = 5000000
    max_steps_per_ep = 10000
    buffer_size = 1000000
    # init_eps = 0.9
    # final_eps = 0.1
    init_eps = 0.0
    final_eps = 0.0
    final_eps_timestep = 1000000
    target_update_freq = 10000 #5000 # 10000
    max_stack_len = 10

    init_count_factor = 1.0
    final_count_factor = 0.0
    final_count_timestep = 1500000

    # factor of 4 = bonus decreases by ~0.2 after the first time an action is tried
    # count_factor = 0.2 # 1.
    count_factor = 0.2

    env = sol_env.SolEnv('toy-klondike')
    # NOTE(nikita): I'm enabling unlimited redeals because it makes the observations
    # more Markovian, just to make sure that running out of cards from the deck
    # isn't what's holding up learning
    env.change_options({'Unlimited redeals': True})
    obs = env.reset()
    model = Model(len(obs),max_stack_len)
    counter = Counter(model.n_actions())
    # counter = HashCounter(model.n_actions(), model.max_stack_len)
    buff = get_initial_sample(buffer_size, env, model, max_steps_per_ep)

    debug_helper = DebugHelper(len(obs), max_stack_len)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        raw_env = env
        env = gym.wrappers.Monitor(env, os.path.join('/tmp/sol_vid', "gym"), force=True)
        obs = env.reset()
        tf.global_variables_initializer().run()

        episode_t = 0
        episode_reward = 0
        num_param_updates = 0
        finished_episode_rewards = []
        errors = []
        for t in range(1,max_steps+1):
            episode_t += 1

            if t >= final_eps_timestep:
                eps = final_eps
            else:
                eps = init_eps + (final_eps-init_eps)*t/final_eps_timestep

            if t >= final_count_timestep:
                count_factor = final_count_factor
            else:
                count_factor = init_count_factor + (final_count_factor-init_count_factor)*t/final_count_timestep

            action_mask = model.valid_action_mask(obs)

            # count_bonuses, obs_token = counter.visit(obs)
            # counts_uniform = True
            count_bonuses, counts_uniform, obs_token = counter.visit(obs)
            # if random.random() < 0.01:
            #     cb = count_bonuses[action_mask > 0]
            #     print(np.max(cb) - np.mean(cb),
            #         cb[:2] - np.mean(cb), cb[-50:] - np.mean(cb))

            # XXX(nikita): the probabilities assigned to the different strategies
            # are probably suboptimal
            if random.random() < eps:
                random_action = True
                action_id = (np.random.random(model.n_actions())*action_mask).argmax()
                # if random.random() < 0.8:
                #     possible_actions = raw_env.get_hint_actions()
                # else:
                #     possible_actions = []
                #
                # if not possible_actions:
                #     action_id = (np.random.random(model.n_actions())*action_mask).argmax()
                # else:
                #     action = random.choice(possible_actions)
                #     action_id = model.action_to_action_id(action)
            else:
                q_values = model.evaluate_q_values(obs, sess)
                q_values -= (1-action_mask)*1e8

                # valid_qs = q_values[action_mask > 0]
                # qstat1 = np.mean(valid_qs), np.std(valid_qs)

                q_values += count_factor * count_bonuses

                # valid_qs = count_factor * count_bonuses[action_mask > 0]
                # qstat2 = np.mean(valid_qs), np.std(valid_qs)
                #
                # if random.random() < 0.1:
                #     print(qstat1, qstat2)

                # action_id = (q_values+np.random.randn(*q_values.shape)*.01).argmax()
                action_id = q_values.argmax()
                # random_action = False
                random_action = not counts_uniform

            last_obs = obs
            counter.record_action(obs_token, action_id)
            action = model.action_id_to_action(action_id, obs)
            obs, rew, done, info = env.step(action)
            episode_reward += rew

            buff.insert((last_obs, action_id, obs, rew, done))
            if not random_action:
                debug_helper.insert(last_obs, q_values, action, rew)
            else:
                debug_helper.insert_random(last_obs, action, rew)

            if done or episode_t > max_steps_per_ep:
                # finished_episode_rewards.append(episode_reward)
                obs = env.reset()
                counter.reset()
                finished_episode_rewards.append(episode_reward)
                episode_t = 0
                episode_reward = 0

            if t % update_freq == 0:
                error = model.train_step(buff.sample(batch_size), sess)
                errors.append(error)
                num_param_updates += 1

            if num_param_updates % target_update_freq == 0:
                model.update_target(sess)

            if t % 1000 == 0:
                print('============================================')
                print('iteration:', t)
                print('num updates:', num_param_updates)
                print('epsilon greedy:', eps)
                mean_rew = np.mean(finished_episode_rewards[-10:])
                print('mean reward:', mean_rew)
                print('error:',np.mean(errors[-100:]))
                print('count factor:', count_factor)
                print('score:', info['score'])
                print('latest reward:', finished_episode_rewards[-1:])

                if debug:
                    debug_helper.print_analysis()

                if not path.exists('models'):
                    os.mkdir('models')
                saver.save(sess, 'models/model', global_step=t)

# %%

if __name__ == '__main__':
    main()
