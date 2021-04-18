'''
IADAM-Attention-V4-2 model which is the IART model in the paper.
Developed based on DAM model

@author: Liu Yang (yangliuyx@gmail.com / lyang@cs.umass.edu)
@homepage: https://sites.google.com/site/lyangwww/
'''

import tensorflow as tf
import numpy as np
import cPickle as pickle

import utils.layers as layers
import utils.operations as op

class Net(object):
    '''Add positional encoding(initializer lambda is 0),
       cross-attention, cnn integrated and grad clip by value.

    Attributes:
        conf: a configuration paramaters dict
        word_embedding_init: a 2-d array with shape [vocab_size+1, emb_size]
        there is one dimension in vocab_size which is corresponding to _eos_.
        in our preprocessing, _eos_ is always the last dimension
        +1 to add one more embedding vector for padding and masking
        We add an "all 0" vector in the 0-th row of word_embedding_init in order
        to denote the padding word
        when call tf.nn.embedding_lookup(), if word_id = 0, then this is a paded
        word; if word_id > 0 (from 1 to vocab_size), then this is a real word
    '''

    def __init__(self, conf):
        self._graph = tf.Graph()
        self._conf = conf

        if self._conf['word_emb_init'] is not None:
            print('loading word emb init')
            self._word_embedding_init = pickle.load(
                open(self._conf['word_emb_init'], 'rb'))
        else:
            self._word_embedding_init = None

    def build_graph(self):
        with self._graph.as_default():
            if self._conf['rand_seed'] is not None:
                rand_seed = self._conf['rand_seed']
                tf.set_random_seed(rand_seed)
                print('set tf random seed: %s' % self._conf['rand_seed'])

            # word embedding
            if self._word_embedding_init is not None:
                word_embedding_initializer = tf.constant_initializer(
                    self._word_embedding_init)
            else:
                word_embedding_initializer = tf.random_normal_initializer(
                    stddev=0.1)

            self._word_embedding = tf.get_variable(
                name='word_embedding',
                shape=[self._conf['vocab_size'] + 1, self._conf['emb_size']],
                dtype=tf.float32,
                initializer=word_embedding_initializer)

            # define placehloders
            self.turns = tf.placeholder(
                tf.int32,
                shape=[self._conf["batch_size"], self._conf["max_turn_num"],
                       self._conf["max_turn_len"]])

            self.tt_turns_len = tf.placeholder(  # turn_num
                tf.int32,
                shape=[self._conf["batch_size"]])

            self.every_turn_len = tf.placeholder(
                tf.int32,
                shape=[self._conf["batch_size"], self._conf["max_turn_num"]])

            self.turns_intent = tf.placeholder(
                tf.float32,
                shape=[self._conf["batch_size"], self._conf["max_turn_num"],
                       self._conf["intent_size"]])

            self.response = tf.placeholder(
                tf.int32,
                shape=[self._conf["batch_size"], self._conf["max_turn_len"]])

            self.response_len = tf.placeholder(
                tf.int32,
                shape=[self._conf["batch_size"]])

            self.response_intent = tf.placeholder(
                tf.float32,
                shape=[self._conf["batch_size"], self._conf["intent_size"]])

            self.label = tf.placeholder(
                tf.float32,
                shape=[self._conf["batch_size"]])

            # define operations
            # response part
            Hr = tf.nn.embedding_lookup(self._word_embedding, self.response)
            # [batch_size, max_turn_len, embed_size]

            # print('[after embedding_lookup] Hr shape: %s' % Hr.shape)

            if self._conf['is_positional'] and self._conf['stack_num'] > 0:
                with tf.variable_scope('positional'):
                    Hr = op.positional_encoding_vector(Hr, max_timescale=10)
            Hr_stack = [Hr]  # 1st element of Hr_stack is the orginal embedding
            # lyang comments: self attention
            for index in range(self._conf['stack_num']):
                # print('[self attention for response] stack index: %d ' % index)
                with tf.variable_scope('self_stack_' + str(index)):
                    # [batch, max_turn_len, emb_size]
                    Hr = layers.block(  # attentive module
                        Hr, Hr, Hr,
                        Q_lengths=self.response_len,
                        K_lengths=self.response_len)
                    # print('[after layers.block] Hr shape: %s' % Hr.shape)
                    # Hr is still [batch_size, max_turn_len, embed_size]
                    Hr_stack.append(Hr)

            # print('[after self attention of response] len(Hr_stack)',
            #       len(Hr_stack))  # 1+stack_num
            # context part
            # a list of length max_turn_num, every element is a tensor with shape [batch, max_turn_len]
            list_turn_t = tf.unstack(self.turns, axis=1)
            list_turn_length = tf.unstack(self.every_turn_len, axis=1)
            list_turn_intent = tf.unstack(self.turns_intent, axis=1)

            sim_turns = []
            attention_turns = [] # intent based attention on each turn
            # for every turn_t calculate matching vector
            turn_index = 0
            for turn_t, t_turn_length, t_intent in zip(list_turn_t, list_turn_length, list_turn_intent):
                print('current turn_index : ', turn_index)
                turn_index += 1
                Hu = tf.nn.embedding_lookup(self._word_embedding,
                                            turn_t)  # [batch, max_turn_len, emb_size]
                # print('[after embedding_lookup] Hu shape: %s' % Hu.shape)

                if self._conf['is_positional'] and self._conf['stack_num'] > 0:
                    with tf.variable_scope('positional', reuse=True):
                        Hu = op.positional_encoding_vector(Hu,
                                                           max_timescale=10)
                Hu_stack = [Hu]  # 1st element of Hu_stack is the orginal embedding

                # lyang comments: self attention
                for index in range(self._conf['stack_num']):
                    # print('[self attention for context turn] stack index: %d ' % index)
                    with tf.variable_scope('self_stack_' + str(index),
                                           reuse=True):
                        # [batch, max_turn_len, emb_size]
                        Hu = layers.block(  # attentive module
                            Hu, Hu, Hu,
                            Q_lengths=t_turn_length, K_lengths=t_turn_length)
                        # print('[after layers.block] Hu shape: %s' % Hu.shape)
                        Hu_stack.append(Hu)
                # print('[after self attention of context turn] len(Hu_stack)',
                #       len(Hu_stack))  # 1+stack_num

                # lyang comments: cross attention
                # print('[cross attention ...]')
                r_a_t_stack = []
                t_a_r_stack = []
                # cross attention
                for index in range(self._conf['stack_num'] + 1):
                    # print('[cross attention] stack index = ', index)
                    with tf.variable_scope('t_attend_r_' + str(index)):
                        try:
                            # [batch, max_turn_len, emb_size]
                            t_a_r = layers.block(  # attentive module
                                Hu_stack[index], Hr_stack[index],
                                Hr_stack[index],
                                Q_lengths=t_turn_length,
                                K_lengths=self.response_len)
                        except ValueError:
                            tf.get_variable_scope().reuse_variables()
                            t_a_r = layers.block(
                                # [batch, max_turn_len, emb_size]
                                Hu_stack[index], Hr_stack[index],
                                Hr_stack[index],
                                Q_lengths=t_turn_length,
                                K_lengths=self.response_len)
                        # print('[cross attention t_attend_r_] stack index: %d, t_a_r.shape: %s' % (
                        #         index, t_a_r.shape))

                    with tf.variable_scope('r_attend_t_' + str(index)):
                        try:
                            # [batch, max_turn_len, emb_size]
                            r_a_t = layers.block(  # attentive module
                                Hr_stack[index], Hu_stack[index],
                                Hu_stack[index],
                                Q_lengths=self.response_len,
                                K_lengths=t_turn_length)
                        except ValueError:
                            tf.get_variable_scope().reuse_variables()
                            r_a_t = layers.block(
                                Hr_stack[index], Hu_stack[index],
                                Hu_stack[index],
                                Q_lengths=self.response_len,
                                K_lengths=t_turn_length)
                        # print('[cross attention r_a_t_] stack index: %d, r_a_t.shape: %s' % (
                        #         index, r_a_t.shape))

                    t_a_r_stack.append(t_a_r)
                    r_a_t_stack.append(r_a_t)
                    # print('[cross attention] len(t_a_r_stack):', len(t_a_r_stack))
                    # print('[cross attention] len(r_a_t_stack):', len(r_a_t_stack))

                # print('[before extend] len(t_a_r_stack):', len(t_a_r_stack))
                # print('[before extend] len(r_a_t_stack):', len(r_a_t_stack))
                # lyang comments: 3D aggregation
                t_a_r_stack.extend(
                    Hu_stack)  # half from self-attention; half from cross-attention
                r_a_t_stack.extend(
                    Hr_stack)  # half from self-attention; half from cross-attention
                # after extend, len(t_a_r_stack)) = 2*(stack_num+1)

                # print('[after extend] len(t_a_r_stack):', len(t_a_r_stack))
                # print('[after extend] len(r_a_t_stack):', len(r_a_t_stack))

                t_a_r = tf.stack(t_a_r_stack, axis=-1)
                r_a_t = tf.stack(r_a_t_stack, axis=-1)

                # print('after stack along the last dimension: ')
                # print('t_a_r shape: %s' % t_a_r.shape)
                # print('r_a_t shape: %s' % r_a_t.shape)
                # after stack, t_a_r and r_a_t are (batch, max_turn_len, embed_size, 2*(stack_num+1))

                with tf.variable_scope('intent_based_attention',
                                       reuse=tf.AUTO_REUSE): # share parameter across different turns
                    # there are 3 different ways to implement intent based attention
                    # implement these three different variations and compare the
                    # effectiveness as model abalation analysis
                    # let I_u_t and I_r_k are intent vector in [12,1]
                    # 1. dot: w * [I_u_t, I_r_k], where w is [24,1]
                    # 2. biliear: I_u_t' * w * I_r_k, where w is [12,12]
                    # 3. outprod: I_u_t * I_r_k' -> [12,12] out product ->
                    #             flaten to [144,1] outprod -> w*outprod
                    #             where w is [1,144]
                    attention_logits = layers.attention_intent(t_intent,
                                        self.response_intent,
                                        self._conf['intent_attention_type'])
                    # print('[intent_based_attention] attention_logits.shape: %s' % attention_logits.shape)
                    attention_turns.append(attention_logits)

                    # calculate similarity matrix
                with tf.variable_scope('similarity'):
                    # sim shape [batch, max_turn_len, max_turn_len, 2*(stack_num+1)]
                    # divide sqrt(200) to prevent gradient explosion
                    # A_biks * B_bjks -> C_bijs
                    sim = tf.einsum('biks,bjks->bijs', t_a_r, r_a_t) / tf.sqrt(
                        200.0)
                    # (batch, max_turn_len, embed_size, 2*(stack_num+1)) *
                    # (batch, max_turn_len, embed_size, 2*(stack_num+1)) ->
                    # [batch, max_turn_len, max_turn_len, 2*(stack_num+1)]
                    # where k is corresponding to the dimension of embed_size,
                    # which can be eliminated by dot product with einsum
                    # print('[similarity] after einsum dot prod sim shape: %s' % sim.shape)
                    # [batch, max_turn_len, max_turn_len, 2*(stack_num+1)]
                    # ! Here we multipy sim by intent based attention weights before
                    # append sim into sim_turns in order to generate the weighted
                    # stack in the next step

                sim_turns.append(sim)
                # print('[similarity] after append, len(sim_turns):', len(sim_turns))

            attention_logits = tf.stack(attention_turns, axis=1) # [batch, max_turn_num]
            print('[attention_logits] after stack attention_logits.shape: %s' % attention_logits.shape)
            # add mask in attention following the way in BERT
            # real turn_num is in self.tt_turns_len [batch]
            # return a mask tensor with shape [batch,  conf['max_turn_num']]
            attention_mask = tf.sequence_mask(self.tt_turns_len, self._conf['max_turn_num'],
                                              dtype=tf.float32)
            print('[attention_mask] attention_mask.shape: %s' % attention_mask.shape)
            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            adder = (1.0 - attention_mask) * -10000.0

            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_logits += adder
            attention = tf.nn.softmax(attention_logits) # by default softmax along dim=-1 [batch, max_turn_num]
            print('[attention] attention.shape: %s' % attention_mask.shape)
            self.attention = attention # will print it for visualization

            # cnn and aggregation
            # lyang comments aggregation by 3D CNN layer
            # [3d cnn aggregation] sim shape: (32, 9, 180, 180, 10)
            # conv_0 shape: (32, 9, 180, 180, 16)
            # pooling_0 shape: (32, 3, 60, 60, 16)
            # conv_1 shape: (32, 3, 60, 60, 16)
            # pooling_1 shape: (32, 1, 20, 20, 16)
            # [3d cnn aggregation] final_info: (32, 6400) # [batch * feature_size]
            # [batch, max_turn_num, max_turn_len, max_turn_len, 2*(stack_num+1)]
            # (32, 9, 180, 180, 10)
            sim = tf.stack(sim_turns, axis=1)
            # multipy sim by attention score
            sim = tf.einsum('bijks,bi->bijks', sim, attention)
            print('[3d cnn aggregation] sim shape: %s' % sim.shape)
            with tf.variable_scope('cnn_aggregation'):
                final_info = layers.CNN_3d(sim, self._conf['cnn_3d_oc0'],
                                           self._conf['cnn_3d_oc1'])
                # for udc
                # final_info = layers.CNN_3d(sim, 32, 16)
                # for douban
                # final_info = layers.CNN_3d(sim, 16, 16)

            print('[3d cnn aggregation] final_info: %s' % final_info.shape)
            # loss and train
            with tf.variable_scope('loss'):
                self.loss, self.logits = layers.loss(final_info, self.label)

                self.global_step = tf.Variable(0, trainable=False)
                initial_learning_rate = self._conf['learning_rate']
                self.learning_rate = tf.train.exponential_decay(
                    initial_learning_rate,
                    global_step=self.global_step,
                    decay_steps=400,
                    decay_rate=0.9,
                    staircase=True)

                Optimizer = tf.train.AdamOptimizer(self.learning_rate)
                self.optimizer = Optimizer.minimize(
                    self.loss,
                    global_step=self.global_step)

                self.init = tf.global_variables_initializer()
                self.saver = tf.train.Saver(
                    max_to_keep=self._conf["max_to_keep"])
                self.all_variables = tf.global_variables()
                self.all_operations = self._graph.get_operations()
                self.grads_and_vars = Optimizer.compute_gradients(self.loss)

                for grad, var in self.grads_and_vars:
                    if grad is None:
                        print var

                self.capped_gvs = [(tf.clip_by_value(grad, -1, 1), var) for
                                   grad, var in self.grads_and_vars]
                self.g_updates = Optimizer.apply_gradients(
                    self.capped_gvs,
                    global_step=self.global_step)

        return self._graph


