import os
import time
import pickle
import json

import numpy as np
from copy import deepcopy
from tqdm import tqdm
import tensorflow as tf
# from dmn.attention_gru_cell import AttentionGRUCell
from dmn.attention_gru_cell import AttentionGRUCell
from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops

# from dmn.dmn_fasttext.config import Config
from dmn.dmn_fasttext.config import Config
# config = Config()


def _add_gradient_noise(t, stddev=1e-3, name=None):
    """Adds gradient noise as described in http://arxiv.org/abs/1511.06807
    The input Tensor `t` should be a gradient.
    The output will be `t` + gaussian noise.
    0.001 was said to be a good fixed value for memory networks."""
    # with tf.op_scope([t, stddev], name, "add_gradient_noise") as name:
    with tf.name_scope(name, "add_gradient_noise", [t, stddev]) as name:
        t = tf.convert_to_tensor(t, name="t")
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name=name)

# from https://github.com/domluna/memn2n


def _position_encoding(sentence_size, embedding_size):
    """Position encoding described in section 4.1 in "End to End Memory Networks" (http://arxiv.org/pdf/1503.08895v5.pdf)"""
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size + 1
    le = embedding_size + 1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i - 1, j - 1] = (i - (le - 1) / 2) * (j - (ls - 1) / 2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    return np.transpose(encoding)


class DMN_PLUS(object):
    """FUCKING DMN_PLUS MODEL"""

    def _load_metadata(self):
        self.max_input_len = self.metadata['max_input_len']
        self.max_q_len = self.metadata['max_q_len']
        self.max_sen_len = self.metadata['max_sen_len']
        self.max_mask_len = self.metadata['max_mask_len']
        self.num_supporting_facts = self.metadata['num_supporting_facts']
        self.candidate_size = self.metadata['candidate_size']
        self.candid2idx = self.metadata['candid2idx']
        self.idx2candid = self.metadata['idx2candid']
        self.w2idx = self.metadata['w2idx']
        self.idx2w = self.metadata['idx2w']
        self.vocab_size = self.metadata['vocab_size']

    def _create_placeholders(self):
        """add data and placeholder to graph"""

        if self.config.word:
            self.question_placeholder = tf.placeholder(
                tf.float32, shape=(None, self.max_q_len, self.config.embed_size), name='questions')
            # self.question_placeholder = tf.placeholder(
            #         tf.float32, shape=(None, None, None), name='questions')
            self.question_len_placeholder = tf.placeholder(
                tf.int32, shape=(None,), name='question_lens')

            self.input_placeholder = tf.placeholder(tf.float32, shape=(
                None, self.max_input_len, self.max_sen_len, self.config.embed_size), name='inputs')
            # self.input_placeholder = tf.placeholder(tf.float32, shape=(
            #     None, None, None, self.config.embed_size), name='inputs')
            self.input_len_placeholder = tf.placeholder(
                tf.int32, shape=(None,), name='input_lens')

            self.embeddings = None
        else:
            self.question_placeholder = tf.placeholder(
                tf.int32, shape=(None, self.max_q_len), name='questions')
            self.question_len_placeholder = tf.placeholder(
                tf.int32, shape=(None,), name='question_lens')

            self.input_placeholder = tf.placeholder(tf.int32, shape=(
                None, self.max_input_len, self.max_sen_len), name='inputs')
            self.input_len_placeholder = tf.placeholder(
                tf.int32, shape=(None,), name='input_lens')

            self.embeddings = tf.Variable(tf.random_uniform(
                [self.vocab_size, self.config.embed_size], -1.0, 1.0),
                trainable=True, dtype=tf.float32, name='embeddings')
            self.embedding_placeholder = tf.placeholder(
                tf.float32, [None, None])
            self.embedding_init = self.embeddings.assign(
                self.embedding_placeholder)

        if self.config.multi_label:
            self.answer_placeholder = tf.placeholder(
                tf.float32, shape=(None, self.candidate_size), name='answers')
        else:
            self.answer_placeholder = tf.placeholder(
                tf.int32, shape=(None,), name='answers')

        self.rel_label_placeholder = tf.placeholder(tf.int32, shape=(
            None, self.num_supporting_facts), name='rel_labels')
        self.dropout_placeholder = tf.placeholder(tf.float32, name='dropout')

    def _create_loss(self, output):
        """Calculate loss"""
        # optional strong supervision of attention with supporting facts
        gate_loss = 0
        if self.config.strong_supervision:
            for i, att in enumerate(self.attentions):
                labels = tf.gather(tf.transpose(self.rel_label_placeholder), 0)
                gate_loss += tf.reduce_sum(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=att, labels=labels))

        # loss = self.config.beta * tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
        #     logits=output, labels=self.answer_placeholder)) + gate_loss
        if self.config.multi_label:
            # multi targets
            loss = self.config.beta * tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=output, labels=self.answer_placeholder)) + gate_loss
        else:
            loss = self.config.beta * tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=output, labels=self.answer_placeholder)) + gate_loss

        # stackoverflow :https://stackoverflow.com/questions/33712178/tensorflow-nan-bug
        # cross_entropy = - tf.reduce_sum(self.answer_placeholder *
        #                                 tf.log(tf.clip_by_value(output, 1e-10, 1.0)))
        # loss = self.config.beta * tf.reduce_sum(cross_entropy) + gate_loss

        # add l2 regularization for all variables except biases
        for v in tf.trainable_variables():
            if not 'bias' in v.name.lower():
                loss += self.config.l2 * tf.nn.l2_loss(v)

        # tf.summary.scalar('loss', loss)

        return loss

    def _create_training_op(self, loss):
        """Calculate and apply gradients"""
        opt = tf.train.AdamOptimizer(
            learning_rate=self.config.lr, epsilon=1e-8)
        gvs = opt.compute_gradients(loss)

        # optionally cap and noise gradients to regularize
        if self.config.cap_grads:
            gvs = [(tf.clip_by_norm(grad, self.config.max_grad_val), var)
                   for grad, var in gvs]
        if self.config.noisy_grads:
            gvs = [(_add_gradient_noise(grad), var) for grad, var in gvs]

        train_op = opt.apply_gradients(gvs)
        return train_op

    def get_question_representation(self, embeddings):
        """Get question vectors via embedding and GRU"""
        if embeddings:
            questions = tf.nn.embedding_lookup(
                embeddings, self.question_placeholder)
        else:
            questions = self.question_placeholder

        gru_cell = tf.contrib.rnn.GRUCell(self.config.hidden_size)
        _, q_vec = tf.nn.dynamic_rnn(gru_cell,
                                     questions,
                                     dtype=np.float32,
                                     sequence_length=self.question_len_placeholder
                                     )
        return q_vec

    def get_input_representation(self, embeddings):
        """Get fact (sentence) vectors via embedding, positional encoding and bi-directional GRU"""
        # get word vectors from embedding
        if embeddings:
            inputs = tf.nn.embedding_lookup(embeddings, self.input_placeholder)
        else:
            inputs = self.input_placeholder

        # use encoding to get sentence representation
        inputs = tf.reduce_sum(inputs * self.encoding, 2)

        forward_gru_cell = tf.contrib.rnn.GRUCell(self.config.hidden_size)
        backward_gru_cell = tf.contrib.rnn.GRUCell(self.config.hidden_size)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            forward_gru_cell,
            backward_gru_cell,
            inputs,
            dtype=np.float32,
            sequence_length=self.input_len_placeholder
        )

        # f<-> = f-> + f<-
        fact_vecs = tf.reduce_sum(tf.stack(outputs), axis=0)
        # apply dropout
        fact_vecs = tf.nn.dropout(fact_vecs, self.dropout_placeholder)

        return fact_vecs

    def get_attention(self, q_vec, prev_memory, fact_vec, reuse):
        """Use question vector and previous memory to create scalar attention for current fact"""
        with tf.variable_scope("attention", reuse=reuse):
            features = [fact_vec * q_vec,
                        fact_vec * prev_memory,
                        tf.abs(fact_vec - q_vec),
                        tf.abs(fact_vec - prev_memory)]

            feature_vec = tf.concat(features, 1)

            attention = tf.contrib.layers.fully_connected(feature_vec,
                                                          self.config.embed_size,
                                                          activation_fn=tf.nn.tanh,
                                                          reuse=reuse, scope="fc1")

            attention = tf.contrib.layers.fully_connected(attention,
                                                          1,
                                                          activation_fn=None,
                                                          reuse=reuse, scope="fc2")

        return attention

    def generate_episode(self, memory, q_vec, fact_vecs, hop_index):
        """Generate episode by applying attention to current fact vectors through a modified GRU"""

        attentions = [tf.squeeze(
            self.get_attention(q_vec, memory, fv, bool(hop_index) or bool(i)), axis=1)
            for i, fv in enumerate(tf.unstack(fact_vecs, axis=1))]

        attentions = tf.transpose(tf.stack(attentions))
        self.attentions.append(attentions)
        attentions = tf.nn.softmax(attentions)
        attentions = tf.expand_dims(attentions, axis=-1)

        reuse = True if hop_index > 0 else False

        # concatenate fact vectors and attentions for input into attGRU
        gru_inputs = tf.concat([fact_vecs, attentions], 2)

        with tf.variable_scope('attention_gru', reuse=reuse):
            _, episode = tf.nn.dynamic_rnn(AttentionGRUCell(self.config.hidden_size),
                                           gru_inputs,
                                           dtype=np.float32,
                                           sequence_length=self.input_len_placeholder
                                           )

        return episode

    def add_answer_module(self, rnn_output, q_vec):
        """Linear softmax answer module"""

        rnn_output = tf.nn.dropout(rnn_output, self.dropout_placeholder)

        output = tf.layers.dense(tf.concat([rnn_output, q_vec], 1),
                                 self.candidate_size,
                                 activation=None)

        return output

    def _inference(self):
        """Performs inference on the DMN model"""
        # input fusion module
        with tf.variable_scope("question", initializer=tf.contrib.layers.xavier_initializer()):
            print('==> get question representation')
            q_vec = self.get_question_representation(self.embeddings)

        with tf.variable_scope("input", initializer=tf.contrib.layers.xavier_initializer()):
            print('==> get input representation')
            fact_vecs = self.get_input_representation(self.embeddings)

        # keep track of attentions for possible strong supervision
        self.attentions = []

        # memory module
        with tf.variable_scope("memory", initializer=tf.contrib.layers.xavier_initializer()):
            print('==> build episodic memory')

            # generate n_hops episodes
            prev_memory = q_vec

            for i in range(self.config.num_hops):
                # get a new episode
                print('==> generating episode', i)
                episode = self.generate_episode(
                    prev_memory, q_vec, fact_vecs, i)

                # untied weights for memory update
                with tf.variable_scope("hop_%d" % i):
                    prev_memory = tf.layers.dense(tf.concat([prev_memory, episode, q_vec], 1),
                                                  self.config.hidden_size,
                                                  activation=tf.nn.relu)

            output = prev_memory

        # pass memory module output through linear answer module
        with tf.variable_scope("answer", initializer=tf.contrib.layers.xavier_initializer()):
            output = self.add_answer_module(output, q_vec)

        return output

    def get_predictions(self, output):
        if self.config.multi_label:
            # multi targets
            predict_by_value = tf.nn.top_k(
                output, k=self.config.top_k, name="predict_op")
            predict_proba_op = tf.nn.softmax(output, name="predict_proba_op")
            pred = tf.nn.top_k(
                predict_proba_op, k=self.config.top_k, name="top_predict_proba_op")
        else:
            predict_proba_op = tf.nn.softmax(output)
            predict_proba_top_op = tf.nn.top_k(
                predict_proba_op, k=self.config.top_k, name='top_predict_proba_op')
            pred = tf.argmax(predict_proba_op, 1, name='pred')

        # predict_proba_op = tf.nn.softmax(output, name="predict_proba_op")
        # preds = tf.nn.top_k(
        #     predict_proba_op, k=self.config.top_k, name="top_predict_proba_op")
        return pred, predict_proba_top_op

    def predict(self, session, inputs, input_lens, max_sen_len, questions, q_lens):
        feed = {
            self.question_placeholder: questions,
            self.input_placeholder: inputs,
            self.question_len_placeholder: q_lens,
            self.input_len_placeholder: input_lens,
            self.dropout_placeholder: self.config.dropout
        }
        pred, prob_top = session.run(
            [self.pred, self.prob_top_k], feed_dict=feed)
        return pred, prob_top

    def __init__(self, config, metadata):
        self.config = config
        self.metadata = metadata
        self._load_metadata()
        self.encoding = _position_encoding(
            self.max_sen_len, self.config.embed_size)
        self._create_placeholders()
        self.output = self._inference()
        self.pred, self.prob_top_k = self.get_predictions(self.output)
        # self.pred = self.get_predictions(self.output)
        self.calculate_loss = self._create_loss(self.output)
        self.train_step = self._create_training_op(self.calculate_loss)
        # self.merged = tf.summary.merge_all()
