from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

import time
import pickle
import argparse
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

parentdir = os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))
grandfatherdir = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, parentdir)
sys.path.insert(0, grandfatherdir)

from utils.query_util import tokenize
from utils.translator import Translator

from dmn.dmn_fasttext.config import Config
from dmn.dmn_fasttext.vector_helper import getVector
from dmn.dmn_fasttext.dmn_plus import DMN_PLUS

translator = Translator()


class DmnSession():
    def __init__(self, session, model, config, metadata, char=2):
        self.u = None
        self.r = None
        self.session = session
        self.model = model
        self.config = config
        self.idx2candid = metadata['idx2candid']
        self.w2idx = metadata['w2idx']
        self.max_sen_len = metadata['max_sen_len']
        self.max_input_len = metadata['max_input_len']
        # self.context = [[data_helper.ff_embedding_local(self.config.EMPTY) for _ in range(self.max_sen_len)]]
        self.context = [[getVector(self.config.EMPTY)
                         for _ in range(self.max_sen_len)]]
        self.context_raw = [[self.config.EMPTY]]
        self.char = char

    def append_memory(self, m):
        if not m:
            return
        m = translator.en2cn(m)
        m = tokenize(m, self.char)
        q_vector = m + \
                   [self.config.PAD for _ in range(
                       self.max_sen_len - len(m))]
        q_vector = [getVector(word) for word in q_vector]
        self.context.append(q_vector)
        self.context = self.context[-self.config.max_memory_size:]

    def clear_memory(self, history=0):
        if history == 0:
            # self.context = [[data_helper.ff_embedding_local(self.config.EMPTY) for _ in range(self.max_sen_len)]]
            self.context = [[getVector(self.config.EMPTY)
                             for _ in range(self.max_sen_len)]]
            # self.context = [[]]
        else:
            self.context = self.context[-history:]

    def reply(self, msg):
        line = msg.strip().lower()
        if line == 'clear':
            # self.context = [[data_helper.ff_embedding_local(self.config.EMPTY) for _ in range(self.max_sen_len)]]
            self.context = [[getVector(self.config.EMPTY)
                             for _ in range(self.max_sen_len)]]
            reply_msg = ['memory cleared!']
            top_prob = [0]
        else:
            inputs = []
            questions = []

            q = tokenize(line, self.char)
            q_len=len(q)
            q = q[:self.max_sen_len]
            self.context_raw.append(q)
            q_vector = q + \
                       [self.config.PAD for _ in range(
                               self.max_sen_len - len(q))]
            q_vector = [getVector(word) for word in q_vector]

            inp_vector = self.context
            pad_vector = [getVector(self.config.PAD)
                          for _ in range(self.max_sen_len)]
            inp_vector = inp_vector + \
                         [pad_vector for _ in range(
                                 self.max_input_len - len(inp_vector))]

            inputs.append(inp_vector)
            questions.append(q_vector)

            # with tf.Session() as session:
            #     session.run(tf.global_variables_initializer())
            #
            #     top_predict_proba = self.graph.get_tensor_by_name(
            #         'pred:0')
            #     qp = self.graph.get_tensor_by_name('questions:0')
            #     ql = self.graph.get_tensor_by_name('question_lens:0')
            #     ip = self.graph.get_tensor_by_name('inputs:0')
            #     il = self.graph.get_tensor_by_name('input_lens:0')
            #     dp = self.graph.get_tensor_by_name('dropout:0')
            #     output = session.run(top_predict_proba, feed_dict={
            #         qp: questions, ql: [self.max_sen_len], ip: inputs, il: [len(self.context)], dp: self.config.dropout})

            pred, top_prob = self.model.predict(self.session,
                                                inputs, [len(self.context)], self.max_sen_len, questions,
                                                [q_len])

            print('pred:', pred, top_prob)
            # indices = output.indices.tolist()[0]
            # values = output.values.tolist()[0]

            reply_msg = [self.idx2candid[ind] for ind in pred]
            r = reply_msg[0]
            r = translator.en2cn(r)
            r = tokenize(r, self.char)
            r = r[:self.max_sen_len]
            self.context_raw.append(r)
            r_vector = r + \
                       [self.config.PAD for _ in range(self.max_sen_len - len(r))]
            r_vector = [getVector(word) for word in r_vector]
            self.context.append(q_vector)
            self.context.append(r_vector)
            if len(self.context) > self.config.max_memory_size:
                self.context = self.context[-self.config.max_memory_size:]

        return reply_msg[0], top_prob[0]


class DmnInfer:
    def __init__(self):
        self.config = Config()
        with open(self.config.metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        self.model = self._load_model()
        self.session = tf.Session()

    def _load_model(self):
        self.config.train_mode = False
        model = DMN_PLUS(self.config, self.metadata)
        return model

    def get_session(self):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        self.session.run(init)

        # restore checkpoint
        ckpt = tf.train.get_checkpoint_state(self.config.ckpt_path)
        if ckpt and ckpt.model_checkpoint_path:
            print('\n>> restoring checkpoint from',
                  ckpt.model_checkpoint_path)
        saver.restore(self.session, ckpt.model_checkpoint_path)

        # saver = tf.train.import_meta_graph(
        #     self.config.ckpt_path + 'dmn.weights.meta')
        # graph = tf.get_default_graph()

        char = 2 if self.config.word else 1
        isess = DmnSession(self.session, self.model,
                           self.config, self.metadata, char)
        return isess


def main():
    di = DmnInfer()
    sess = di.get_session()

    query = ''
    while query != 'exit':
        query = input('>> ')
        print(sess.reply(query))


if __name__ == '__main__':
    main()
