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

import dmn.dmn_data_utils2 as dmn_data_utils
from dmn.dmn_plus2 import Config, DMN_PLUS

translator = Translator()


class DmnSession():
    def __init__(self, session, model, config, char=2):
        self.context = [['此', '乃', '空', '文']]
        self.u = None
        self.r = None
        self.model = model
        self.config = config
        self.session = session
        self.idx2candid = self.model.idx2candid
        self.w2idx = self.model.w2idx
        self.char = char

    def append_memory(self, m):
        if not m:
            return
        m = translator.en2cn(m)
        m = tokenize(m, self.char)
        self.context.append(m)

    def clear_memory(self, history=0):
        if history == 0:
            self.context = [['此', '乃', '空', '文']]
            # self.context = [[]]
        else:
            self.context = self.context[-history:]

    def reply(self, msg):
        line = msg.strip().lower()
        if line == 'clear':
            self.context = []
            reply_msg = ['memory cleared!']
            values = [0]
        else:
            inputs = []
            questions = []

            q = tokenize(line, self.char)
            q_vector = [self.w2idx.get(w, 0) for w in q]
            print('q_vector:', q_vector)
            inp_vector = [[self.w2idx.get(w, 0) for w in s]
                          for s in self.context]

            inputs.append(inp_vector)
            questions.append(np.vstack(q_vector).astype(np.float32))

            input_lens, sen_lens, max_sen_len = dmn_data_utils.get_sentence_lens(
                    inputs)

            q_lens = dmn_data_utils.get_lens(questions)

            max_input_len = self.model.max_input_len
            max_sen_len = self.model.max_sen_len
            max_q_len = self.model.max_q_len

            inputs = dmn_data_utils.pad_inputs(inputs, input_lens, max_input_len,
                                               "split_sentences", sen_lens, max_sen_len)

            inputs = np.asarray(inputs)

            questions = dmn_data_utils.pad_inputs(
                    questions, q_lens, max_q_len)
            questions = np.asarray(questions)

            preds = self.model.predict(self.session,
                                       inputs, input_lens, max_sen_len, questions, q_lens)

            # print('preds:', preds)
            # if self.config.multi_label:
            indices = preds[0].indices.tolist()[0]
            values = preds[0].values.tolist()[0]
            # else:
            #     indices = preds[1].tolist()[0]
            #     values = preds[0].tolist()[0]
            # print('indices:{0},values:{1}'.format(indices, values))
            reply_msg = [self.idx2candid[ind] for ind in indices]
            # print(reply_msg)
            r = reply_msg[0]
            # print('r:',r)
            r = translator.en2cn(r)
            r = tokenize(r, self.char)
            self.context.append(r)

        if self.config.multi_label:
            return reply_msg, values
        else:
            return reply_msg[0], values[0]


class DmnInfer:
    def __init__(self):
        self.config = Config()
        self.model = self._load_model()
        self.session = tf.Session()

    def _load_model(self):
        self.config.train_mode = False
        model = DMN_PLUS(self.config)
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

        char = 2 if self.config.word2vec_init else 1
        isess = DmnSession(self.session, self.model, self.config, char)
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
