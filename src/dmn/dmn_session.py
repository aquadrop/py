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

import dmn_data_utils2 as dmn_data_utils
from dmn_plus2 import Config, DMN_PLUS

translator = Translator()


class DmnSession():
    def __init__(self, session, model):
        self.context = [['此', '乃', '空', '文']]
        self.u = None
        self.r = None
        self.model = model
        self.session = session
        self.idx2candid = self.model.idx2candid
        self.w2idx = self.model.w2idx

    def reply(self, msg):
        line = msg.strip().lower()
        if line == 'clear':
            self.context = []
            replp_msg = 'memory cleared!'
        else:
            inputs = []
            questions = []

            q = tokenize(line)
            q_vector = [self.w2idx[w] for w in q]
            inp_vector = [[self.w2idx[w] for w in s] for s in self.context]

            inputs.append(inp_vector)
            questions.append(np.vstack(q_vector).astype(np.float32))

            input_lens, sen_lens, max_sen_len = dmn_data_utils.get_sentence_lens(
                inputs)

            q_lens = dmn_data_utils.get_lens(questions)
            # max_q_len = np.max(q_lens)

            # max_input_len = min(np.max(input_lens),
            #                     self.config.max_allowed_inputs)

            max_input_len = self.model.max_input_len
            max_sen_len = self.model.max_sen_len
            max_q_len = self.model.max_q_len

            inputs = dmn_data_utils.pad_inputs(inputs, input_lens, max_input_len,
                                               "split_sentences", sen_lens, max_sen_len)

            # inputs = [inputs[0] for _ in range(self.config.batch_size)]
            inputs = np.asarray(inputs)

            questions = dmn_data_utils.pad_inputs(
                questions, q_lens, max_q_len)
            # questions = [questions[0] for _ in range(self.config.batch_size)]
            questions = np.asarray(questions)

            preds = self.model.predict(self.session,
                                       inputs, input_lens, max_sen_len, questions, q_lens)
            preds = preds[0].tolist()
            print(preds)
            r = self.idx2candid[preds[0]]
            reply_msg = r
            r = translator.en2cn(r)
            r = tokenize(r)
            self.context.append(r)

        return reply_msg


class DmnInfer:
    def __init__(self):
        self.config = Config()
        self.model = self._load_model()
        self.session = tf.Session()


    def _load_model(self):
        self.config.train_mode=False
        model = DMN_PLUS(self.config)
        return model

    def get_session(self):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        self.session.run(init)

        # restore checkpoint
        ckpt = tf.train.get_checkpoint_state(self.config.ckpt_path+'weights5/')
        if ckpt and ckpt.model_checkpoint_path:
            print('\n>> restoring checkpoint from',
                  ckpt.model_checkpoint_path)
        saver.restore(self.session, ckpt.model_checkpoint_path)

        isess = DmnSession(self.session, self.model)
        return isess


def main():
    di = DmnInfer()
    sess = di.get_session()

    query = ''
    while query != 'exit':
        query = input('>> ')
        print('>> ' + sess.reply(query))


if __name__ == '__main__':
    main()
