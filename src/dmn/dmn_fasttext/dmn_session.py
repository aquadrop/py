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
from config import Config
import data_helper

translator = Translator()


class DmnSession():
    def __init__(self, config, graph, metadata, char=2):
        self.context = [[' ']]
        self.u = None
        self.r = None
        self.config = config
        self.graph = graph
        self.idx2candid = metadata['idx2candid']
        self.w2idx = metadata['w2idx']
        self.metadata = metadata
        self.char = char

    def append_memory(self, m):
        if not m:
            return
        m = translator.en2cn(m)
        m = tokenize(m, self.char)
        self.context.append(m)

    def clear_memory(self, history=0):
        if history == 0:
            self.context = [[' ']]
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
            if not self.config.word:
                q_vector = [self.w2idx.get(w, 0) for w in q]
                inp_vector = [[self.w2idx.get(w, 0) for w in s]
                              for s in self.context]
            else:
                q_vector = q
                inp_vector = self.context
            inputs.append(inp_vector)
            questions.append(q_vector)
            input_lens, sen_lens, max_sen_len = data_helper.get_sentence_lens(
                inputs)
            q_lens = data_helper.get_lens(questions)

            data = questions, inputs, q_lens, sen_lens, input_lens, [], [], []
            qp_vc, ip_vc, ql_vc, il_vc, im_vc, a_vc, r_vc = data_helper.vectorize_data(
                data, self.config, self.metadata)

            with tf.Session() as session:
                session.run(tf.global_variables_initializer())

                top_predict_proba = self.graph.get_tensor_by_name(
                    'pred:0')
                qp = self.graph.get_tensor_by_name('questions:0')
                ql = self.graph.get_tensor_by_name('question_lens:0')
                ip = self.graph.get_tensor_by_name('inputs:0')
                il = self.graph.get_tensor_by_name('input_lens:0')
                dp = self.graph.get_tensor_by_name('dropout:0')
                output = session.run(top_predict_proba, feed_dict={
                    qp: qp_vc, ql: ql_vc, ip: ip_vc, il: il_vc, dp: self.config.dropout})

            print('output:', output)
            # indices = output.indices.tolist()[0]
            # values = output.values.tolist()[0]

            reply_msg = [self.idx2candid[ind] for ind in output]
            # print(reply_msg)
            r = reply_msg[0]
            # print('r:',r)
            r = translator.en2cn(r)
            r = tokenize(r, self.char)
            self.context.append(r)

        if self.config.multi_label:
            return reply_msg
        else:
            return reply_msg[0]


class DmnInfer:
    def __init__(self):
        self.config = Config()

    def get_session(self):
        with open(self.config.metadata_path, 'rb') as f:
            metadata = pickle.load(f)

        saver = tf.train.import_meta_graph(
            self.config.ckpt_path + 'dmn.weights.meta')
        graph = tf.get_default_graph()

        char = 2 if self.config.word else 1
        isess = DmnSession(self.config, graph, metadata, char)
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
