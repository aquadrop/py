""" A neural chatbot using sequence to sequence model with
attentional decoder.

This is based on Google Translate Tensorflow model
https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/

Sequence to sequence model by Cho et al.(2014)

Created by Chip Huyen as the starter code for assignment 3,
class CS 20SI: "TensorFlow for Deep Learning Research"
cs20si.stanford.edu

This file contains the code to run the model.

See readme.md for instruction on how to run the starter code.

This implementation learns NUMBER SORTING via seq2seq. Number range: 0,1,2,3,4,5,EOS

https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf

See README.md to learn what this code has done!

Also SEE https://stackoverflow.com/questions/38241410/tensorflow-remember-lstm-state-for-next-batch-stateful-lstm
for special treatment for this code

Belief Tracker
"""

import re
import requests

import jieba

import os
dir_path = os.path.dirname(os.path.realpath(__file__))

jieba.load_userdict(dir_path + "/../../data/dict/ext1.dic")


def jieba_cut(query, smart=True):
    seg = jieba.cut(query, cut_all=not smart)
    result = []
    for s in seg:
        result.append(s)
    return result

if __name__ == "__main__":
    print(' '.join(jieba_cut('华为num元手机phone.mmem')))