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

Render. render api_call_request_price to 什么价格...
"""
import os
import sys
import logging
import traceback
import time

import numpy as np

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
grandfatherdir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parentdir)
sys.path.append(grandfatherdir)

import utils.solr_util as solr_util
from qa.iqa import Qa as QA
current_date = time.strftime("%Y.%m.%d")


class AdKernel:

    AD_THRES = 1
    def __init__(self, config):
        self._load_faq_ad_anchor(config['faq_ad'])

    def _load_faq_ad_anchor(self, file):
        self.faq_ads = {}
        with open(file, 'r') as f:
            for line in f:
                line = line.strip('\n')
                _, _id, answer, ad = line.split('\t')
                ads = ad.split('/')
                self.faq_ads[answer] = ads

    def anchor_faq_ad(self, _id):
        if np.random.random() < self.AD_THRES:
            if _id in self.faq_ads:
                return np.random.choice(self.faq_ads[_id])
        return ''

    def render_ad(self):
        return ''

if __name__ == '__main__':
    ad_kernel = AdKernel({'faq_ad': os.path.join(grandfatherdir, 'model/ad/faq_ad.txt'),})
    print(ad_kernel.anchor_faq_ad('d8335d24-11e4-4f1c-b6b5-4ed94d6d44ae'))