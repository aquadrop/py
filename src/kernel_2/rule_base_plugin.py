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
import hashlib

import numpy as np

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
grandfatherdir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parentdir)
sys.path.append(grandfatherdir)

import utils.solr_util as solr_util
from qa.iqa import Qa as QA
from kernel_2.ad_kernel import AdKernel
current_date = time.strftime("%Y.%m.%d")
logging.basicConfig(handlers=[logging.FileHandler(os.path.join(grandfatherdir,
                    'logs/log_corpus_' + current_date + '.log'), 'w', 'utf-8')],
                    format='%(asctime)s %(message)s', datefmt='%Y.%m.%dT%H:%M:%S', level=logging.INFO)

class RuleBasePlugin:
    def __init__(self, config):
        self.api_list = ['api_call_faq_info']
        self.should_clear_list = ['api_call_request_reg.complete']
        self._load_key_word_file(config['key_word_file'])
        self._load_noise_filter(config['noise_keyword_file'])

    def _load_key_word_file(self, key_word_file):
        self.key_words = []
        with open(key_word_file, 'r') as f:
            for line in f:
                line = line.strip('\n')
                self.key_words.append(line)

    def _load_noise_filter(self, noise_file):
        self.noise_keywords = set()
        with open(noise_file, 'r') as f:
            for line in f:
                line = line.strip('\n')
                self.noise_keywords.add(line)

    def request_clear_memory(self, api, sess, belief_tracker):
        if api.startswith('api_call_search') or api in self.should_clear_list:
            sess.clear_memory(0)
            belief_tracker.clear_memory()
            print('rule base cleared..')

    def filter(self, q):
        if q in self.noise_keywords or len(q) <= 1:
            return ''
        return q

    def fix(self, q, api):
        should_fix = False
        for listed in self.api_list:
            if api.startswith(listed):
                should_fix = True
                break
        if not should_fix:
            return api

        components = api.split('_')[-1]
        values = [component.split(":")[1] for component in components.split(",")]
        if len(values) == 0:
            return api

        for value in values:
            if value not in q:
                key = self.find_key_word(q)
                if key:
                    api = api.replace(value, key)
        return api

    def find_key_word(self, q):
        for key in self.key_words:
            if key in q:
                return key
        return None

if __name__ == "__main__":
    config = {"belief_graph": "../../model/graph/belief_graph.pkl",
              "solr.facet": 'on',
              "metadata_dir": os.path.join(grandfatherdir, 'data/memn2n/processed/metadata.pkl'),
              "data_dir": os.path.join(grandfatherdir, 'data/memn2n/processed/data.pkl'),
              "ckpt_dir": os.path.join(grandfatherdir, 'model/memn2n/ckpt'),
              "gbdt_model_path": grandfatherdir + '/model/ml/belief_clf.pkl',
              "render_api_file": os.path.join(grandfatherdir, 'model/render/render_api.txt'),
              "render_location_file": os.path.join(grandfatherdir, 'model/render/render_location.txt'),
              "render_recommend_file": os.path.join(grandfatherdir, 'model/render/render_recommend.txt'),
              "render_ambiguity_file": os.path.join(grandfatherdir, 'model/render/render_ambiguity_removal.txt'),
              "render_price_file": os.path.join(grandfatherdir, 'model/render/render_price.txt'),
              "render_media_file":os.path.join(grandfatherdir, 'model/render/render_media.txt'),
              "faq_ad": os.path.join(grandfatherdir, 'model/ad/faq_ad.txt'),
              "location_ad": os.path.join(grandfatherdir, 'model/ad/category_ad_anchor.txt'),
              "clf": 'memory',  # or memory,
              "key_word_file": os.path.join(grandfatherdir, 'model/render_2/key_word.txt')
              }
    rule = RuleBasePlugin(config)
    print(rule.fix('三楼有什么好玩的','api_call_faq_info:一楼'))

