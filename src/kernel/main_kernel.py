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

Belief Graph
"""

import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
grandfatherdir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parentdir)
sys.path.append(grandfatherdir)
import traceback
# for pickle
from graph.belief_graph import Graph
from kernel.belief_tracker import BeliefTracker
from memory.memn2n_session import MemInfer
from utils.cn2arab import *

import utils.query_util as query_util
from ml.belief_clf import Multilabel_Clf


class MainKernel:

    static_memory = None

    def __init__(self, config):
        self.config = config
        self.belief_tracker = BeliefTracker(config)
        if config['clf'] == 'memory':
            self._load_memory(config)
            self.sess = self.memory.get_session()
        else:
            self.sess = Multilabel_Clf.load(model_path=config['gbdt_model_path'])

    def _load_memory(self, config):
        if not MainKernel.static_memory:
            self.memory = MemInfer(config)
            MainKernel.static_memory = self.memory
        else:
            self.memory = MainKernel.static_memory

    def kernel(self, q, user='solr'):
        rande_rendered, wild_card = self.range_render(q)
        if self.config['clf'] == 'gbdt':
            requested = self.belief_tracker.get_requested_field()
            api = self.gbdt_reply(rande_rendered, requested)
            print(api)
            if 'api_call_slot' == api['plugin']:
                del api['plugin']
                response, avails = self.belief_tracker.memory_kernel(q, api)
            else:
                response = api['plugin']
                avails = []
            if len(avails) == 0:
                return self.render_response(response)
            return self.render_response(response) + '#avail_vals:' + str(avails)
        else:
            api = self.sess.reply(q)
            if api.startswith('api_call_slot'):
                api_json = self.api_call_slot_json_render(api)
                response = self.belief_tracker.memory_kernel(api_json, wild_card)
            else:
                response = api

            return self.render_response(response.split('#')[0])

    def gbdt_reply(self, q, requested=None):
        if requested:
            print(requested + '$' + q)
            classes, probs = self.sess.predict(requested + '$' + q)
        else:
            classes, probs = self.sess.predict(q)

        api = dict()
        for c in classes:
            key, value = c.split(':')
            api[key] = value
        return api

    def range_render(self, query):
        query, wild_card = query_util.rule_base_num_retreive(query)
        return query, wild_card

    def render_response(self, response):
        if response.startswith('api_call_request_'):
            if response.startswith('api_call_request_ambiguity_removal_'):
                params = response.replace('api_call_request_ambiguity_removal_', '')
                rendered = '你要哪一个呢,' + params
                return rendered + "@@" + response
            params = response.replace('api_call_request_', '')
            params = self.belief_tracker.belief_graph.slots_trans[params]
            rendered = '什么' + params
            return rendered + "@@" + response
        return response

    def api_call_slot_json_render(self, api):
        api = api.replace('api_call_slot_', '').split(",")
        api_json = dict()
        for item in api:
            key, value = item.split(":")
            api_json[key] = value
        return api_json


if __name__ == '__main__':
    # metadata_dir = os.path.join(
    #     grandfatherdir, 'data/memn2n/processed/metadata.pkl')
    # data_dir = os.path.join(
    #     grandfatherdir, 'data/memn2n/processed/data.pkl')
    # ckpt_dir = os.path.join(grandfatherdir, 'model/memn2n/ckpt')
    config = {"belief_graph": "../../model/graph/belief_graph.pkl",
              "solr.facet": 'on',
              "metadata_dir": os.path.join(grandfatherdir, 'data/memn2n/processed/metadata.pkl'),
              "data_dir": os.path.join(grandfatherdir, 'data/memn2n/processed/data.pkl'),
              "ckpt_dir": os.path.join(grandfatherdir, 'model/memn2n/ckpt'),
              "gbdt_model_path": grandfatherdir + '/model/ml/belief_clf.pkl',
              "clf": 'gbdt' # or memory
              }
    kernel = MainKernel(config)
    while True:
        ipt = input("input:")
        resp = kernel.kernel(ipt)
        print(resp)