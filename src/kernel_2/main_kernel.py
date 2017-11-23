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

Main Kernel
"""

import os
import sys
import logging
import time
from datetime import datetime

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
grandfatherdir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parentdir)
sys.path.append(grandfatherdir)
import traceback
# for pickle
from graph.belief_graph import Graph
from kernel_2.belief_tracker import BeliefTracker
from memory.memn2n_session import MemInfer
from dmn.dmn_fasttext.dmn_session import DmnInfer
from utils.cn2arab import *

import utils.query_util as query_util
from ml.belief_clf import Multilabel_Clf
import utils.solr_util as solr_util
from qa.iqa import Qa as QA
import memory.config as config
from kernel_2.render import Render

current_date = time.strftime("%Y.%m.%d")
logging.basicConfig(handlers=[logging.FileHandler(os.path.join(grandfatherdir,
                                                               'logs/log_corpus_' + current_date + '.log'), 'w',
                                                  'utf-8')],
                    format='%(asctime)s %(message)s', datefmt='%Y.%m.%dT%H:%M:%S', level=logging.INFO)


# os.environ['CUDA_VISIBLE_DEVICES'] = config.CUDA_DEVICE


class MainKernel:
    static_memory = None
    static_dmn = None
    static_render = None

    def __init__(self, config):
        self.config = config
        self.belief_tracker = BeliefTracker(config)
        # self.render = Render(self.belief_tracker, config)
        self._load_render(config)
        self.base_counter = 0
        self.base_clear_memory = 2
        if config['clf'] == 'memory':
            self._load_memory(config)
            self.sess = self.memory.get_session()
        elif config['clf'] == 'dmn':
            self._load_dmn(config)
            self.sess = self.dmn.get_session()
        else:
            self.sess = Multilabel_Clf.load(
                model_path=config['gbdt_model_path'])

    def _load_memory(self, config):
        if not MainKernel.static_memory:
            self.memory = MemInfer(config)
            MainKernel.static_memory = self.memory
        else:
            self.memory = MainKernel.static_memory

    def _load_dmn(self, config):
        if not MainKernel.static_dmn:
            self.dmn = DmnInfer()
            MainKernel.static_dmn = self.dmn
        else:
            self.dmn = MainKernel.static_dmn

    def _load_render(self, config):
        if not MainKernel.static_render:
            self.render = Render(config)
            MainKernel.static_render = self.render
        else:
            self.render = MainKernel.static_render

    def kernel(self, q, user='solr', recursive=False):
        if not q:
            return 'api_call_error'
        range_rendered, wild_card = self.range_render(q)
        print(range_rendered, wild_card)
        prob = -1
        response = ''
        memory = ''
        avails = []
        if self.config['clf'] == 'gbdt':
            pass
        else:
            if q.lower() == 'clear':
                self.belief_tracker.clear_memory()
                self.sess.clear_memory()
                return 'memory cleared@@[]'
            exploited = False
            prefix = ''
            if self.belief_tracker.shall_exploit_range():
                # exploited = self.belief_tracker.exploit_wild_card(wild_card)
                # if exploited:
                #     response, avails = self.belief_tracker.issue_api()
                #     memory = ''
                #     api = 'api_call_slot_range_exploit'
                    # if response.startswith('api_call_search'):
                    #     print('clear memory')
                    #     self.sess.clear_memory()
                    #     self.belief_tracker.clear_memory()
                    #     memory = ''
                pass
            if not exploited:
                api, prob = self.sess.reply(range_rendered)
                print(api, prob[0][0], prob)
                response = api
                if api.startswith('reserved_'):
                    print('miss placing cls...')
                    self.belief_tracker.clear_memory()
                    self.sess.clear_memory()
                    return self.kernel(q, user)
                if api.startswith('api_call_base') or api.startswith('api_call_query_location'):
                    memory = ''
                    response = api
                    self.base_counter += 1
                    if self.base_counter >= self.base_clear_memory:
                        self.base_counter = 0
                        print('clear memory due to base...')
                        self.belief_tracker.clear_memory()
                        self.sess.clear_memory()
                        return self.kernel(q, user)
                else:
                    self.base_counter = 0
                if api.startswith('api_call_slot'):
                    if api.startswith('api_call_slot_virtual_category'):
                        response = api
                        avails = []
                    elif api == 'api_call_slot_whatever':
                        response, avails = self.belief_tracker.defaulting_call(
                            q, wild_card)
                        prefix = self.render.random_prefix()
                    else:
                        api_json = self.api_call_slot_json_render(api)
                        response, avails, should_clear_memory = self.belief_tracker.memory_kernel(
                            q, api_json, wild_card)
                        if should_clear_memory:
                            print('restart xinhua bookstore session..')
                            self.sess.clear_memory(history=2)
                    memory = response
                    print('tree rendered..', response)
                    if response.startswith('api_call_search'):
                        # print('clear memory')
                        # self.sess.clear_memory()
                        # self.belief_tracker.clear_memory()
                        memory = ''
                if api == 'api_call_deny_all':
                    response, avails = self.belief_tracker.deny_call(slot=None)
                    memory = response
                    prefix = self.render.random_prefix()
                    print('tree rendered after deny..', response)
                if api == 'api_call_deny_brand':
                    response, avails = self.belief_tracker.deny_call(
                        slot='brand')
                    memory = response
                    prefix = self.render.random_prefix()
                    print('tree rendered after deny brand..', response)
                    # print(response, type(response))
                    # elif api.startswith('api_call_base') or api.startswith('api_call_greet'):
                    #     # self.sess.clear_memory()
                    #     matched, answer, score = self.interactive.get_responses(
                    #         query=q)
                    #     response = answer
                    #     memory = api
                    #     avails = []
            self.sess.append_memory(memory)
            render = self.render.render(q, response, self.belief_tracker.avails, prefix)
            # render = api
            logging.info("C@user:{}##model:{}##query:{}##class:{}##prob:{}##render:{}".format(
                user, 'memory', q, api, prob, render))
            result = render
            result['sentence'] = q
            result['score'] = float(prob[0][0])
            result['class'] = api
            result['graph_rendered'] = response
            return result

    def gbdt_reply(self, q, requested=None):
        if requested:
            print(requested + '$' + q)
            classes, probs = self.sess.predict(requested + '$' + q)
        else:
            classes, probs = self.sess.predict(q)

        print(probs)
        api = dict()
        for c in classes:
            print(c)
            key, value = c.split(':')
            api[key] = value
        return api

    def range_render(self, query):
        query, wild_card = query_util.rule_base_num_retreive(query)
        return query, wild_card

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
    # config = {"belief_graph": "../../model/graph/belief_graph.pkl",
    #           "solr.facet": 'on',
    #           "metadata_dir": os.path.join(grandfatherdir, 'data/memn2n/processed/archive/metadata.pkl'),
    #           "data_dir": os.path.join(grandfatherdir, 'data/memn2n/processed/archive/data.pkl'),
    #           "ckpt_dir": os.path.join(grandfatherdir, 'model/memn2n/ckpt'),
    #           "gbdt_model_path": grandfatherdir + '/model/ml/belief_clf.pkl',
    #           "clf": 'memory'  # or memory
    #           }
    config = {"belief_graph": "../../model/graph/belief_graph.pkl",
              "solr.facet": 'off',
              "metadata_dir": os.path.join(grandfatherdir, 'model/memn2n/processed/metadata.pkl'),
              "data_dir": os.path.join(grandfatherdir, 'model/memn2n/processed/data.pkl'),
              "ckpt_dir": os.path.join(grandfatherdir, 'model/memn2n/ckpt'),
              "gbdt_model_path": grandfatherdir + '/model/ml/belief_clf.pkl',
              "render_api_file": os.path.join(grandfatherdir, 'model/render_2/render_api.txt'),
              "render_location_file": os.path.join(grandfatherdir, 'model/render_2/render_location.txt'),
              "render_recommend_file": os.path.join(grandfatherdir, 'model/render_2/render_recommend.txt'),
              "render_ambiguity_file": os.path.join(grandfatherdir, 'model/render_2/render_ambiguity_removal.txt'),
              "render_price_file": os.path.join(grandfatherdir, 'model/render_2/render_price.txt'),
              "render_media_file":os.path.join(grandfatherdir, 'model/render_2/render_media.txt'),
              "faq_ad": os.path.join(grandfatherdir, 'model/ad_2/faq_ad_anchor.txt'),
              "location_ad": os.path.join(grandfatherdir, 'model/ad_2/category_ad_anchor.txt'),
              "clf": 'dmn',  # or memory`
              "shuffle": False
              }
    kernel = MainKernel(config)
    while True:
        ipt = input("input:")
        resp = kernel.kernel(ipt)
        print(resp)
