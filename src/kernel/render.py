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
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
grandfatherdir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parentdir)
sys.path.append(grandfatherdir)

import utils.solr_util as solr_util

class Render:
    def __init__(self, belief_tracker):
        self.index_cls_name_mapper = dict()
        self.belief_tracker = belief_tracker

    def render(self, response):
        rendered = False
        if response.startswith('api_call_slot_virtual_category'):
            return '您要买什么?'
        if response.startswith('api_call_request_'):
            if response.startswith('api_call_request_ambiguity_removal_'):
                params = response.replace(
                    'api_call_request_ambiguity_removal_', '')
                rendered = '你要哪一个呢,' + params
                return rendered + "@@" + response
            params = response.replace('api_call_request_', '')
            params = self.belief_tracker.belief_graph.slots_trans[params]
            rendered = '什么' + params
            return rendered + "@@" + response
        if response.startswith('api_call_rhetorical_'):
            entity = response.replace('api_call_rhetorical_', '')
            if entity in self.belief_tracker.avails:
                return '我们有' + ",".join(self.belief_tracker.avails[entity])
            else:
                return '无法查阅'
        if response.startswith('api_call_search_'):
            tokens = response.replace('api_call_search_', '').split(',')
            and_mapper = dict()
            or_mapper = dict()
            for t in tokens:
                key, value = t.split(':')
                if key == 'price':
                    or_mapper[key] = value
                else:
                    and_mapper[key] = value
            docs = solr_util.query(and_mapper, or_mapper)
            if len(docs) > 0:
                doc = docs[0]
                if 'discount' in doc and doc['discount']:
                    return '为您推荐' + doc['title'][0] + ',目前' + doc['discount'][0]
                else:
                    return '为您推荐' + doc['title'][0]

        if response.startswith('api_call_query_price_'):
            params = response.replace('api_call_query_price_' ,'')
            if not params:
                return '无法查阅'
            else:
                mapper = dict()
                for kv in params.split(','):
                    key, value = kv.split(':')
                    mapper[key] = value
            return response

        if response.startswith('api_call_query_location_', ''):
            return response
        return response
