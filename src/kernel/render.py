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

import numpy as np

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
grandfatherdir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parentdir)
sys.path.append(grandfatherdir)

import utils.solr_util as solr_util
from qa.qa import Qa as QA


class Render:

    prefix = ['这样啊..', 'OK..', '好吧']

    def __init__(self, belief_tracker, config):
        self.index_cls_name_mapper = dict()
        self._load_major_render(config['renderer_file'])
        # self.belief_tracker = belief_tracker
        self.interactive = QA('interactive')
        self.faq = QA('faq')
        print('attaching rendering file...')

    def _load_major_render(self, file):
        self.major_render_mapper = dict()
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip('\n')
                key, replies = line.split('|')
                key = key.split('##')[0]
                replies = replies.split('/')
                self.major_render_mapper[key] = replies

    def render_mapper(self, mapper):
        mapper_render = []
        if 'brand' in mapper:
            mapper_render.append(mapper['brand'])
        if 'category' in mapper:
            mapper_render.append(mapper['category'])
        return ''.join(mapper_render)

    def random_prefix(self):
        return np.random.choice(self.prefix)

    def render_api(self, api):
        if api not in self.major_render_mapper:
            return api
        return np.random.choice(self.major_render_mapper[api])

    def render(self, q, response, avails=dict(), prefix=''):
        if response.startswith('api_call_base') or response.startswith('api_call_greet')\
                or response.startswith('reserved_'):
            # self.sess.clear_memory()
            matched, answer, score = self.interactive.get_responses(
                query=q)
            return answer
        if response.startswith('api_call_faq') or response.startswith('api_call_query_discount'):
            matched, answer, score = self.faq.get_responses(
                query=q)
            return answer
        if response.startswith('api_call_slot_virtual_category') or response == 'api_greeting_search_normal':
            return '您要买什么?'
        if response.startswith('api_call_request_'):
            if response.startswith('api_call_request_ambiguity_removal_'):
                # params = response.replace(
                #     'api_call_request_ambiguity_removal_', '')
                # rendered = '你要哪一个呢,' + params
                # return rendered + "@@" + response
                return self.render_api(response)
            # params = response.replace('api_call_request_', '')
            # params = self.belief_tracker.belief_graph.slots_trans[params]
            # rendered = '什么' + params
            # return rendered + "@@" + response
            if prefix:
                return prefix + self.render_api(response)
            return self.render_api(response)
        if response.startswith('api_call_rhetorical_'):
            entity = response.replace('api_call_rhetorical_', '')
            if entity in avails and len(avails[entity]) > 0:
                return '我们有' + ",".join(avails[entity])
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
            else:
                # use loose search, brand and category is mandatory
                and_mapper.clear()
                or_mapper.clear()
                for t in tokens:
                    key, value = t.split(':')
                    if key in ['category', 'brand']:
                        and_mapper[key] = value
                    else:
                        or_mapper[key] = value
                docs = solr_util.query(and_mapper, or_mapper)
                if len(docs) > 0:
                    doc = docs[0]
                    if 'discount' in doc and doc['discount']:
                        return '没有找到完全符合您要求的商品,为您推荐' + doc['title'][0] + ',目前' + doc['discount'][0]
                    else:
                        return '没有找到完全符合您要求的商品,为您推荐' + doc['title'][0]
                return response

        if response.startswith('api_call_query_price_'):
            params = response.replace('api_call_query_price_' ,'')
            if not params:
                return '无法查阅'
            else:
                mapper = dict()
                for kv in params.split(','):
                    key, value = kv.split(':')
                    mapper[key] = value

            facet = solr_util.solr_facet(mappers=mapper, facet_field='price', is_range=True)
            response = self.render_mapper(mapper) + '目前价位在' + ','.join(facet[0])
            return response

        if response.startswith('api_call_query_location_'):
            params = response.replace('api_call_query_location_', '')
            if not params:
                return '无法查阅'
            else:
                mapper = dict()
                for kv in params.split(','):
                    key, value = kv.split(':')
                    mapper[key] = value
            facet = solr_util.solr_facet(mappers=mapper, facet_field='location', is_range=False)
            response = '您要找的' + self.render_mapper(mapper) + '在' + ','.join(facet[0])
            return response

        return response
