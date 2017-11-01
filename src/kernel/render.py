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
logging.basicConfig(handlers=[logging.FileHandler(os.path.join(grandfatherdir,
                    'logs/log_corpus_' + current_date + '.log'), 'w', 'utf-8')],
                    format='%(asctime)s %(message)s', datefmt='%Y.%m.%dT%H:%M:%S', level=logging.INFO)

class Render:

    prefix = ['这样啊.', '没问题.', '好吧']
    ANY = 'any'
    def __init__(self, config):
        self.index_cls_name_mapper = dict()
        self._load_major_render(config['render_api_file'])
        self._load_location_render(config['render_location_file'])
        self._load_ambiguity_render(config['render_ambiguity_file'])
        self._load_recommend_render(config['render_recommend_file'])
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
                filtered = []
                for r in replies:
                    if r:
                        filtered.append(r)
                self.major_render_mapper[key] = filtered

    def _load_location_render(self, file):
        self.location_templates = []
        self.location_applicables = dict()
        self.location_precludes = dict()
        index = 0
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip('\n')
                template, applicable, preclude = line.split('#')
                self.location_templates.append(template)
                self.location_applicables[index] = applicable.split(',')
                self.location_precludes[index] = preclude.split(',')
                index += 1

    def _load_recommend_render(self, file):
        self.recommend_templates = []
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip('\n')
                self.recommend_templates.append(line)

    def _load_ambiguity_render(self, file):
        self.dual_removal = []
        self.mlt_removal = []
        index = 0
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip('\n')
                if index == 0:
                    self.dual_removal = line.split('/')
                if index > 0:
                    self.mlt_removal = line.split('/')
                index += 1

    def render_location(self, category, location):
        template = '<category> <location>'
        for i in range(20):
            index = np.random.randint(len(self.location_templates))
            applicables = self.location_applicables[index]
            precludes = self.location_precludes[index]
            if self.ANY in applicables:
                if category not in precludes:
                    template = self.location_templates[index]
                    break
            else:
                if category in applicables:
                    template = self.location_templates[index]
                    break
        rendered = template.replace('<category>', category).replace('<location>', location)
        return rendered

    def render_ambiguity(self, ambiguity_slots):
        if len(ambiguity_slots) == 2:
            a = ambiguity_slots[0]
            b = ambiguity_slots[1]
            template = np.random.choice(self.dual_removal)
            rendered = template.replace('<0>', a).replace('<1>', b)
        if len(ambiguity_slots) > 2:
            a = ','.join(ambiguity_slots[0:-1])
            b = ambiguity_slots[-1]
            template = np.random.choice(self.mlt_removal)
            rendered = template.replace('<pre>', a).replace('<post>', b)
        return rendered

    def render_recommend(self, title):
        template = np.random.choice(self.recommend_templates)
        rendered = template.replace('<title>', title)
        return rendered

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
        try:
            if response.startswith('api_call_base') or response.startswith('api_call_greet')\
                    or response.startswith('reserved_'):
                # self.sess.clear_memory()
                matched, answer, score = self.interactive.get_responses(
                    query=q)
                return answer
            if response.startswith('api_call_faq'):
                matched, answer, score = self.faq.get_responses(
                    query=q)
                return answer
            if response.startswith('api_call_query_discount'):
                return self.render_api(response)
            if response.startswith('api_call_slot_virtual_category') or response == 'api_greeting_search_normal':
                return np.random.choice(['您要买什么?我们有手机,冰箱,电视,电脑和空调.', '你可以看看我们的手机,冰箱,电视空调电脑'])
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
                    return np.random.choice(['您好,我们这里卖各种空调电视电脑冰箱等,价格不等,您可以来看看呢',
                                             '您好啊,这里有各种冰箱空调电视等,价格在3000-18000,您可以来看看呢'])
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
                    return self.render_recommend(doc['title'][0])
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
                        return '没有找到完全符合您要求的商品.' + self.render_recommend(doc['title'][0])
                    return response

            if response.startswith('api_call_query_price_'):
                params = response.replace('api_call_query_price_' ,'')
                if not params:
                    return '价位在3000-18000'
                else:
                    mapper = dict()
                    for kv in params.split(','):
                        key, value = kv.split(':')
                        mapper[key] = value

                facet = solr_util.solr_facet(mappers=mapper, facet_field='price', is_range=True)
                response = self.render_mapper(mapper) + '目前价位在' + ','.join(facet[0])
                return response

            if response.startswith('api_call_query_brand_'):
                params = response.replace('api_call_query_brand_' ,'')
                if not params:
                    raise ValueError('api_call_query must have params provided...')
                else:
                    mapper = dict()
                    for kv in params.split(','):
                        key, value = kv.split(':')
                        mapper[key] = value

                facet = solr_util.solr_facet(mappers=mapper, facet_field='brand', is_range=False)
                response = self.render_mapper(mapper) + '有' + ','.join(facet[0])
                return response

            if response.startswith('api_call_query_location_'):
                params = response.replace('api_call_query_location_', '')
                if not params or 'category' not in params:
                    return '我们这里按照商品种类分布,您可以咨询我商品的方位信息'
                else:
                    mapper = dict()
                    for kv in params.split(','):
                        key, value = kv.split(':')
                        mapper[key] = value
                facet = solr_util.solr_facet(mappers=mapper, facet_field='location', is_range=False)
                location = ','.join(facet[0])
                category = mapper['category']
                response = self.render_location(category, location)
                return response

            return response
        except:
            print(traceback.format_exc())
            matched, answer, score = self.interactive.get_responses(
                query=q)
            logging.error("C@code:{}##error_details:{}".format('render', traceback.format_exc()))
            return answer

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
              "clf": 'memory'  # or memory
              }
    render = Render(config)
    print(render.render_recommend('空调'))
    print(render.render_ambiguity(['空调','洗衣机','电视']))