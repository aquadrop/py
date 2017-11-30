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

class Render:

    prefix = ['这样啊.', '没问题.', '好吧']
    ANY = 'any'
    def __init__(self, config):
        self.index_cls_name_mapper = dict()
        self._load_major_render(config['render_api_file'])
        self._load_location_render(config['render_location_file'])
        self._load_ambiguity_render(config['render_ambiguity_file'])
        self._load_recommend_render(config['render_recommend_file'])
        self._load_price_render(config['render_price_file'])
        self._load_media_render(config['render_media_file'])
        self._load_emotion_render(config['emotion_file'])
        self.ad_kernel = AdKernel(config)
        # self.belief_tracker = belief_tracker
        self.interactive = QA('base')
        self.faq = QA('base')
        print('attaching rendering file...')

    def _load_emotion_render(self, file):
        self.emotion = []
        self.emotion_prob = 0.8
        with open (file, 'r') as f:
            for line in f:
                line = line.strip('\n')
                self.emotion.append(line)

    def render_emotion(self):
        if np.random.uniform() < self.emotion_prob:
            return np.random.choice(self.emotion)
        return 'null'

    def _load_media_render(self, file):
        self.media_render_mapper=dict()
        with open(file,'r') as f:
            for line in f:
                line=line.strip('\n')
                values=line.split('#')
                if not values[1]:
                    self.media_render_mapper[values[0]]=hashlib.sha256(values[0].encode('utf-8')).hexdigest()
                else:
                    self.media_render_mapper[values[0]]=values[1]

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

    def _load_price_render(self, file):
        self.price_templates = []
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip('\n')
                self.price_templates.append(line)

    def render_price(self, mapper, price):
        rendered = self.render_mapper(mapper) + '目前价位在' + price
        if 'brand' in mapper and 'category' in mapper:
            template = np.random.choice(self.price_templates)
            rendered = template.replace('[brand]', mapper['brand']).\
                replace('[category]', mapper['category']).replace('[price]', price)
        return rendered

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
        rendered = template.replace('[category]', category).replace('[location]', location)
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

    def render_api(self, api, replacements={}):
        if api not in self.major_render_mapper:
            return api
        # if api == 'api_call_request_brand':
        #     return self.render_brand(self.major_render_mapper[api], replacements)
        return np.random.choice(self.major_render_mapper[api])

    def render_media(self, api):
        if api not in self.media_render_mapper:
            return 'null'
        return self.media_render_mapper[api]

    def render_brand(self, templates, replacements={}):
        brands = []
        if 'brand' in replacements:
            brands = replacements['brand']
        template = ','.join(brands) + '品牌您要哪一个' # default
        for i in range(10):
            _template = np.random.choice(templates)
            if len(brands) < 2\
                and ('[' in _template or ']' in _template):
                continue
            else:
                template = _template
                break

        rendered = template
        if len(brands) >= 2:
            pre = brands[0:-1]
            post = brands[-1]
            rendered = template.replace('[pre]', ','.join(pre)).replace('[post]', post)
        return rendered

    def render(self, q, response, avails=dict(), prefix=''):
        result = {"answer":"", "media":"null", 'from':"memory", "sim":0, "timeout":-1}
        try:
            # media=self.render_media(response)
            if response.startswith('api_call_base') or response.startswith('api_call_greet')\
                    or response.startswith('reserved_'):
                # self.sess.clear_memory()
                matched, answer, score = self.interactive.get_responses(
                    query=q)
                result['answer'] = answer
                result['from'] = 'base'
                result['sim'] = score
                result['emotion'] = self.render_emotion()
                return result
            if response.startswith('api_call_faq_general'):
                matched, answer, score = self.faq.get_responses(
                    query=q)
                ad = self.ad_kernel.anchor_faq_ad(answer)
                answer = answer + ' ' + ad
                result['answer'] = answer
                return result
            if response.startswith('api_call_faq_info'):
                result['media'] = self.render_media(response)
                answer = self.render_api(response, avails)
                result['answer'] = answer
                # result['avail_vals'] = avails
                return result
            if response.startswith('api_call_query_discount'):
                answer = self.render_api(response)
                result['answer'] = answer
                return result
            if response.startswith('api_call_query_general'):
                answer = self.render_api(response)
                result['answer'] = answer
                return result
            if response.startswith('api_call_slot_virtual_category') or response == 'api_greeting_search_normal':
                answer = self.render_api(response, {})
                result['answer'] = answer
                return result
            if response.startswith('api_call_request_') or response.startswith('api_call_search_'):
                if prefix:
                    answer = prefix + self.render_api(response, avails)
                    result['answer'] = answer
                    #result['avail_vals'] = avails
                    return result
                result['media'] = self.render_media(response)
                answer = self.render_api(response, avails)
                result['answer'] = answer
                #result['avail_vals'] = avails
                return result
            # if response.startswith('api_call_search_'):
            #     return response
            #     tokens = response.replace('api_call_search_', '').split(',')
            #
            #     and_mapper = dict()
            #     or_mapper = dict()
            #     for t in tokens:
            #         key, value = t.split(':')
            #         if key == 'price':
            #             or_mapper[key] = value
            #         else:
            #             and_mapper[key] = value
            #     docs = solr_util.query(and_mapper, or_mapper)
            #     if len(docs) > 0:
            #         doc = docs[0]
            #         return self.render_recommend(doc['title'][0])
            #     else:
            #         # use loose search, brand and category is mandatory
            #         and_mapper.clear()
            #         or_mapper.clear()
            #         for t in tokens:
            #             key, value = t.split(':')
            #             if key in ['category', 'brand']:
            #                 and_mapper[key] = value
            #             else:
            #                 or_mapper[key] = value
            #         docs = solr_util.query(and_mapper, or_mapper)
            #         if len(docs) > 0:
            #             doc = docs[0]
            #             return '没有找到完全符合您要求的商品.' + self.render_recommend(doc['title'][0])
            #         return response

            if response.startswith('api_call_query_location_'):
                params = response.replace('api_call_query_location_', '')
                if not params:
                    return '我们这里按照商品种类分布,您可以咨询我商品的方位信息'
                else:
                    mapper = dict()
                    for kv in params.split(','):
                        key, value = kv.split(':')
                        mapper[key] = value
                facet = solr_util.solr_facet(mappers=mapper,
                                             facet_field='location',
                                             is_range=False, prefix='',
                                             postfix='_str',
                                             core='bookstore_map')
                location = ','.join(facet[0])
                category = ','.join(mapper.values())
                image_key = ''
                try:
                    if 'category' in mapper or 'virtual_category' in mapper:
                        title = category
                    else:
                        title = facet[2][0]['title']
                    image_key = facet[2][0]['image_key']
                except:
                    title = category
                response = self.render_location(category, location)
                response = response.replace(category, title)
                if 'category' in mapper:
                    ad = self.ad_kernel.anchor_category_ad(mapper['category'])
                    response = response + ' ' + ad
                result = {'answer': response, 'media': image_key, 'avail_vals':""}
                return result
            return response
        except:
            print(traceback.format_exc())
            matched, answer, score = self.interactive.get_responses(
                query=q)
            logging.error("C@code:{}##error_details:{}".format('render', traceback.format_exc()))
            result['answer'] = answer
            return result

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
              "clf": 'memory'  # or memory
              }
    render = Render(config)
    print(render.render_recommend('空调'))
    print(render.render('你好', 'api_call_request_ambiguity_removal_手机,苹果'))
    print(render.render_price({'brand':'西门子', 'category':'空调'}, '2000-3000'))
    print(render.render_media('api_call_request_category'))
    print(render.render_media('吴江新华书店咖啡馆'))