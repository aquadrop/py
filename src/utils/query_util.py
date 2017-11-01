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
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
grandfatherdir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parentdir)
sys.path.append(grandfatherdir)

from utils.cn2arab import *
import memory.config as config
dir_path = os.path.dirname(os.path.realpath(__file__))

jieba.load_userdict(dir_path + "/../../data/dict/ext1.dic")
STOP_WORDS = set(["！", "？", "，", "。", "，", '*', ',', '_', ':', ' ', ',', '.',
                  '\t', '?', '(', ')', '!', '~', '“', '”', '《', '》', '+', '-', '='])

STOP_WORDS_0 = set(["！", "？", "，", "。", "，", '*', ":", '_', '.', ' ', ',',
                    '\t', '?', '(', ')', '!', '~', '“', '”', '《', '》', '+', '-', '=',"%","……",
                    "啊", "呢", "吗", '呀', '哒'])


def tokenize(sent, char=config.TOKENIZE_CHAR):
    sent = sent.lower().strip()
    tokens = list()
    if char == 0:
        for s in STOP_WORDS_0:
            sent = sent.replace(s, '')
        tokens = list(sent)
        return tokens
    elif char == 1:
        split_list = [',', ':', '_']
        zh_pattern = re.compile(u'[\u4e00-\u9fa5]+')
        en = list()
        digits = list()
        state = 'cn'
        # sent = sent.replace(' ', '').replace('\t', '')
        sent = sent.replace('range', ' range ')
        for c in sent:
            match = zh_pattern.search(c)
            if match:
                if state == 'en':
                    tokens.append(''.join(en))
                    en = list()
                if state == 'digits':
                    tokens.append(''.join(digits))
                    digits = list()
                state = 'cn'
                tokens.append(c)
                continue
            if c.isdigit():
                if state == 'en':
                    tokens.append(''.join(en))
                    en = list()
                state = 'digits'
                digits.append(c)
                continue
            if c == '.' and len(digits) > 0:
                digits.append(c)
                continue
            if c == '$':
                if state == 'en':
                    tokens.append(''.join(en))
                    en = list()
                if state == 'digits':
                    tokens.append(''.join(digits))
                    digits = list()
                state = '$'
                continue
            if (c == 'r' or c == 'u') and state == '$':
                tokens.append('$' + c)
                state = ''
                continue
            if c in STOP_WORDS:
                if state == 'en':
                    tokens.append(''.join(en))
                    en = list()
                if state == 'digits':
                    tokens.append(''.join(digits))
                    digits = list()
            if c.isalpha():
                if state == 'digits':
                    tokens.append(''.join(digits))
                    digits = list()
                state = 'en'
                en.append(c)

        if state == 'en':
            if len(en) > 0:
                tokens.append(''.join(en))
            en = list()
        if state == 'digits':
            if len(digits) > 0:
                tokens.append(''.join(digits))
            digits = list()
        for s in STOP_WORDS_0:
            if s in tokens:
                tokens.remove(s)
    else:
        tokens = [w for w in list(jieba.cut(sent.strip()))
                  if w not in STOP_WORDS_0]
    return tokens


def jieba_cut(query, smart=True):
    seg = jieba.cut(query, cut_all=not smart)
    result = []
    for s in seg:
        result.append(s)
    return result

def remove_stop_words(q):
    # q = re.sub(ur"[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）：；《）《》“”()»〔〕-]+", "", q)

    pu = re.compile(r'[啊|呢|哦|哈|呀|捏|撒|哟|呐|吧|吗|么|嘛]')
    try:
        return re.sub(pu, '', q.decode('utf-8'))
    except:
        return q

rp = {"大一匹": "1.3p", "大1匹":"1.3p"}
def supersede(query):
    for key, value in rp.items():
        query = query.replace(key, value)
    return query

def rule_base_num_retreive(query):

    query = supersede(query)

    inch_dual = r"(([-+]?\d*\.\d+|\d+)[到|至]([-+]?\d*\.\d+|\d+)寸)"
    meter_dual = r"([-+]?\d*\.\d+|\d+)[到|至]([-+]?\d*\.\d+|\d+)米"
    ac_power_dual = r"([-+]?\d*\.\d+|\d+)[到|至]([-+]?\d*\.\d+|\d+)[p|P|匹]"
    price_dual = r"([-+]?\d*\.\d+|\d+)[到|至]([-+]?\d*\.\d+|\d+)[块|元]"
    people_dual = r"(([-+]?\d*\.\d+|\d+)[到|至]([-+]?\d*\.\d+|\d+)人)"

    inch_single = r"([-+]?\d*\.\d+|\d+)寸"
    meter_single = r"([-+]?\d*\.\d+|\d+)米"
    ac_power_single = r"([-+]?\d*\.\d+|\d+)[p|P|匹]"
    price_single = r"([-+]?\d*\.\d+|\d+)[块|元]"
    people_single = r"([-+]?\d*\.\d+|\d+)人"
    height = r"高([-+]?\d*\.\d+|\d+)米"
    width = r"宽([-+]?\d*\.\d+|\d+)米"
    memory = r"([-+]?\d*\.\d+|\d+)[g|G]"

    dual = {"__inch__": inch_dual, "__meter__": meter_dual,
            "ac.power": ac_power_dual,
            "price": price_dual}
    single = {"__inch__": inch_single, "__meter__": meter_single,
              "ac.power": ac_power_single,
              "price": price_single, "people": people_single, "height": height, "width":width,
              "memory": memory}

    wild_card = dict()
    query = str(new_cn2arab(query))
    flag = False
    for key, value in dual.items():
        render, numbers = range_extract(value, query, False, True)
        if numbers:
            flag = True
            wild_card[key] = numbers

    for key, value in single.items():
        render, numbers = range_extract(value, query, True, True)
        if numbers:
            flag = True
            wild_card[key] = numbers

    if flag:
        return render, wild_card
    price_dual_default = r"([-+]?\d*\.\d+|\d+)[到|至]([-+]?\d*\.\d+|\d+)(?!P|匹|米|寸|p|T|t|级|k|K|人|g|G)"
    price_single_default = r"([-+]?\d*\.\d+|\d+)(?!P|匹|米|寸|p|T|t|级|k|K|人|g|G)"
    remove_regex = r"\d+[个|只|条|部|本|台]"
    query = re.sub(remove_regex, '', query)
    render, numbers = range_extract(price_dual_default, query, False, True)
    if numbers:
        wild_card['number'] = numbers
        return render, wild_card
    render, numbers = range_extract(price_single_default, query, True, True)
    if numbers:
        wild_card['number'] = numbers
    return render, wild_card


def range_extract(pattern, query, single, range_render=False):
    """

    :param pattern:
    :param query:
    :param single:
    :return:
    """
    numbers = []
    match = re.search(pattern, query)
    if range_render:
        range_rendered_query = re.sub(pattern, 'range', query)
    else:
        range_rendered_query = query
    if single:
        if match:
            numbers = match.group(0)
            numbers = float(re.findall(r"[-+]?\d*\.\d+|\d+", numbers)[0])
            numbers = [numbers * 0.9, numbers * 1.1]
            numbers = '[' + str(numbers[0]) + " TO " + \
                str(numbers[1]) + "]"
    else:
        if match:
            numbers = match.group(0)
            numbers = [float(r) for r in re.findall(
                r"[-+]?\d*\.\d+|\d+", numbers)[0:2]]
            numbers = '[' + str(numbers[0]) + " TO " + \
                str(numbers[1]) + "]"
    return range_rendered_query, numbers


if __name__ == "__main__":
    # print(' '.join(jieba_cut('华为num元手机phone.mmem')))
    # print(rule_base_num_retreive('50寸电视'))
    print(rule_base_num_retreive('哪点事三人3000,高4米iphone6s, 大一匹'))
    # print(tokenize('plugin:api_call_slot,phone.mmem:1.5g do you speak', char=1))
    print(rule_base_num_retreive(''))
