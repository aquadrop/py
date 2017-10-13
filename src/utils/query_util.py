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
dir_path = os.path.dirname(os.path.realpath(__file__))

jieba.load_userdict(dir_path + "/../../data/dict/ext1.dic")
STOP_WORDS = set(["！", "？", "，", "。", "，", '*', ":",
                  '\t', '?', '(', ')', '!', '~', '“', '”', '《', '》', '+', '-', '='])


def tokenize(sent, char=1):
    sent = sent.lower()
    tokens = list()
    if char == 0:
        tokens = list(sent)
        for s in STOP_WORDS:
            if s in tokens:
                tokens.remove(s)
        return tokens
    elif char == 1:
        split_list = [',', ':']

        zh_pattern = re.compile(u'[\u4e00-\u9fa5]+')
        en = list()
        for c in sent:
            if c in STOP_WORDS:
                continue
            match = zh_pattern.search(c)
            if match:
                if en:
                    ll = ''.join(en).split()
                    for l in ll:
                        tokens.append(l)
                    en = list()
                tokens.append(c)
            else:
                if c in split_list:
                    if en:
                        ll = ''.join(en).split()
                        for l in ll:
                            tokens.append(l)
                        en = list()
                        # tokens.append(c)
                else:
                    en.append(c)
        if en:
            ll = ''.join(en).split()
            for l in ll:
                tokens.append(l)
    else:
        tokens = [w for w in list(jieba.cut(sent.strip()))
                  if w not in STOP_WORDS]
    return tokens


def jieba_cut(query, smart=True):
    seg = jieba.cut(query, cut_all=not smart)
    result = []
    for s in seg:
        result.append(s)
    return result


def rule_base_num_retreive(query):
    inch_dual = r".*(([-+]?\d*\.\d+|\d+)[到|至]([-+]?\d*\.\d+|\d+)寸).*"
    meter_dual = r".*([-+]?\d*\.\d+|\d+)[到|至]([-+]?\d*\.\d+|\d+)米.*"
    ac_power_dual = r".*([-+]?\d*\.\d+|\d+)[到|至]([-+]?\d*\.\d+|\d+)[P|匹].*"
    price_dual = r".*([-+]?\d*\.\d+|\d+)[到|至]([-+]?\d*\.\d+|\d+)[块|元].*"

    inch_single = r".*([-+]?\d*\.\d+|\d+)寸.*"
    meter_single = r".*([-+]?\d*\.\d+|\d+)米.*"
    ac_power_single = r".*([-+]?\d*\.\d+|\d+)[P|匹].*"
    price_single = r".*([-+]?\d*\.\d+|\d+)[块|元].*"

    dual = {"_inch_": inch_dual, "_meter_": meter_dual, "ac.power": ac_power_dual,
            "price": price_dual}
    single = {"_inch_": inch_single, "_meter_": meter_single, "ac.power": ac_power_single,
              "price": price_single}

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
    price_dual_default = r".*([-+]?\d*\.\d+|\d+)[到|至]([-+]?\d*\.\d+|\d+)(?!P|匹|米|寸).*"
    price_single_default = r".*([-+]?\d*\.\d+|\d+)(?!P|匹|米|寸).*"
    remove_regex = r"\d+[个|只|条|部|本|台]"
    query = re.sub(remove_regex, '', query)
    render, numbers = range_extract(price_dual_default, query, False, True)
    if numbers:
        wild_card['price'] = numbers
        return render, wild_card
    render, numbers = range_extract(price_single_default, query, True, True)
    if numbers:
        wild_card['price'] = numbers
    return render, wild_card


def range_extract(pattern, query, single, range_render=False):
    """

    :param pattern:
    :param query:
    :param single:
    :return:
    """
    numbers = []
    match = re.match(pattern, query)
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
    print(' '.join(jieba_cut('华为num元手机phone.mmem')))
    print(rule_base_num_retreive('华为num元手机phone.mmem'))
    print(tokenize('华为num元手机phone.mmem', char=0))
