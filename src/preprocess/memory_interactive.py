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
"""
from __future__ import division
from __future__ import print_function

import argparse
import os
import random
import sys
import time
import inspect
import json

import uuid

import re

topic_sign = ['一.','二.','三.','四.','五.','六.','七.']
talk_sign = r'^[0-9]+.*$'
talk_pattern = re.compile(talk_sign)
guest_sign = r'G:.*'
guest_pattern = re.compile(guest_sign)
bot_sign = r'B:.*'
bot_pattern = re.compile(bot_sign)

def get_topic(line):
    tt = []
    start = False
    for c in line:
        if c == ':':
            start = True
            continue
        if start:
            tt.append(c)
    return ''.join(tt)

def topic_start(line):
    return '话题:' in line

def interactive(file_, write_file_):
    D = []
    with open(file_, 'r', encoding='utf-8') as f:
        data = []
        for line in f:
            line = line.strip('\n')
            # line = unicodedata.normalize('NFKC', line)
            if topic_start(line):
                continue
            if talk_pattern.match(str(line)):
                line = re.sub('^[0-9]+(.)', '', str(line)).strip()
                if len(data) > 1:
                    D.append(data)
                data = []
                data.append(line)
                continue
            if guest_pattern.match(str(line)):
                data.append(line)
            if bot_pattern.match(str(line)):
                data.append(line)
    if len(data) > 1:
        D.append(data)
    newD = []
    mapper = dict()
    last_g = ''
    for data in D:
        for d in data:
            if d.startswith('G:'):
                g = d.replace('G:', '').strip('\n')
                more = g.split('/')
                for m in more:
                    mapper[m] = more[0]
                # if g not in mapper:
                #     mapper[g] = 'api_call_base_' + str(index)
                #     index += 1
            if d.startswith('B'):
                b = d.replace('B:', '').strip('\n')
                if g == last_g:
                    continue
                last_g = g
                newD.append(g + '\t' + g + '\t' + b)
        newD.append('')

    return newD, mapper
    # with open(write_file_,'w') as f:
    #     json.dump(D, f, ensure_ascii=False)


if __name__ == '__main__':
    D1, c1 = interactive('整理后的客服接待语料.txt','base-all.txt')
    D2, c2 = interactive('2017互动话术汇总版4.10.txt','train.txt')

    with open('train.txt','w', encoding='utf-8') as f:
        for a in D1:
            f.writelines(a + '\n')
        for a in D2:
            f.writelines(a + '\n')

    c1.update(c2)
    with open('candidates.txt','w', encoding='utf-8') as f:
        for a in c1:
            f.writelines(a + '\n')