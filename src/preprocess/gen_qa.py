#!/usr/bin/env python3
import os,sys
import random
import json

def load_synonyms(filepath):
    Synonyms = {}
    for line in open(filepath, encoding='utf8'):
        line = line.strip()
        if not line:
            continue
        line = line.split('|')
        for x in line[2].split('/'):
            Synonyms[line[0]+':'+x] = line[0]+':'+line[1]
    return Synonyms

dirpath = os.path.dirname(os.path.abspath(__file__))
Synonyms = load_synonyms(os.path.join(dirpath, 'synonyms'))

def load_template(filepath):
    Template = []
    for line in open(filepath, encoding='utf-8'):
        line = line.strip()
        if not line:
            continue
        Template.append(line)
    return Template

def load_attrs(filepath):
    Attrs = {}
    for line in open(filepath, encoding='utf-8'):
        line = line.strip()
        if not line:
            continue
        line = line.split('|')
        if line[0] not in Attrs.keys():
            Attrs[line[0]] = line[1].split('/')
        else:
            Attrs[line[0]] += line[1].split('/')
    return Attrs

def gen_sentence(Template, Attrs, Sentences, Type):
    for sen in Template:
        data = {'question':'', 'type':Type, 'entity':[]}
        for x in Attrs.keys():
            if '['+x+']' in sen:
                entity = random.sample(Attrs[x], 1)[0]
                sen = sen.replace('['+x+']', entity)
                if x+':'+entity in Synonyms:
                    data['entity'].append(Synonyms[x+':'+entity])
                else:
                    data['entity'].append(x+':'+entity)
        data['question'] = sen
        data['entity'] = sorted(data['entity'])
        if data not in Sentences:
            Sentences.append(data)

def gen(mode, num):
    Sentences = []
    Template = load_template(mode+'_template')
    Attrs = load_attrs(mode+'_attrs')
    for i in range(num):
        gen_sentence(Template, Attrs, Sentences, mode.split('/')[-1])
    return Sentences

if __name__ == '__main__':
    # N = 3
    # if os.path.exists(os.path.join(dirpath, 'where')):
    #     os.remove(os.path.join(dirpath, 'where'))
    # s1 = gen(os.path.join(dirpath, 'where'), N)
    # for x in s1:
    #     s = x['question'] + '\t'
    #     s += 'api_call_query_location_'+','.join(x['entity'])
    #     s = s.replace('','')
    #     print(s, file=open(os.path.join(dirpath, 'where'), 'a', encoding='utf-8'))
    # print(len(s1))

    N = 3000
    if os.path.exists(os.path.join(dirpath, 'discount')):
        os.remove(os.path.join(dirpath, 'discount'))
    s2 = gen(os.path.join(dirpath, 'discount'), N)
    for x in s2:
        s = x['question'] + '\t'
        s += 'api_call_query_discount' #+','.join(x['entity'])
        s = s.replace('','')
        print(s, file=open(os.path.join(dirpath, 'discount'), 'a', encoding='utf-8'))
    print(len(s2))
