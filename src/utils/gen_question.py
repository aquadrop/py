#!/usr/bin/env python3

import os
import sys
import random
import json

Data = []
sen_mode = []

import os
dir_path = os.path.dirname(os.path.realpath(__file__))

def read_attrs(filepath):
    product_attr = {}
    for line in open(filepath):
        line = line.strip()
        if not line:
            continue
        line = line.split('|')
        product_attr[line[0]] = line[1].split()
    return product_attr

def read_sen_mode(filepath):
    for line in open(filepath):
        line = line.strip()
        if not line:
            continue
        sen_mode.append(line)

def get_dic(d):
    product_attr = read_attrs(d['file'])
    dic = {}
    dic['name'] = random.sample(product_attr[d['name']], 1)[0]
    dic['brand'] = random.sample(product_attr[d['brand']], 1)[0]
    dic['price'] = random.sample(product_attr[d['price']], 1)[0]
    dic['Type'] = random.sample(product_attr[d['Type']], 1)[0]
    dic['size'] = random.sample(product_attr[d['size']], 1)[0]
    return dic

def gen_sen(dic):
    for s in sen_mode:
        cls = s.split('|')[0]
        new_s = s.split('|')[1]
        data = {'class':cls, 'question':''}
        if '[name]' in new_s:
            new_s = new_s.replace('[name]', dic['name'])
            data['name'] = dic['name']
        if '[brand]' in new_s:
            new_s = new_s.replace('[brand]', dic['brand'])
            data['brand'] = dic['brand']
        if '[price]' in new_s:
            new_s = new_s.replace('[price]', dic['price'])
            data['price'] = dic['price']
        if '[type]' in new_s:
            new_s = new_s.replace('[type]', dic['Type'])
            data['type'] = dic['Type']
        if '[size]' in new_s:
            new_s = new_s.replace('[size]', dic['size'])
            data['size'] = dic['size']
        data['question'] = new_s
        if data not in Data:
            Data.append(data)

if __name__ == '__main__':
    file0 = dir_path + "/../../data/gen_product/mode.txt"
    read_sen_mode(file0)
    file1 = dir_path + "/../../data/gen_product/kongtiao.txt"
    file2 = dir_path + "/../../data/gen_product/dianshi.txt"
    file3 = dir_path + "/../../data/gen_product/bingxiang.txt"
    file4 = dir_path + "/../../data/gen_product/shouji.txt"
    dic_kt = {'file':file1, 'name':'名称', 'brand':'品牌', 'price':'价格', 'Type':'类型', 'size':'匹数'}
    dic_ds = {'file':file2, 'name':'名称', 'brand':'品牌', 'price':'价格', 'Type':'类型', 'size':'屏幕尺寸'}
    dic_bx = {'file':file3, 'name':'名称', 'brand':'品牌', 'price':'价格', 'Type':'类型', 'size':'适用人数'}
    dic_sj = {'file':file4, 'name':'名称', 'brand':'品牌', 'price':'价格', 'Type':'操作系统', 'size':'屏幕尺寸'}
    for i in range(2000):
        gen_sen(get_dic(dic_kt))
        gen_sen(get_dic(dic_ds))
        gen_sen(get_dic(dic_bx))
        gen_sen(get_dic(dic_sj))

    for d in Data:
        print(json.dumps(d, ensure_ascii=False))
