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

Belief Graph
"""
import pickle
import os
import sys
import urllib.request
import requests
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)

import json
import numpy as np

ac_power = [1.0, 1.5, 2, 1.5, 3,2.5,4]
ac_type = ["圆柱", "立式", "挂壁式", "立柜式", "中央空调"]
ac_brand = ["三菱", "松下", "科龙", "惠而浦", "大金", "目立", "海尔", "美的", "卡萨帝",
            "奥克斯", "长虹", "格力", "莱克", "艾美特", "dyson", "智高", "爱仕达", "格兰仕"]
price = [1000, 20000]
location = ["一楼", "二楼", "三楼", "地下一楼"]
ac_fr = ["变频", "定频"]
ac_cool_type = ["单冷", "冷暖"]
discount = ["满1000减200", "满4000减300", "双十一大促销"]

ac_content = {"ac.power": ac_power, "ac.type": ac_type,
              "brand": ac_brand, "ac.fr": ac_fr, "ac.cool_type": ac_cool_type, "location": location}

tv_size = [2.5, 3, 42, 43, 39, 40, 50, 51, 55, 67, 70,
           52, 45, 46, 47, 48, 49, 58, 59, 60, ]
tv_type = ["智能", "普通", "互联网"]
tv_brand = ["索尼", "乐视", "三星", "海信", "创维", "TCL"]
tv_distance = [2, 2.5, 3, 3.5, 4, 0.2]
tv_resolution = ["4K超高清", "全高清", "高清", "其他"]
tv_panel = ["LED", "OLEC", "LCD", "等离子"]
tv_power_level = [1, 2, 3]

tv_content = {"tv.size": tv_size, "tv.type": tv_type, "brand": tv_brand,
              "tv.distance": tv_distance, "tv.resolution": tv_resolution, "tv.panel": tv_panel, "location":location}

phone_brand = ["华为","oppo","苹果","vivo","金立","三星","荣耀","魅族","moto","小米"]
phone_sys = ["android","ios"]
phone_net = ["全网通", "移动4G", "联通4G", "电信4G", "双卡双4G", "双卡单4G"]
phone_feature = ["老人手机", "拍照神器", "女性手机", "儿童手机"]
phone_color = ["红","黑","白","深空灰","玫瑰金"]
phone_mem_size = ["16G", "32G", "64G", "128G", "256G"]

phone_content = {"brand": phone_brand, "phone.sys":phone_sys, "phone.net":phone_net, "phone.feature":phone_feature,
                 "phone.color":phone_color, "phone.memsize":phone_mem_size, "location": location}

N = 4000
def ac_product_gen(product_file, data_file):
    profile = dict()
    title = []
    with open(data_file, 'r') as infile:
        line = infile.readline()
        title = line.strip('\n').split("|")[4].split(',')
        for line in infile:
            line = line.replace(' ', '').strip('\n')
            mark, a, b, _, c = line.split("|")
            if mark == '*':
                continue
            profile[b] = c.split(",")

    with open(product_file, 'w') as output:
        for i in range(N):
            ac = dict()
            for key, value in profile.items():
                data = np.random.choice(value)
                ac[key] = data
            ac['price'] = np.random.randint(low=price[0], high=price[1])
            ac['category'] = '空调'
            if np.random.uniform() < 0.4:
                ac['discount'] = np.random.choice(discount)
            ## ac.power
            power = float(ac['ac.power_float'])
            if power <= 1:
                ac['ac.power'] = '1P'
            elif power > 1 and power < 1.5:
                ac['ac.power'] = '大1P'
            else:
                ac['ac.power'] = ac['ac.power_float'] + 'P'
            tt = []
            for t in title:
                tt.append(ac[t])

            ac['title'] = " ".join(tt)

            output.write(json.dumps(ac, ensure_ascii=False) + '\n')


def tv_product_gen(product_file, data_file):
    profile = dict()
    title = []
    with open(data_file, 'r') as infile:
        line = infile.readline()
        title = line.strip('\n').split("|")[4].split(',')
        for line in infile:
            line = line.replace(' ', '').strip('\n')
            mark, a, b, _, c = line.split("|")
            if mark == '*':
                continue
            profile[b] = c.split(",")

    with open(product_file, 'w') as output:
        for i in range(N):
            ac = dict()
            for key, value in profile.items():
                data = np.random.choice(value)
                ac[key] = data
            ac['price'] = np.random.randint(low=price[0], high=price[1])
            if np.random.uniform() < 0.4:
                ac['discount'] = np.random.choice(discount)
            tt = []
            for t in title:
                tt.append(ac[t])

            ac['title'] = " ".join(tt)

            output.write(json.dumps(ac, ensure_ascii=False) + '\n')


def fr_product_gen(product_file, data_file):
    profile = dict()
    title = []
    with open(data_file, 'r') as infile:
        line = infile.readline()
        title = line.strip('\n').split("|")[4].split(',')
        for line in infile:
            line = line.replace(' ', '').strip('\n')
            mark, a, b, _, c = line.split("|")
            if mark == '*':
                continue
            profile[b] = c.split(",")

    with open(product_file, 'w') as output:
        for i in range(N):
            ac = dict()
            for key, value in profile.items():
                data = np.random.choice(value)
                ac[key] = data
            ac['price'] = np.random.randint(low=price[0], high=price[1])
            if np.random.uniform() < 0.4:
                ac['discount'] = np.random.choice(discount)
            tt = []
            for t in title:
                tt.append(ac[t])

            ac['title'] = " ".join(tt)

            output.write(json.dumps(ac, ensure_ascii=False) + '\n')


def phone_product_gen(product_file, data_file):
    profile = dict()
    title = []
    with open(data_file, 'r') as infile:
        line = infile.readline()
        title = line.strip('\n').split("|")[4].split(',')
        for line in infile:
            line = line.replace(' ', '').strip('\n')
            mark, a, b, _, c = line.split("|")
            if mark == '*':
                continue
            profile[b] = c.split(",")

    with open(product_file, 'w') as output:
        for i in range(N):
            ac = dict()
            for key, value in profile.items():
                data = np.random.choice(value)
                ac[key] = data
            ac['price'] = np.random.randint(low=price[0], high=price[1])
            if np.random.uniform() < 0.4:
                ac['discount'] = np.random.choice(discount)
            ac["phone.series"] = ""
            if ac["brand"] == "苹果":
                ac["phone.sys"] = "ios"
                ac["phone.series"] = np.random.choice("iphone,iphone6,iphone7,iphone8,iphonex,iphone6s,iphone7p".split(","))
            else:
                ac["phone.sys"] = "android"
            if ac["brand"] == "华为":
                ac["phone.series"] = np.random.choice("荣耀,P9".split(","))
            if ac["brand"] == "三星":
                ac["phone.series"] = np.random.choice("galaxy,s7".split(","))
            if ac["brand"] == "小米":
                ac["phone.series"] = np.random.choice("红米".split(","))
            tt = []
            for t in title:
                tt.append(ac[t])

            ac['title'] = " ".join(tt)

            output.write(json.dumps(ac, ensure_ascii=False) + '\n')

def pc_product_gen(product_file, data_file):
    profile = dict()
    title = []
    with open(data_file, 'r') as infile:
        line = infile.readline()
        title = line.strip('\n').split("|")[4].split(',')
        for line in infile:
            line = line.replace(' ', '').strip('\n')
            mark, a, b, _, c = line.split("|")
            if mark == '*':
                continue
            profile[b] = c.split(",")

    with open(product_file, 'w') as output:
        for i in range(N):
            ac = dict()
            for key, value in profile.items():
                data = np.random.choice(value)
                ac[key] = data
            ac['price'] = np.random.randint(low=price[0], high=price[1])
            if np.random.uniform() < 0.4:
                ac['discount'] = np.random.choice(discount)
            ac["pc.series"] = ""
            if ac["brand"] == "苹果":
                ac["pc.sys"] = "macos"
                ac["pc.series"] = np.random.choice("macbookair,macbookpro".split(","))
            else:
                ac["pc.sys"] = np.random.choice(["chromeos","windows"])
            if ac["brand"] == "索尼":
                ac["phone.series"] = np.random.choice("vaio".split(","))
            tt = []
            for t in title:
                tt.append(ac[t])

            ac['title'] = " ".join(tt)

            output.write(json.dumps(ac, ensure_ascii=False) + '\n')


def update_solr(solr_file):

    with open(solr_file, 'r', encoding='utf-8') as data_file:
        for line in data_file:
            line = "[" + line + "]"
            # line = str.encode(line)
            print(line)
            line = str.encode(line)
            req = urllib.request.Request(url='http://localhost:11403/solr/category/update?commit=true',
                                  data=line)
            headers = {"content-type": "text/json"}
            req.add_header('Content-type', 'application/json')
            f = urllib.request.urlopen(req)
            # Begin using data like the following
            print
            f.read()


if __name__ == "__main__":
    phone_product_gen("../../data/raw/product_phone.txt", '../../data/gen_product/shouji.txt')
    ac_product_gen("../../data/raw/product_ac.txt", '../../data/gen_product/kongtiao.txt')
    tv_product_gen("../../data/raw/product_tv.txt", '../../data/gen_product/dianshi.txt')
    pc_product_gen("../../data/raw/pc.txt", '../../data/gen_product/pc.txt')
    fr_product_gen("../../data/raw/product_fr.txt", '../../data/gen_product/bingxiang.txt')
    print('updating')
    update_solr("../../data/raw/pc.txt")
    update_solr("../../data/raw/product_tv.txt")
    update_solr("../../data/raw/product_ac.txt")
    update_solr("../../data/raw/product_phone.txt")
    update_solr("../../data/raw/product_fr.txt")
