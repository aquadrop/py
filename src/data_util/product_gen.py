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

ac_power = [1.0, 1.5, 2, 2.5, 3, 3, 5]
ac_type = ["圆柱", "立式", "挂壁式", "立柜式", "中央空调"]
ac_brand = ["三菱", "松下", "科龙", "惠而浦", "大金", "目立", "海尔", "美的", "卡萨帝",
            "奥克斯", "长虹", "格力", "莱克", "艾美特", "dyson", "智高", "爱仕达", "格兰仕"]
price = [2000, 15000]
location = ["一楼", "二楼", "三楼", "地下一楼"]
ac_fr = ["变频", "定频"]
ac_cool_type = ["单冷", "冷暖"]
discount = ["满1000减200", "满2000减300", "十一大促销"]

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


def ac_product_gen(product_file):
    with open(product_file, 'w') as output:
        for i in range(500):
            ac = dict()
            for key, value in ac_content.items():
                data = np.random.choice(value)
                ac[key] = data
            ac['price'] = np.random.randint(low=price[0], high=price[1])
            ac['category'] = '空调'
            if np.random.uniform() < 0.4:
                ac['discount'] = np.random.choice(discount)
            ac['title'] = ac['brand'] + " " + ac["ac.fr"] + " " + ac["ac.type"] + "空调"

            output.write(json.dumps(ac, ensure_ascii=False) + '\n')


def tv_product_gen(product_file):
    with open(product_file, 'w') as output:
        for i in range(500):
            tv = dict()
            for key, value in tv_content.items():
                data = np.random.choice(value)
                tv[key] = data
            tv['price'] = np.random.randint(low=price[0], high=price[1])
            if np.random.uniform() < 0.4:
                tv['discount'] = np.random.choice(discount)
            tv['category'] = '电视机'
            tv['title'] = tv['brand'] + " " + tv["tv.resolution"] + " " + tv["tv.type"]+ " " + tv["tv.panel"] + "电视机"

            output.write(json.dumps(tv, ensure_ascii=False) + '\n')

def phone_product_gen(product_file):
    with open(product_file, 'w') as output:
        for i in range(500):
            phone = dict()
            for key, value in phone_content.items():
                data = np.random.choice(value)
                phone[key] = data
                phone['price'] = np.random.randint(low=price[0], high=price[1])
            if phone["brand"] == "苹果":
                phone["phone.sys"] = "ios"
            else:
                phone["phone.sys"] = "android"
            if np.random.uniform() < 0.4:
                phone['discount'] = np.random.choice(discount)
            phone['category'] = '手机'
            phone['title'] = phone['brand'] + " " + phone["phone.memsize"] + " " + phone["phone.net"] + "手机"
            output.write(json.dumps(phone, ensure_ascii=False) + '\n')

def update_solr(solr_file):

    with open(solr_file, 'rb') as data_file:
        for line in data_file:
            line = "[" + line.decode("utf-8") + "]"
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
    phone_product_gen("../../data/raw/product_phone.txt")
    ac_product_gen("../../data/raw/product_ac.txt")
    tv_product_gen("../../data/raw/product_tv.txt")
    print('updating')
    update_solr("../../data/raw/product_tv.txt")
    update_solr("../../data/raw/product_ac.txt")
    update_solr("../../data/raw/product_phone.txt")
