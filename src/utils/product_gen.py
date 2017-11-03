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

ac_power = [1.0, 1.5, 2, 1.5, 3, 2.5, 4]
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
              "tv.distance": tv_distance, "tv.resolution": tv_resolution, "tv.panel": tv_panel, "location": location}

phone_brand = ["华为", "oppo", "苹果", "vivo",
               "金立", "三星", "荣耀", "魅族", "moto", "小米"]
phone_sys = ["android", "ios"]
phone_net = ["全网通", "移动4G", "联通4G", "电信4G", "双卡双4G", "双卡单4G"]
phone_feature = ["老人手机", "拍照神器", "女性手机", "儿童手机"]
phone_color = ["红", "黑", "白", "深空灰", "玫瑰金"]
phone_mem_size = ["16G", "32G", "64G", "128G", "256G"]

phone_content = {"brand": phone_brand, "phone.sys": phone_sys, "phone.net": phone_net, "phone.feature": phone_feature,
                 "phone.color": phone_color, "phone.memsize": phone_mem_size, "location": location}

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
            ac['price'] = np.random.randint(low=2000, high=10000)
            ac['category'] = '空调'
            if np.random.uniform() < 0.4:
                ac['discount'] = np.random.choice(discount)
            # ac.power
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
            ac['ac.power_float'] = power
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
            ac['price'] = np.random.randint(low=2000, high=15000)
            if ac["brand"] == "索尼":
                ac['price'] = np.random.randint(low=8000, high=15000)
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
            ac['price'] = np.random.randint(low=2000, high=14000)
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
            ac['price'] = np.random.randint(low=2000, high=10000)
            if np.random.uniform() < 0.4:
                ac['discount'] = np.random.choice(discount)
            ac["phone.series"] = ""
            if ac["brand"] == "苹果":
                ac["phone.sys"] = "ios"
                series = "Apple iPhone6,\
Apple iPhone6s Plus,\
Apple iPhone7,\
Apple iPhone7 Plus,\
Apple iPhone8,\
Apple iPhone8 Plus\
".split(",")
                prices = [2500,
4000,
4500,
5000,
7000,
8000]
                index = np.random.randint(len(series))
                #print(len(prices), len(series))
                ac["phone.series"] = series[index]
                ac['price'] = np.random.randint(low=prices[index] * 0.96, high=prices[index] * 1.04)
            else:
                ac["phone.sys"] = "android"
            if ac["brand"] == "华为":
                series = "华为麦芒6,\
华为畅想6,\
华为畅想6S,\
华为畅想7,\
华为畅想7 Plus,\
华为麦芒5,\
华为mate9(MHA-AL00),\
华为nova2 Plus(BAC-AL00),\
华为P10 Plus,\
华为mate10,\
华为nova 青春版\
".split(",")
                prices = [2500,
800,
1300,
1000,
1500,
1600,
4000,
2500,
4500,
5000,
1500
]

                index = np.random.randint(len(series))
                #print(len(prices), len(series))
                ac["phone.series"] = series[index]
                ac['price'] = np.random.randint(low=prices[index] * 0.96, high=prices[index] * 1.04)
            if ac["brand"] == "三星":
                series = "Galaxy S8(G9500),\
Galaxy S8 Plus S8(G9550),\
Galaxy C7 Pro(C7010),\
Galaxy C7 (C700),\
Galaxy S7 edge(G9350),\
Galaxy C5 (C500),\
Galaxy S7 (G9300)\
".split(",")
                prices = [5500,
6000,
3000,
2000,
3500,
1500,
2500
]

                index = np.random.randint(len(series))
                #print(len(prices), len(series))
                ac["phone.series"] = series[index]
                ac['price'] = np.random.randint(low=prices[index] * 0.96, high=prices[index] * 1.04)
            if ac["brand"] == "小米":
                series = "小米5s,\
小米5s Plus,\
小米6,\
小米note2,\
小米5C,\
小米MIX\
".split(",")
                prices = [2000,
2500,
3000,
2800,
1300,
3500
]

                index = np.random.randint(len(series))
                #print(len(prices), len(series))
                ac["phone.series"] = series[index]
                ac['price'] = np.random.randint(low=prices[index] * 0.96, high=prices[index] * 1.04)
            if ac["brand"] == "魅族":
                series = "魅蓝note5,\
魅族MX6,\
魅蓝E,\
魅蓝E2,\
魅蓝note6,\
魅蓝Max,\
魅族PRO6S,\
魅族PRO6 Plus,\
魅蓝X".split(",")
                prices = [1000,
                            1500,
                            1000,
                            1200,
                            1500,
                            1500,
                            2000,
                            2200,
                            1000
                          ]

                index = np.random.randint(len(series))
                # print(len(prices), len(series))
                ac["phone.series"] = series[index]
                ac['price'] = np.random.randint(low=prices[index] * 0.96, high=prices[index] * 1.04)
            if ac["brand"] == "OPPO":
                series = "OPPO R11,\
OPPO R11 Plus,\
OPPO A77,\
OPPO A57,\
OPPO A37,\
OPPO A59s\
".split(",")
                prices = [3000,
3700,
2200,
1500,
1200,
1800]

                index = np.random.randint(len(series))
                #print(len(prices), len(series))
                ac["phone.series"] = series[index]
                ac['price'] = np.random.randint(low=prices[index] * 0.96, high=prices[index] * 1.04)
            if ac["brand"] == "VIVO":
                series = "vivo X20A,\
vivo Y66,\
vivo X9,\
vivo Y67,\
vivo Y66,\
vivo Y55,\
vivo X9s Plus,\
vivo X20 Plus,\
vivo Xplay6,\
vivo Y79A\
".split(",")
                prices = [3000,
1500,
2500,
1800,
1300,
1000,
3000,
3500,
4000,
2300
]
                index = np.random.randint(len(series))
                #print(len(prices), len(series))
                ac["phone.series"] = series[index]
                #print(len(prices), len(series))
                ac['price'] = np.random.randint(low=prices[index] * 0.96, high=prices[index] * 1.04)
            if ac["brand"] == "美图":
                series = "美图T8,\
美图T8s,\
美图M8,\
美图M8s,\
MEITU M8 hellokitty,\
美图M8 美少女战士\
".split(",")
                prices = [3600,
4200,
2800,
3000,
3200,
4000
]
                index = np.random.randint(len(series))
                #print(len(prices), len(series))
                ac["phone.series"] = series[index]
                ac['price'] = np.random.randint(low=prices[index] * 0.96, high=prices[index] * 1.04)
            if ac["brand"] == "努比亚":
                series = "努比亚Z17S(NX595J),\
努比亚Z17 畅享版,\
努比亚Z17mini,\
努比亚Z17(NX563J)\
".split(",")
                prices = [3000,
2500,
1500,
3200
]
                index = np.random.randint(len(series))
                #print(len(prices), len(series))
                ac["phone.series"] = series[index]
                ac['price'] = np.random.randint(low=prices[index] * 0.96, high=prices[index] * 1.04)
            if ac["brand"] == "索尼":
                ac["phone.series"] = np.random.choice("xperia".split(","))
                ac['price'] = np.random.randint(low=8000, high=10000)
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
            ac['price'] = np.random.randint(low=3000, high=15000)
            if np.random.uniform() < 0.4:
                ac['discount'] = np.random.choice(discount)
            ac["pc.series"] = ""
            if ac["brand"] == "苹果":
                ac["pc.sys"] = "macos"
                if ac['pc.type'] == '台式':
                    ac["pc.series"] = np.random.choice(
                        "iMac,macpro".split(","))
                else:
                    ac["pc.series"] = np.random.choice(
                        "macbook air,macbook pro".split(","))
                ac['price'] = np.random.randint(low=7000, high=15000)
            else:
                ac["pc.sys"] = np.random.choice(["chromeos", "windows"])
            if ac["brand"] == "索尼":
                ac["phone.series"] = np.random.choice("vaio".split(","))
                ac['price'] = np.random.randint(low=8000, high=15000)
            tt = []
            for t in title:
                tt.append(ac[t])

            ac['title'] = " ".join(tt)

            output.write(json.dumps(ac, ensure_ascii=False) + '\n')


def household_product_gen(product_file, data_file):
    profile = dict()
    title = []
    _data_file = data_file.replace(" ", "")
    _product_file = product_file.replace(" ", "")
    data_file = "../../data/gen_product/" + _data_file.replace(" ","")
    product_file = '../../data/raw/' + _product_file.replace(" ","")
    with open(data_file, 'r') as infile:
        line = infile.readline()
        title = line.strip('\n').split("|")[4].split(',')
        for line in infile:
            line = line.replace(' ', '').strip('\n')
            mark, a, b, _, c = line.split("|")
            if mark == '*':
                continue
            profile[b] = c.split(",")

    with open(product_file, 'a') as output:
        for i in range(5):
            household = dict()
            for key, value in profile.items():
                data = np.random.choice(value)
                household[key] = data
            household['category'] = profile['category'][0]
            household['price'] = np.random.randint(low=200, high=1000)
            if np.random.uniform() < 0.4:
                household['discount'] = np.random.choice(discount)
            print(household)
            tt = []
            for t in title:
                tt.append(household[t])
            household['title'] = " ".join(tt)

            output.write(json.dumps(household, ensure_ascii=False) + '\n')


def update_solr(solr_file):

    with open(solr_file, 'r', encoding='utf-8') as data_file:
        for line in data_file:
            line = "[" + line + "]"
            # line = str.encode(line)
            print(line)
            line = str.encode(line)
            req = urllib.request.Request(url='http://10.89.100.12:11403/solr/category/update?commit=true',
                                         data=line)
            headers = {"content-type": "text/json"}
            req.add_header('Content-type', 'application/json')
            f = urllib.request.urlopen(req)
            # Begin using data like the following
            print
            f.read()


if __name__ == "__main__":
    categories = ["净水器.txt",
                  "剃毛器.txt", "加湿器.txt", "取暖器.txt", "吸尘器.txt",
                  '咖啡机.txt', '垃圾处理机.txt', '多用途锅.txt', '干衣机.txt',
                  '微波炉.txt', '打蛋器.txt', '扫地机器人.txt', '挂烫机.txt',
                  '按摩器.txt', '按摩椅.txt', '排气扇.txt', '搅拌机.txt',
                  '料理机.txt', '榨汁机.txt', '油烟机.txt', '洗碗机.txt',
                  '洗衣机.txt', '浴霸.txt', '消毒柜.txt', '烟灶套装.txt',
                  '烤箱.txt', '热水器.txt', '煮蛋器.txt', '燃气灶.txt',
                  '电动剃须刀.txt', '电动牙刷.txt', '电压力锅.txt', '电吹风.txt',
                  '电子秤.txt', '电水壶.txt', '电炖锅.txt', '电磁炉.txt',
                  '电蒸炉.txt', '电风扇.txt', '电饭煲.txt', '电饼铛.txt',
                  '相机.txt', '空气净化器.txt', '空调扇.txt', '美发器.txt',
                  '美容器.txt', '豆浆机.txt', '足浴盆.txt', '酸奶机.txt',
                  '采暖炉.txt', '除湿机.txt', '集成灶.txt', '面包机.txt',
                  "饮水机.txt"]
    prefix = '../../data/gen_product/'
    data_files = [os.path.join(prefix, d) for d in categories]
    product_file = "../../data/raw/household.txt"
    # for data_file in data_files:
    #     household_product_gen(product_file, data_file)

    # phone_product_gen("../../data/raw/product_phone.txt", '../../data/gen_product/手机.txt')
    # ac_product_gen("../../data/raw/product_ac.txt", '../../data/gen_product/空调.txt')
    # tv_product_gen("../../data/raw/product_tv.txt", '../../data/gen_product/电视.txt')
    # pc_product_gen("../../data/raw/pc.txt", '../../data/gen_product/电脑.txt')
    # fr_product_gen("../../data/raw/product_fr.txt", '../../data/gen_product/冰箱.txt')

    # update_solr("../../data/raw/product_ac.txt")
    # update_solr("../../data/raw/pc.txt")
    # update_solr("../../data/raw/product_tv.txt")
    #
    update_solr("../../data/raw/product_phone.txt")
    # update_solr("../../data/raw/product_fr.txt")
    # update_solr("../../data/raw/household.txt")
