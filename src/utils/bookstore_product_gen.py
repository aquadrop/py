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

"""
苹果/苹果手机/ipad/iphone|一楼	|一楼 数码产品为主，中心区域为精品图书展台，（数码品牌区有苹果、OPPO、Vivo、华为、小天才）。	屏幕显示一楼平面图
OPPO| 欧珀| 欧珀手机			
Vivo|维沃| 维沃手机			
华为|华为手机			
小天才| 学习机| 小天才学习机			
精品图书展台| 图书展台| 精品展台			
			
茶颜观色| 餐饮区| 	二楼	二楼东侧 是茶颜观色餐饮区(咖啡、奶茶、简餐)	屏幕显示二楼平面图
咖啡| 喝咖啡| 奶茶| 喝奶茶|简餐| 			
小确幸| 绿植区| 小确幸绿植区	二楼	二楼西侧 是小确幸绿植区以及亨通市民书房。	
书房| 亨通书房|市民书房|亨通市民书房			
文学类| 文学书| 杂志|期刊|杂志期刊	二楼	二楼：文学社科、科技生活图书	
社会科学| 社科书| 科学书			
科技生活| 科技书|生活书			
			
诺亚舟| 电子词典	三楼	三楼：少儿图书，电教产品（诺亚舟、步步高、读书郎）创想学习桌！	屏幕显示三楼平面图
步步高| 学习平板电脑			
读书郎| 点读机			
创想学习桌| 创想桌| 学习桌			
			
文具类| 文化用品	四楼	四楼：文教图书及文化用品。	屏幕显示四楼平面图
教辅类书| 小升初|中考| 高考|			
			
服务总台|收银台	一楼	在一楼的电梯出来后 右手边| 在屏幕平面图的正上方	屏幕显示一楼平面图
卫生间	三楼	在三楼的电梯出来后 右手走到底| 咨询工作人员	屏幕显示三楼平面图
电梯	一楼	在屏幕平面图的右上方	屏幕显示一楼平面图
楼梯	一楼	在屏幕平面图的右上方	屏幕显示一楼平面图

* brand
* category
* location
* poi
* facility
* store_id
"""

store_id = "吴江新华书店"
line1 = [
{"brand":["苹果"],"virtual_category":"数码产品","category":"手机","pc.series":"iphone","location":"一楼"},
{"brand":["苹果"],"virtual_category":"数码产品","category":"平板","pc.series":"ipad","location":"一楼"},
{"brand":["oppo","欧珀"],"virtual_category":"数码产品","category":"手机","location":"一楼"},
{"brand":["vivo","维沃"],"virtual_category":"数码产品","category":"手机","location":"一楼"},
{"brand":["华为"],"virtual_category":"数码产品","category":"手机","location":"一楼"},
{"brand":["小天才"],"virtual_category":"数码产品","category":"学习机","location":"一楼"},
{"poi":["精品图书展台","图书展台","精品展台"],"location":"一楼中心区域"},
{"poi":["茶颜观色餐饮区"],"location":"二楼东侧"},
{"poi":["咖啡","奶茶","简餐"],"location":"二楼东侧"},
{"poi":["小确幸绿植区"],"location":"二楼西侧"},
{"poi":["亨通市民书房"],"location":"二楼西侧"},
{"category":["图书"],"book.category":["文学类"],"location":"二楼"},
{"category":["图书"],"book.category":["杂志","期刊","杂志期刊"],"location":"二楼"},
{"category":["图书"],"book.category":["社会科学","社科书","科学书"],"location":"二楼"},
{"category":["图书"],"book.category":["科技","科技书","生活书"],"location":"二楼"},
{"category":["图书"],"book.category":["生活书"],"location":"二楼"},
{"category":["图书"],"book.category":["少儿图书"],"location":"三楼"},
{"brand":["诺亚舟"],"virtual_category":"电教产品","category":"电子词典","location":"三楼"},
{"brand":["步步高"],"virtual_category":"电教产品","category":"学习平板电脑","location":"三楼"},
{"brand":["读书郎"],"virtual_category":"电教产品","category":"点读机","location":"三楼"},
{"brand":["创想"],"virtual_category":"电教产品","category":"学习桌","location":"三楼"},
{"category":["文具","文化用品"],"location":"四楼"},
{"category":["图书"],"book.category":["教辅","小升初","中考","高考"],"location":"四楼"},
{"facility":["服务总台","收银台"],"location":"在一楼的电梯出来后右手边,在屏幕平面图的正上方"},
{"facility":["卫生间","厕所"],"location":"在三楼的电梯出来后右手走到底,咨询工作人员"},
{"facility":["电梯"],"location":"在屏幕平面图的右上方"},
{"facility":["楼梯"],"location":"在屏幕平面图的右上方"}]



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
