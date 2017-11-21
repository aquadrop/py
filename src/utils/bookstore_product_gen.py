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