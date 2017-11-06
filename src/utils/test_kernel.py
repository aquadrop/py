import os
import sys
import random
import numpy as np
import logging
import time
from datetime import datetime

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
grandfatherdir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parentdir)
sys.path.append(grandfatherdir)
import traceback

from kernel.main_kernel import *
from kernel.main_kernel import MainKernel

# 读取所有的产品属性中英文映射到dic中
def eachFile(filepath,dic):
    pathDir =  os.listdir(filepath)
    for allDir in pathDir:
        child = os.path.join('%s%s' % (filepath, allDir))
        #打开文件读取文件
        f = open(child,"r")
        lines = f.readlines()
        for line in lines:
            if line.startswith("+"):
                words = line.split("|")
                dic[words[2]] = words[1]


if __name__ == '__main__':
    config = {"belief_graph": "../../model/graph/belief_graph.pkl",
              "solr.facet": 'on',
              "metadata_dir": os.path.join(grandfatherdir, 'data/memn2n/processed/metadata.pkl'),
              "data_dir": os.path.join(grandfatherdir, 'data/memn2n/processed/data.pkl'),
              "ckpt_dir": os.path.join(grandfatherdir, 'model/memn2n/ckpt'),
              "gbdt_model_path": grandfatherdir + '/model/ml/belief_clf.pkl',
              "renderer_file": os.path.join(grandfatherdir, 'model/render/render_api.txt'),
              "renderer_location_file": os.path.join(grandfatherdir, 'model/render/render_location.txt'),
              "clf": 'memory'  # or memory
              }
    kernel = MainKernel(config)
    model_list = ['我想买','我要买','我准备买','我想看看']
    product_list = ['手机','冰箱','空调','电视','电脑']
    dic = dict()
    eachFile("../../data/gen_product/", dic)
    print(dic)
    out_file = open("../../data/testresult/qatext.txt","w")
    count = 0
    flag = False  # 标记上一轮对话是否反问
    while True:
        if count == 10:
            out_file.close()
            break
        ipt = str(np.random.choice(model_list) + np.random.choice(product_list))
        flag = False
        out_file.write("Q:"+ipt + "\n")
        response, api ,resp, avails = kernel.kernel1(ipt)
        oldavails = avails    #备份选项
        out_file.write("A:" + resp + "\n")
        print(resp, api, response, avails)
        categorys = api.split(":")
        if len(categorys)!=1:
            category = categorys[-1]
        while True:
            try:
                if flag:
                    ipt = str(np.random.choice(oldavails))
                    flag = False
                else:
                    if response.endswith("price"):  #如果询问价格
                        ipt = str(random.randint(1, 10) * 1000)
                        flag = False
                    else:
                        if random.randint(1, 10) > 6:  #60%的关于品牌询问的加反问
                            ends = response.split("_")[-1]   #获取当前的询问的种类[brand,fr.size,fr.cool_type]
                            ipt = str(category + "有哪些"+dic[ends])
                            flag = True
                        else:
                            ipt = str(np.random.choice(avails))
                            flag = False
                out_file.write("Q:" + ipt + "\n")
                oldavails = avails
                response, api, resp, avails = kernel.kernel1(ipt)
                out_file.write("A:" + resp + "\n")
                if "为您推荐" in resp:
                    out_file.write("\n")
                    print(resp)
                    kernel.kernel1("clear")
                    count = count +1
                    break
            except BaseException:
                out_file.write("\n")
                break