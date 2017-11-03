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



if __name__ == '__main__':
    # metadata_dir = os.path.join(
    #     grandfatherdir, 'data/memn2n/processed/metadata.pkl')
    # data_dir = os.path.join(
    #     grandfatherdir, 'data/memn2n/processed/data.pkl')
    # ckpt_dir = os.path.join(grandfatherdir, 'model/memn2n/ckpt')
    # config = {"belief_graph": "../../model/graph/belief_graph.pkl",
    #           "solr.facet": 'on',
    #           "metadata_dir": os.path.join(grandfatherdir, 'data/memn2n/processed/archive/metadata.pkl'),
    #           "data_dir": os.path.join(grandfatherdir, 'data/memn2n/processed/archive/data.pkl'),
    #           "ckpt_dir": os.path.join(grandfatherdir, 'model/memn2n/ckpt'),
    #           "gbdt_model_path": grandfatherdir + '/model/ml/belief_clf.pkl',
    #           "clf": 'memory'  # or memory
    #           }
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
    out_file = open("../../data/testresult/qatext.txt","w")
    count = 0
    while True:
        if count == 100:
            break
        ipt = str(np.random.choice(model_list) + np.random.choice(product_list))
        out_file.write("Q:"+ipt + "\n")
        response, api ,resp, avails = kernel.kernel1(ipt)
        out_file.write("A:" + resp + "\n")
        print(resp, api, response, avails)
        categorys = api.split(":")
        if len(categorys)!=1:
            category = categorys[-1]
        while True:
            if response.endswith("price"):
                ipt = str(random.randint(1, 10)*1000)
            else:
                ipt = str(np.random.choice(avails))
            #print(ipt)
            out_file.write("Q:" + ipt + "\n")
            response, api, resp, avails = kernel.kernel1(ipt)
            out_file.write("A:" + resp + "\n")
            #kernel.kernel1("clear")
            if resp.startswith("为您推荐"):
                out_file.write("\n")
                print(resp)
                kernel.kernel1("clear")
                count = count +1
                break