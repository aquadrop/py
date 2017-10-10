import os
import sys

prefix_dir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))


data_files = ['data/gen_product/shouji.txt',
              'data/gen_product/kongtiao.txt',
              'data/gen_product/bingxiang.txt',
              'data/gen_product/dianshi.txt']

data_files = [os.path.join(prefix_dir, file) for file in data_files]
