import os
import sys
import urllib.request
import json
import requests
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)

def update_solr(solr_file):

    with open(solr_file, 'r', encoding='utf-8') as data_file:
        for line in data_file:
            try:
                g, _, b = line.strip('\n').split('\t')
                doc = dict()
                doc['g'] = g
                doc['b'] = b
                line = json.dumps(doc, ensure_ascii=False)
                line = "[" + line + "]"
                # line = str.encode(line)
                print(line)
                line = str.encode(line)
                req = urllib.request.Request(url='http://10.89.100.12:11403/solr/faq/update?commit=true',
                                      data=line)
                headers = {"content-type": "text/json"}
                req.add_header('Content-type', 'application/json')
                f = urllib.request.urlopen(req)
                # Begin using data like the following
                print
                f.read()
            except:
                print(line)


if __name__ == "__main__":
    # phone_product_gen("../../data/raw/product_phone.txt", '../../data/gen_product/shouji.txt')
    # ac_product_gen("../../data/raw/product_ac.txt", '../../data/gen_product/kongtiao.txt')
    # tv_product_gen("../../data/raw/product_tv.txt", '../../data/gen_product/dianshi.txt')
    # pc_product_gen("../../data/raw/pc.txt", '../../data/gen_product/pc.txt')
    # fr_product_gen("../../data/raw/product_fr.txt", '../../data/gen_product/bingxiang.txt')
    print('updating')
    update_solr("../../data/memn2n/train/faq/discount.txt")