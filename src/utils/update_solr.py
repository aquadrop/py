import os
import sys
import urllib.request
import json
import traceback
import requests
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from SolrClient import SolrClient
sys.path.insert(0, parentdir)

IP = "10.89.100.12"
solr_client = SolrClient('http://{}:11403/solr'.format(IP))

def update_solr(solr_file, cls='base'):

    with open(solr_file, 'r', encoding='utf-8') as data_file:
        for line in data_file:
            try:
                line = line.strip('\n').replace(' ', '')
                if not line:
                    continue
                tokens = line.split('\t')
                print(tokens)
                uid = tokens[0]
                r = solr_client.delete_doc_by_query('base', "uid:"+uid)
                print(r)
                representative_q = tokens[1]
                answer = tokens[2].split('/')
                question = tokens[3].split('/')
                try:
                    emotion = tokens[4]
                    media = tokens[5]
                except:
                    emotion = 'null'
                    media = 'null'
                doc = dict()
                doc['question'] = question
                doc['answer'] = answer
                doc['uid'] = uid
                doc['representative_q'] = representative_q
                doc['emotion'] = emotion
                doc['media'] = media
                doc['store_id'] = '吴江新华书店'
                doc['class'] = cls
                line = json.dumps(doc, ensure_ascii=False)
                line = "[" + line + "]"
                # line = str.encode(line)
                print(line)
                line = str.encode(line)
                req = urllib.request.Request(url='http://{}:11403/solr/base/update?commit=true'.format(IP),
                                      data=line)
                headers = {"content-type": "text/json"}
                req.add_header('Content-type', 'application/json')
                f = urllib.request.urlopen(req)
                # Begin using data like the following
                print
                f.read()
            except:
                traceback.print_exc()


if __name__ == "__main__":
    # phone_product_gen("../../data/raw/product_phone.txt", '../../data/gen_product/shouji.txt')
    # ac_product_gen("../../data/raw/product_ac.txt", '../../data/gen_product/kongtiao.txt')
    # tv_product_gen("../../data/raw/product_tv.txt", '../../data/gen_product/dianshi.txt')
    # pc_product_gen("../../data/raw/pc.txt", '../../data/gen_product/pc.txt')
    # fr_product_gen("../../data/raw/product_fr.txt", '../../data/gen_product/bingxiang.txt')
    print('updating')
    update_solr("greeting.txt")
    # update_solr("faq.txt", 'api_call_faq')