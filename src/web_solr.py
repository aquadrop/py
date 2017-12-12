# -*- coding:utf-8 -*-

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

main.py
"""

from queue import Queue
import argparse
import traceback
import urllib
import time
import logging
from datetime import datetime

from flask import Flask
from flask import request
import json
from tqdm import tqdm
# from lru import LRU

import sys,os
dir_path = os.path.dirname(os.path.abspath(__file__))
parentdir = os.path.dirname(dir_path)
sys.path.insert(0, '{}/qa'.format(dir_path))
from qa.iqa import Qa
current_date = time.strftime("%Y.%m.%d")
logging.basicConfig(filename=os.path.join(parentdir, 'logs/log_corpus_error_' + current_date + '.log')
                    ,format='%(asctime)s %(message)s', datefmt='%Y.%m.%dT%H:%M:%S', level=logging.INFO)

app = Flask(__name__)
qa = Qa('zx_weixin_qa',solr_addr = 'http://10.89.100.14:8999/solr')


@app.route('/e/info', methods=['GET', 'POST'])
def info():
    result = {"question": "request info", "result": {"answer": 10000}, "user": "solr"}
    return json.dumps(result, ensure_ascii=False)



@app.route('/e/chat', methods=['GET', 'POST'])
def chat():
    try:
        args = request.args
        q = args['q']
        print ('q = ', q)
        u = 'solr'
        best_query, best_answer, best_score =qa.get_responses(query=q, user=u)
        # result['sentence'] = q
        # result['score'] = float(prob[0][0])
        # result['class'] = api + '->' + response  # + '/' + 'avail_vals#{}'.format(str(self.belief_tracker.avails))
        result = {"answer": best_answer, "media": "null", 'from': "memory", "sim": 0, 'score':best_score,'class':'null',
                  'sentence': q}
        print ('best answer', best_answer)
        q = urllib.parse.unquote(q)

        result = {"question": q, "sentence": q, "result": result, "user": u}
        return json.dumps(result, ensure_ascii=False) #dict转化成str

    except Exception:
        logging.error("C@user:{}##error_details:{}".format(u, traceback.format_exc()))
        traceback.print_exc()
        result = {"question": q, "result": {"answer": "kernel exception"}, "user": "solr"}
        return json.dumps(result, ensure_ascii=False)


if __name__ == "__main__":
    # SK = SceneKernel()
    # print(SK.kernel('你叫什么名字'))

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--qsize', choices={'1', '20', '200'},
    #                     default='200', help='q_size initializes number of the starting instances...')
    # args = parser.parse_args()
    #
    # QSIZE = int(args.qsize)
    #
    # for i in tqdm(range(QSIZE)):
    #     k = MainKernel(config)
    #     kernel_backups.put_nowait(k)
    print('web started...')
    app.run(host='0.0.0.0', port=21303, threaded=True)