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

from flask import Flask
from flask import request
import json

from functools import lru_cache

# pickle
from graph.belief_graph import Graph
from kernel.main_kernel import MainKernel

import sys

import os
dir_path = os.path.dirname(os.path.realpath(__file__))

app = Flask(__name__)

config = {"belief_graph": dir_path + "/../model/graph/belief_graph.pkl"}

kernel = MainKernel(config)
multi_sn_kernels = lru_cache(1000)

QSIZE = 1
kernel_backups = Queue(1000)


@app.route('/sn/info', methods=['GET', 'POST'])
def info():
    current_q_size = kernel_backups.qsize()
    current_u_size = len(multi_sn_kernels.keys())
    result = {"current_q_size": current_q_size, "current_u_size": current_u_size}
    return json.dumps(result, ensure_ascii=False)


@app.route('/sn/chat', methods=['GET', 'POST'])
def chat():
    try:
        args = request.args
        q = args['q']
        q = urllib.parse.unquote(q)
        try:
            u = args['u']
            if not multi_sn_kernels.has_key(u):
                if kernel_backups.qsize() > 0:
                    ek = kernel_backups.get_nowait()
                    multi_sn_kernels[u] = ek
                else:
                    for i in range(2):
                        k = MainKernel(config)
                        kernel_backups.put_nowait(k)
                        result = {"question": q, "result": \
                            {"answer": "主机负载到达初始上限,正在为您分配实例..."},
                                  "user": u}
                        # print('========================')
                    return json.dumps(result, ensure_ascii=False)
            u_i_kernel = multi_sn_kernels[u]
            r = u_i_kernel.kernel(q=q, user=u)
            result = {"question": q, "result": {"answer": r}, "user": u}
            return json.dumps(result, ensure_ascii=False)

        except:
            traceback.print_exc()
            r = kernel.kernel(q=q)
            result = {"question": q, "result": {"answer": r}, "user": "solr"}
            return json.dumps(result, ensure_ascii=False)
    except Exception:
        traceback.print_exc()
        result = {"question": q, "result": {"answer": "主机负载到达上限或者主机核心出现异常"}, "user": "solr"}
        return json.dumps(result, ensure_ascii=False)

if __name__ == "__main__":
    # SK = SceneKernel()
    # print(SK.kernel('你叫什么名字'))

    parser = argparse.ArgumentParser()
    parser.add_argument('--qsize', choices={'1', '5', '200'},
                        default='5', help='q_size initializes number of the starting instances...')
    args = parser.parse_args()

    QSIZE = int(args.qsize)

    for i in range(QSIZE):
        print('========================')
        k = MainKernel(config)
        kernel_backups.put_nowait(k)
    app.run(host='0.0.0.0', port=21304, threaded=True)