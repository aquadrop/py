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

import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
grandfatherdir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parentdir)
sys.path.append(grandfatherdir)

# for pickle
from graph.belief_graph import Graph
from kernel.belief_tracker import BeliefTracker


class MainKernel:

    def __init__(self, config):
        self.config = config
        self.belief_tracker = BeliefTracker(config['belief_graph'])

    def kernel(self, q, user='solr'):
        response = self.belief_tracker.kernel(q)
        return response

if __name__ == '__main__':
    config = {"belief_graph": "../../model/graph/belief_graph.pkl"}
    kernel = MainKernel(config)
    while (True):
        ipt = input("input:")
        resp = kernel.kernel(ipt)
        print(resp)