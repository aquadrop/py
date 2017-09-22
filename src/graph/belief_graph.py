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
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)

from graph.node import Node


class Graph(Node, object):
    ROOT = 'ROOT'

    def __init__(self, slot, value, fields, node_type, id):
        super(Graph, self).__init__(slot=slot, value=value,
                                    fields=fields, node_type=node_type, id=id)
        # This parameter stores all
        # value is a list of nodes that share the same.
        self.node_header = dict()
        self.id_node = dict()

    def get_node_connected_slots(self, value):
        """
        苹果.slot 可以是品牌,也可以是水果
        """
        nodes = self.node_header[value]
        slots = set()
        for node in nodes:
            slots.add(node.slot)
        return list(slots)

    def get_root_node(self):
        return self.node_header[self.ROOT][0]

    def get_nodes_by_value(self, node_value):
        """
        get_nodes_by_value("苹果")...
        return a list
        """
        return self.node_header[node_value]

    def get_node_by_id(self, id):
        return self.id_node[id]


def load_belief_graph(path, output_model_path):
    belief_graph = None
    node_header = {}
    id_node = {}
    with open(path, 'r') as f:
        for line in f:
            if line.startswith("-"):
                line = line.strip("\n")
                print(line.strip("\n"))
                value, id, slot, fields_, node_type = line.split("#")
                fields = dict()
                value = value.replace("-", "")
                if fields_:
                    for fi in fields_.split(","):
                        field = fi.split(":")[0]
                        prob = float(fi.split(":")[1])
                        fields[field] = prob
                if value == "ROOT":
                    node = Graph(
                        value=value, fields=fields, slot=slot, id=id, node_type=node_type)
                    if not belief_graph:
                        belief_graph = node
                else:
                    node = Node(value=value, fields=fields,
                                slot=slot, id=id, node_type=node_type)
                if value not in node_header:
                    node_header[value] = []
                node_header[value].append(node)
                if id in id_node:
                    raise ValueError("id")
                id_node[id] = node

    belief_graph.node_header = node_header
    belief_graph.id_node = id_node

    with open(path, 'r') as f:
        for line in f:
            if line.startswith("+"):
                line = line.strip("\n")
                print(line.strip("\n"))
                parent_, children_ = line.split('#')
                parent_ = parent_.replace("+", "")
                parent, parent_id = parent_.split("/")
                node = id_node[parent_id]
                if parent != node.value:
                    raise ValueError("id")
                slot = children_.split(":")[0]
                c_ = children_.split(":")[1]
                for c in c_.split(","):
                    value, id = c.split("/")
                    child_node = id_node[id]
                    if value != child_node.value:
                        raise ValueError("id")
                    node.add_node(child_node)
    with open(output_model_path, "wb") as omp:
        pickle.dump(belief_graph, omp)


if __name__ == "__main__":
    load_belief_graph(
        "/home/deep/solr/memory/memory_py/data/graph/belief_graph.txt", "/home/deep/solr/memory/memory_py/model/graph/belief_graph.pkl")