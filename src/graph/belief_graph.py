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
import uuid
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)

from graph.node import Node


class Graph(Node, object):
    ROOT = "ROOT".lower()

    def __init__(self, slot, value, fields, node_type, id):
        super(Graph, self).__init__(slot=slot, value=value,
                                    fields=fields, node_type=node_type, id=id)
        # This parameter stores all
        # value is a list of nodes that share the same.
        self.node_header = dict()
        self.id_node = dict()
        """
        key as slot, value as type
        """
        self.slots = dict()
        self.slots_trans = dict()
        self.range_adapter_mapper = dict()
        self._prebuild_range_adaper()
        """
        price:price
        tv.size, phone.size, pc.size: __inch__
        tv.distance: __meter__
        ac.power: ac.power
        """

    def _prebuild_range_adaper(self):
        self.range_adapter_mapper['price'] = 'price'
        self.range_adapter_mapper['tv.size'] = '__inch__'
        self.range_adapter_mapper['phone.size'] = '__inch__'
        self.range_adapter_mapper['pc.size'] = '__inch__'
        self.range_adapter_mapper['tv.distance'] = '__meter__'
        self.range_adapter_mapper['ac.power_float'] = 'ac.power'
        self.range_adapter_mapper['fr.height'] = 'height'
        self.range_adapter_mapper['fr.width'] = 'width'

    def range_adapter(self, key):
        return self.range_adapter_mapper[key]

    def is_entity_value(self, value):
        if len(self.node_header[value]) == 1:
            return False
        nodes = self.node_header[value]
        slots = [node.slot for node in nodes]
        if len(set(slots)) == 1:
            return False
        else:
            return True

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

    def get_field_type(self, field):
        if field not in self.slots:
            return None
        return self.slots[field]

    def get_nodes_by_slot(self, slot):
        field_nodes = []
        for key, nodes in self.node_header.items():
            for node in nodes:
                if node.slot == slot:
                    field_nodes.append(node)
        return field_nodes

    def get_nodes_by_value(self, node_value):
        """
        get_nodes_by_value("苹果")...
        return a listraw
        """
        if node_value not in self.node_header:
            return []
        return self.node_header[node_value]

    def get_nodes_by_value_and_field(self, value, field):
        nodes = self.get_nodes_by_value(value)
        filtered = []
        for node in nodes:
            if node.slot == field:
                filtered.append(node)
        return filtered

    def has_node_by_value(self, node_value):
        return node_value in self.node_header

    def get_node_by_id(self, id):
        return self.id_node[id]

    def has_slot(self, slot):
        return slot in self.slots


def load_belief_graph(path, output_model_path):
    belief_graph = None
    node_header = {}
    id_node = {}
    slots = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith("-"):
                line = line.strip("\n").replace(" ", "").replace("\t", "")
                print(line.strip("\n"))
                value, id, slot, fields_, node_type = line.split("#")
                fields = dict()
                value = value.replace("-", "")

                if fields_:
                    for fi in fields_.split(","):
                        field = fi.split(":")[0]
                        prob = float(fi.split(":")[1])
                        fields[field] = prob
                if value == "ROOT".lower():
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
                line = line.strip("\n").replace(" ", "").replace("\t", "")
                print(line.strip("\n"))
                parent_, slot, value_type, children_ = line.split('#')
                parent_ = parent_.replace("+", "")
                parent, parent_id = parent_.split("/")
                node = id_node[parent_id]
                if parent != node.value:
                    raise ValueError("id")

                for c in children_.split(","):
                    splitted = c.split("/")
                    if value_type is not "KEY":
                        node.set_field_type(slot, value_type)

                    if len(splitted) == 2:
                        value = splitted[0]
                        id = splitted[1]
                        child_node = id_node[id]
                        if value != child_node.value:
                            raise ValueError("id")
                        node.add_node(child_node)
                    else:
                        value = splitted[0]
                        # property node
                        id = str(uuid.uuid4())
                        child_node = Node(value=value, fields=dict(),
                                    slot=slot, id=id, node_type="property")
                        if value not in node_header:
                            node_header[value] = []
                        node_header[value].append(child_node)
                        node.add_node(child_node)
    with open(output_model_path, "wb") as omp:
        pickle.dump(belief_graph, omp)


def load_belief_graph_from_tables(files, output_file):
    belief_graph = None
    node_header = {}
    id_node = {}
    slots = dict()
    # stage 1, build node
    for f in files:
        with open(f, 'r', encoding='utf-8') as inpt:
            for line in inpt:
                line = line.strip('\n').replace(' ', '').lower()
                note, cn, slot, node_type, slot_value = line.split('|')
                if note == '-':
                    slots[slot] = node_type
                    _id = str(uuid.uuid4())
                    # node = Node(value=slot_value, fields=dict(),
                    #                   slot=slot, id=_id, node_type="property")
                    if slot_value == "ROOT".lower():
                        node = Graph(
                            value=slot_value, fields=dict(), slot=slot, id=id, node_type=slot)
                        if not belief_graph:
                            belief_graph = node
                    else:
                        node = Node(value=slot_value, fields=dict(),
                                    slot=slot, id=id, node_type=slot)
                    if slot_value not in node_header:
                        node_header[slot_value] = []
                    node_header[slot_value].append(node)
                    id_node[_id] = node
                if note == '*':
                    tokens = slot_value.split(",")
                    for t in tokens:
                        a, b = t.split(":")
                        node.fields[a] = float(b)
                    # node.fields = dict()
    belief_graph.id_node = id_node
    belief_graph.node_header = node_header
    belief_graph.slots = slots
    for f in files:
        with open(f, 'r', encoding='utf-8') as inpt:
            for line in inpt:
                line = line.strip('\n').replace(' ', '').lower()
                note, cn, slot, node_type, slot_value = line.split('|')
                if note == '-':
                    nodes = node_header[slot_value]
                    # print(slot_value, len(nodes))
                    if len(nodes) > 1 or len(nodes) == 0:
                        raise ValueError('non property node value should be unique')
                    else:
                        node = nodes[0]
                if note == '+':
                    slots[slot] = node_type
                    note, cn, slot, value_type, slot_value = line.split('|')
                    node.set_node_slot_trans(slot, cn)
                    belief_graph.slots_trans[slot] = cn
                    if value_type != Node.KEY:
                        # print(slot)
                        node.set_field_type(slot, value_type)
                        continue
                    names = slot_value.split(',')
                    for name in names:
                        if 'category' in slot:
                            nodes = node_header[name]
                            if len(nodes) > 1 or len(nodes) == 0:
                                raise ValueError('non property node value should be unique')
                            else:
                                child_node = nodes[0]
                            node.add_node(child_node)
                            continue
                        _id = str(uuid.uuid4())
                        child_node = Node(value=name, fields=dict(),
                                    slot=slot, id=_id, node_type="property")
                        node.add_node(child_node)
                        if name not in node_header:
                            node_header[name] = []
                        node_header[name].append(child_node)
                        id_node[id] = child_node

    with open(output_file, "wb") as omp:
        pickle.dump(belief_graph, omp)


if __name__ == "__main__":
    # load_belief_graph(
    #     "/home/deep/solr/memory/memory_py/data/graph/belief_graph.txt",
    #     "/home/deep/solr/memory/memory_py/model/graph/belief_graph.pkl")
    table_files = ['../../data/gen_product/bingxiang.txt',
                   '../../data/gen_product/dianshi.txt',
                   '../../data/gen_product/digitals.txt',
                   '../../data/gen_product/homewares.txt',
                   '../../data/gen_product/kongtiao.txt',
                   '../../data/gen_product/root.txt',
                   '../../data/gen_product/shouji.txt',
                   '../../data/gen_product/pc.txt',
                   '../../data/gen_product/grocery.txt',
                   '../../data/gen_product/fruits.txt']
    output_file = "../../model/graph/belief_graph.pkl"
    load_belief_graph_from_tables(table_files, output_file)
