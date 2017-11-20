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

Belief Tracker
"""

import numpy as np
import uuid


class Node:

    API_NODE = "category"
    NORMAL_NODE = "normal"
    PROPERTY_NODE = "property"

    RANGE = "range"
    KEY = "key"

    def __init__(self, slot, value, fields, node_type, id):
        '''
        For now only api_node has more than one field
        Args:
            slot: slot serves as edge
            fields: a map indicating fields that are required or not
            node_type: api_node, property_node, normal
        '''
        self.value = value  # value aka slot-value filling
        self.slot = slot  # slot aks slot value filling
        self.parent_node = None
        self.fields = fields  # is a dict, valued as prob
        self.field_type = dict() # RANGE, KEY,...
        self.children = dict()  # value to chidren nodes
        self.node_type = node_type
        self.slot_to_values_mapper = dict()
        self.value_to_slot_mapper = dict()
        self.fields_trans = dict()
        self.id = id  # globally uniq
        self.level = 0

    def get_node_by_value(self, value):
        return self.children[value]

    def set_node_slot_trans(self, field, trans):
        if field not in self.fields:
            raise ValueError(
                'translation field error: node must have field which is included in self.fields.'
                ' Fix the graph file before continue;', field, self.fields)
        self.fields_trans[field] = trans

    def get_node_slot_trans(self, field):
        if field in self.fields_trans:
            return self.fields_trans[field]
        return None

    def set_field_type(self, field, field_type):
        self.field_type[field] = field_type

    def get_field_type(self, field):
        if field not in self.field_type:
            return Node.KEY
        return self.field_type[field]

    def has_field(self, field):
        return field in self.fields

    def add_node(self, node):
        """
        Args:
            node: child_node to append to the current node, there will be a lot other information brought in
        Raises:
            ValueError: child_node slot must be in self.fields.
        """
        node_value = node.value
        node_field = node.slot
        if node_field not in self.fields:
            raise ValueError(
                'error: node must have field which is included in self.fields.'
                ' Fix the graph file before continue;', node_field, self.fields)
        self.children[node_value] = node
        node.level = self.level + 1
        node.parent_node = self

        # supply other information
        if node_field not in self.slot_to_values_mapper:
            self.slot_to_values_mapper[node_field] = []
        self.slot_to_values_mapper[node_field].append(node_value)

        if node_value not in self.value_to_slot_mapper:
            self.value_to_slot_mapper[node_value] = node_field

    def has_ancestor_by_value(self, value):
        node = self.parent_node
        while node != None:
            if node.value == value:
                return True
            node = node.parent_node
        return False

    def is_api_node(self):
        return self.node_type == self.API_NODE

    def is_property_node(self):
        return self.node_type == self.PROPERTY_NODE

    def get_slot_by_value(self, value):
        if value not in self.value_to_slot_mapper:
            return None
        return self.value_to_slot_mapper[value]

    def gen_required_slot_fields(self):
        required_fields = []
        for field, prob in self.fields.items():
            rnd = np.random.uniform(0, 1)
            if rnd < prob:
                required_fields.append(field)
        return required_fields

    def has_child(self, value):
        """
        has direct child
        """
        return value in self.children

    def get_child(self, value):
        return self.children[value]

    def has_posterity(self, value, belief_graph):
        """
        We use belief_graph to achieve faster bottom to up search
        """
        candidate_nodes = belief_graph.get_nodes_by_value(value)
        if self.value == "ROOT".lower():
            return len(candidate_nodes) > 0

        for node in candidate_nodes:
            id = node.id
            a = node
            while a.value != "ROOT".lower():
                if a.value == self.value:
                    return True
                a = a.parent_node
        return False

    def get_posterity_nodes_by_value(self, value, belief_graph):
        candidate_nodes = belief_graph.get_nodes_by_value(value)
        if self.value == "ROOT".lower():
            return candidate_nodes

        posterity = []
        for node in candidate_nodes:
            id = node.id
            a = node
            while a.value != "ROOT".lower():
                if a.value == self.value:
                    posterity.append(belief_graph.get_node_by_id(id))
                    break
                a = a.parent_node
        return posterity

    def get_siblings(self, keep_slot):
        """
        This function might be dangerous to be used
        Args:
            keep_slot: True or False, return only siblings with same slot or not
        """
        if not self.parent_node:
            return []

        siblings = []
        children = self.parent_node.chidren
        for key, value in children.items():
            if key != self.value:
                sibling = value
                if sibling.slot == self.slot or keep_slot == False:
                    siblings.append(sibling)
        return siblings

    def get_ancestry_values(self):
        node = self.parent_node
        anc_values = []
        while node.value != "ROOT".lower():
            anc_values.append(node.value)
            node = node.parent_node
        return anc_values

    def get_sibling_names(self, value, keep_slot):
        siblings = self.get_siblings(keep_slot)
        sibling_names = []
        for node in siblings:
            sibling_names.append(node.value)
        return sibling_names

    def get_parent_node(self):
        return self.parent_node

    def get_children_names_by_slot(self, slot):
        children_names = []
        for key, value in self.children.items():
            if self.value_to_slot_mapper[key] == slot:
                children_names.append(key)
        return children_names

    def get_children_names(self, max_num, required_only):
        children_names = []
        for key, value in self.children.items():
            if self.fields[key] == 1 or required_only == False:
                children_names.append(key)
        children_names = np.array(children_names)
        children_names = np.random.choice(children_names, max_num)
        return children_names

    def is_leaf(self):
        return len(self.children) == 0