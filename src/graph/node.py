import numpy as np
import uuid


class Node:
    def __init__(self, slot, value, fields, id=None):
        '''
        Args:
            slot: slot serves as edge
            fields: a map indicating fields that are required or not
        '''
        self.value = value  # value aka slot-value filling
        self.slot = slot  # slot aks slot value filling
        self.fields = fields
        self.parent_node = None
        self.slot_values_mapper = dict()
        self.value_slot_mapper = dict()
        self.children = dict()  # value to chidren nodes
        if not id:
            self.id = str(uuid.uuid4)  # globally uniq
        self.level = 0

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
                'error: node must have field which is included in self.fields. Fix the graph file before continue')
        self.children[node_value] = node
        node.level = self.level + 1
        node.parent_node = self

        # supply other information
        if node_field not in self.slot_values_mapper:
            self.slot_values_mapper[node_field] = []
        self.slot_values_mapper[node_field].append(node_value)

    def has_ancestor_by_value(self, value):
        node = self.parent_node
        while node != None:
            if node.value == value:
                return True
            node = node.parent_node
        return False

    def is_api_call_node(self):
        return False

    def get_num_required_slots(self):
        return 0

    def get_slot_by_value(self, value):
        if value not in self.value_slot_mapper:
            return None
        return self.value_slot_mapper[value]

    def get_required_slot_fields(self):
        return dict()

    def get_optional_slot_fields(self):
        return dict()

    def has_child(self, value):
        """
        has direct child
        """
        return value in self.children

    def get_child(self, value):
        return self.children[value]

    def has_posterity(self, value):
        return False

    def get_posterity_nodes_by_value(self, value):
        return None

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
        for key, value in children.iteritems():
            if key != self.value:
                sibling = value
                if sibling.slot == self.slot or keep_slot == False:
                    siblings.append(sibling)
        return siblings

    def get_sibling_names(self, value, keep_slot):
        siblings = self.get_siblings(keep_slot)
        sibling_names = []
        for node in siblings:
            sibling_names.append(node.value)
        return sibling_names

    def get_parent_node(self):
        return self.parent_node

    def get_children_names(self, max_num, required_only):
        children_names = []
        for key, value in self.children.iteritems()
            if self.fields[key] == True or required_only == False:
                children_names.append(key)
        children_names = np.array(children_names)
        children_names = np.random.choice(children_names, max_num)
        return children_names
