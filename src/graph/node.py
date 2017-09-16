class Node:
    def __init__(self, slot, value, fields):
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
        self.children = dict()  # value to chidren nodes

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
        node.parent_node = self

        # supply other information
        if node_field not in self.slot_values_mapper:
            self.slot_values_mapper[node_field] = []
        self.slot_values_mapper[node_field].append(node_value)

    def siblings(self, keep_slot):
        """
        Args:
            keep_slot: True or False, return only siblings with same slot or not
        """
        if not self.parent_node:
            return []

        siblings = []
        if keep_slot == True:
            children = self.parent_node.chidren
            for key, value in children.iteritems():
                pass

    def sibling_names(self, value, keep_slot):
        pass

    def get_parent_node(self):
        return self.parent_node

    def get_children_names(self, max_num, required_only):
        pass
