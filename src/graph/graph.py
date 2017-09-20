import node


class Graph(node.Node, object):
    ROOT = 'ROOT'

    def __init__(self):
        super(Graph, self).__init__(slot="ROOT")
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

    def get_node_values_by_slot(self, slot_name):
        """
        用于全局列举品牌名
        """
        pass

    def get_nodes_by_value(self, node_value):
        """
        get_nodes_by_value("苹果")...
        return a list
        """
        return self.node_header[node_value]

    def get_node_by_id(self, id):
        return self.id_node[id]
