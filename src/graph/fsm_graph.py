import os
import sys
import json

from collections import OrderedDict


class LogicNode(object):
    def __init__(self, state):
        self.state = state
        self.children = list()
        self.values = list()

    def get_values(self):
        return self.values

    def get_state(self):
        return self.state

    def get_child(self):
        return self.children

    def set_values(self, values):
        self.values = values

    def set_children(self, children):
        self.children = children


class LogicEdge(object):
    def __init__(self, head, tail):
        self.head = head
        self.tail = tail
        self.state = self.head + '-' + self.tail
        self.values = list()

    def get_values(self):
        return self.values

    def get_state(self):
        return self.state

    def get_head(self):
        return self.head

    def get_tail(self):
        return self.tail

    def set_values(self, values):
        self.values = values


def build_graph(path):
    with open(path, 'r') as f:
        graph = json.load(f)

    logic_nodes = dict()
    logic_edges = dict()
    states = graph.keys()

    for state in states:
        logic_node = LogicNode(state)
        children = graph[state]
        logic_node.set_children(children)
        logic_nodes[state] = logic_node

        logic_edge = [LogicEdge(state, child) for child in children]
        for e in logic_edge:
            logic_edge_state = e.get_state()
            logic_edges[logic_edge_state] = e

    return graph, logic_nodes, logic_edges


def main():
    prefix = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(prefix, 'data/bookstore/logic_graph.txt')
    graph, logic_nodes, logic_edges = build_graph(path)

    print(graph)
    print(logic_nodes)
    print(logic_edges)


def test():
    test = {
        'root': ['scan'],
        'scan': ['auth', 'scan'],
        'auth': ['auth', 'complete'],
        'complete': []
    }
    prefix = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(prefix, 'data/bookstore/test.txt')
    print(path)
    with open(path, 'w') as f:
        json.dump(test, f)


if __name__ == '__main__':
    main()
