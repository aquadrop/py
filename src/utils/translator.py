import sys
import os
import pickle

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)
grandfatherdir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
from graph.belief_graph import Graph
import utils.query_util as query_util


class Translator:
    def __init__(self):
        self._pickle()

    def _pickle(self):
        graph_dir = os.path.join(grandfatherdir, "model/graph/belief_graph.pkl")
        with open(graph_dir, "rb") as input_file:
            belief_graph = pickle.load(input_file)
        self.dic = belief_graph.slots_trans
        self.dic['entity'] = '实体'
        self.dic['search_'] = '搜索'
        self.dic['request_'] = '确认'
        self.dic['rhetorical_'] = '反问'
        self.dic['placeholder'] = '占位'
        self.dic['ambiguity_removal'] = '消除歧义'
        self.dic['slot_'] = '买'
        self.dic['virtual_'] = '虚'
        self.dic['api_call_'] = ''
        self.dic['query'] = '查询'
        self.dic['location'] = '地点'

    def _load(self,path):
        with open(path,'rb') as f:
            return pickle.load(f)

    def en2cn(self,query):
        for k,v in self.dic.items():
            query=query.replace(k,v)
        return query

    def cn2en(self,query):
        for k,v in self.dic.items():
            query=query.replace(v,k)
        return query

def test():
    with open(os.path.join(grandfatherdir,'data/memn2n/train/tree/train.txt'),'r', encoding='utf-8') as f:
        candidates=f.readlines()
    translator=Translator()
    for line in candidates:
        line = line.strip('\n')
        line = translator.en2cn(line)
        line = query_util.tokenize(line, char=0)
        print(line)



if __name__ == '__main__':
    test()
