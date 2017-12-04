import sys
import os
import pickle

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)
grandfatherdir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
from graph.belief_graph import Graph
import utils.query_util as query_util



def _pickle():
    graph_dir = os.path.join(grandfatherdir, "model/graph/belief_graph.pkl")
    with open(graph_dir, "rb") as input_file:
        belief_graph = pickle.load(input_file)
    slots_trans = belief_graph.slots_trans
    slots_trans['entity'] = '实体'
    slots_trans['search_'] = '搜索'
    slots_trans['request_'] = '确认'
    slots_trans['rhetorical_'] = '反问'
    slots_trans['placeholder'] = '占位'
    slots_trans['ambiguity_removal'] = '消除歧义'
    slots_trans['slot_'] = ''
    slots_trans['virtual_'] = '虚'
    slots_trans['api_call_'] = ''
    slots_trans['sunning'] = '苏宁'
    slots_trans['plugin'] = ''
    slots_trans['act'] = ''
    slots_trans['discount'] = '打折'
    slots_trans['query'] = '查询'
    slots_trans['float'] = ''
    slots_trans['deny'] = '拒绝'
    slots_trans['all'] = ''
    slots_trans['location'] = '地点'
    slots_trans['whatever'] = '随便'
    slots_trans['general'] = '通用'


    translator_graph_dir=os.path.join(grandfatherdir, "model/graph/translator_graph.pkl")
    with open(translator_graph_dir,'wb') as f:
        pickle.dump(slots_trans,f)


class Translator():
    def __init__(self,path=os.path.join(grandfatherdir, "model/graph/translator_graph.pkl")):
        self.dic=self._load(path)

    def _load(self, path):
        with open(path,'rb') as f:
            return pickle.load(f)

    def en2cn(self, query):
        _q = query
        for k,v in self.dic.items():
            query=query.replace(k,v)
        # print('translated..{}->{}'.format(_q, query))
        return query

    def cn2en(self, query):
        for k,v in self.dic.items():
            query=query.replace(v,k)
        return query


def test():
    with open(os.path.join(grandfatherdir,'data/memn2n/train/tree/train.txt'),'r', encoding='utf-8') as f:
        candidates=f.readlines()
    translator=Translator(os.path.join(grandfatherdir, "model/graph/translator_graph.pkl"))
    for line in candidates:
        line = line.strip('\n')
        line = translator.en2cn(line)
        line = query_util.tokenize(line, char=0)
        print(line)


if __name__ == '__main__':
    l=[
       "api_call_slot_reg.repeat_auth:点击验证成功"]
    _pickle()
    tr = Translator()
    for ll in l:
        print(tr.en2cn(ll))
    # test()