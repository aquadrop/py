import pickle
import os
import sys

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
grandfatherdir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parentdir)
sys.path.append(grandfatherdir)

from graph.belief_graph import *

def read_node_header(output_file):
    with open(output_file, 'rb') as f:
        info = pickle.load(f)
    product = info.node_header.keys()
    ofile = open("../../data/dict/ext2.dic", 'w')
    for key in product:
        #print(key)
        ofile.write(key + "\n")
    ofile.close()

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
    read_node_header("../../model/graph/belief_graph.pkl")