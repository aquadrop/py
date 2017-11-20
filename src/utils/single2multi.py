import os
import sys

parentdir = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))
grandfatherdir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, parentdir)
sys.path.insert(0, grandfatherdir)

from memory import config
from collections import OrderedDict

DATA_DIR = os.path.join(grandfatherdir, 'data/memn2n/train/tree/origin')
TARGET_DATA_DIR = os.path.join(grandfatherdir, 'data/memn2n/train/multi_tree')
CANDIDATE_POOL = 1000


def single2multi_core(lines, candidates):
    apis = OrderedDict()
    apis['api_call_query_'] = 'api_call_query,'
    apis['api_call_slot_'] = 'api_call_slot,'
    apis['api_call_query,discount'] = 'api_call_query_discount'
    apis['api_call_query,general'] = 'api_call_query_general'
    apis['api_call_slot,whatever'] = 'api_call_slot_whatever'

    skip_apis = ['api_call_slot_whatever',
                 'api_call_query_general', 'api_call_query_discount']
    new_lines = []
    for line in lines:
        if line != '\n':
            # print(line)
            [query, label, salt] = line.split('\t')
            for k, v in apis.items():
                label = label.replace(k, v)
            for l in label.split(','):
                candidates.add(l)
            line = '\t'.join([query, label, salt])
        new_lines.append(line)

    return new_lines


def single2multi():
    data_files = ['train.txt', 'val.txt', 'test.txt']
    candidate_file = 'candidates.txt'
    candidate_file = os.path.join(TARGET_DATA_DIR, candidate_file)

    candidates = set()
    for data_set in data_files:
        with open(os.path.join(DATA_DIR, data_set), 'r') as f:
            lines = f.readlines()
        lines = single2multi_core(lines, candidates)
        with open(os.path.join(TARGET_DATA_DIR, data_set), 'w') as f:
            f.writelines(lines)
    candidates = list(candidates)
    candidates.sort()
    candidates = [candid + '\n' for candid in candidates]
    for i in range(CANDIDATE_POOL - len(candidates)):
        candidates.append('reserved_' + str(i + 1) + '\n')
    with open(candidate_file, 'w') as f:
        f.writelines(candidates)


def main():
    single2multi()


if __name__ == '__main__':
    main()
