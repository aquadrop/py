import argparse
import pickle as pkl
import gensim
import sys
import os
import time
import traceback
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
grandfatherdir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, parentdir)
import memory.config as config

extra_path = os.path.join(grandfatherdir, 'data/memn2n/train/tree/extra/data')
train_path = os.path.join(grandfatherdir, 'data/memn2n/train/tree/train.txt')
candidate_path = os.path.join(
    grandfatherdir, 'data/memn2n/train/tree/candidates.txt')


def update():
    reserved_idx = 0
    candidate = set()
    with open(extra_path, 'r') as f:
        extra_data = f.readlines()
        extra_data = [a.lower() for a in extra_data]
    with open(train_path, 'r') as f:
        train_data = f.readlines()
        train_data = [a.lower() for a in train_data]
    with open(candidate_path, 'r') as f:
        for line in f:
            line = line.strip('\n').lower()
            if len(line):
                candidate.add(line)
                if line.find('reserved_') == -1:
                    reserved_idx += 1

    for line in extra_data:
        line = line.strip()
        if len(line):
            candid = line.split('\t')[1].lower()
            # if candid not in candidate:
            #     candidate.append(candid)
            candidate.add(candid)

    candidate = list(candidate)
    candidate.sort()
    print('new candidates num', len(candidate))
    len_origin = len(candidate)
    if len_origin < config.CANDIDATE_POOL:
        for i in range(config.CANDIDATE_POOL - len_origin):
            candidate.append('reserved_' + str(i + len_origin))

    while train_data[-1] == '\n':
        train_data.pop()
    train_data.append('\n')
    train_data.extend(extra_data)

    with open(train_path, 'w') as f:
        for l in train_data:
            f.write(l)
    with open(candidate_path, 'w') as f:
        for l in candidate:
            f.write(l + '\n')


def main():
    update()


if __name__ == '__main__':
    main()
