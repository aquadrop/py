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

extra_path = os.path.join(
    grandfatherdir, 'data/memn2n/train/tree/extra/data_test')
train_path = os.path.join(
    grandfatherdir, 'data/memn2n/train/tree/origin/train.txt')
new_train_path = os.path.join(
    grandfatherdir, 'data/memn2n/train/tree/train.txt')
candidate_path = os.path.join(
    grandfatherdir, 'data/memn2n/train/tree/origin/candidates.txt')
new_candidate_path = os.path.join(
    grandfatherdir, 'data/memn2n/train/tree/candidates.txt')


def update2():
    candidates = set()
    with open(candidate_path, 'r') as f:
        for line in f:
            line = line.strip().lower()
            if len(line):
                if not line.startswith('reserved_'):
                    candidates.add(line)

    with open(extra_path, 'r') as f:
        extra_data = f.readlines()
    # extra_data=[a.strip('\n').lower() for a in extra_data]
    print(extra_data)
    extra = []
    tmp = []
    for line in extra_data:
        if line == '\n':
            extra.append(tmp)
            tmp = []
        else:
            line = line.strip().lower()
            tmp.append(line)
    extra.append(tmp)
    print(extra)
    extra_data = []
    for dialog in extra:
        if len(dialog) and len(candidates) < config.CANDIDATE_POOL:
            candids = [line.split('\t')[1] for line in dialog]
            num = sum([c in candidates for c in candids])
            if len(candidates) + num <= config.CANDIDATE_POOL:
                for c in candids:
                    candidates.add(c)
                extra_data.append(dialog)
            else:
                continue
    print(extra_data)
    candidates = list(candidates)
    candidates.sort()
    print('new candidates num', len(candidates))
    len_origin = len(candidates)
    if len_origin < config.CANDIDATE_POOL:
        for i in range(config.CANDIDATE_POOL - len_origin):
            candidates.append('reserved_' + str(i + len_origin))

    with open(new_train_path, 'w') as f:
        with open(train_path, 'r') as tf:
            for line in tf:
                f.writelines(line)
            f.writelines('\n')
        for dialog in extra_data:
            for l in dialog:
                f.write(l + '\n')
            f.write('\n')
    with open(new_candidate_path, 'w') as f:
        for l in candidates:
            f.write(l + '\n')


def update():
    reserved_idx = 0
    candidate = set()
    with open(extra_path, 'r') as f:
        extra_data = f.readlines()
        extra_data = [a.strip('\n').lower() for a in extra_data]
    # with open(train_path, 'r') as f:
    #     train_data = f.readlines()
    #     train_data = [a.lower() for a in train_data]
    with open(candidate_path, 'r') as f:
        for line in f:
            line = line.strip('\n').lower()
            if len(line):
                if not line.startswith('reserved_'):
                    candidate.add(line)

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

    # while train_data[-1] == '\n':
    #     train_data.pop()
    # train_data.append('\n')
    # train_data.extend(extra_data)

    with open(new_train_path, 'w') as f:
        with open(train_path, 'r') as tf:
            for line in tf:
                f.writelines(line)
            f.writelines('\n')
        for l in extra_data:
            f.write(l + '\n')
    with open(new_candidate_path, 'w') as f:
        for l in candidate:
            f.write(l + '\n')


def main():
    update2()


if __name__ == '__main__':
    main()
