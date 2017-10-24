import os
import sys

parentdir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
# print(parentdir)

extra_path = os.path.join(parentdir, 'data/memn2n/train/tree/extra/data')
train_path = os.path.join(parentdir, 'data/memn2n/train/tree/train.txt')
candidate_path = os.path.join(
    parentdir, 'data/memn2n/train/tree/candidates.txt')


def update():
    reserved_idx = 0
    candidate = []
    with open(extra_path, 'r') as f:
        extra_data = f.readlines()
    with open(train_path, 'r') as f:
        train_data = f.readlines()
    with open(candidate_path, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line):
                candidate.append(line)
                if line.find('reserved_') == -1:
                    reserved_idx += 1

    for line in extra_data:
        line = line.strip()
        if len(line):
            candid = line.split('\t')[1]
            # if candid not in candidate:
            #     candidate.append(candid)
            if candid not in candidate:
                print(candid)
                candidate[reserved_idx] = candid
                reserved_idx += 1

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
