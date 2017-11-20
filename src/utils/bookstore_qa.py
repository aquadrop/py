import os
import sys
from collections import OrderedDict

prefix = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

template_path = os.path.join(prefix, 'data/gen_product/bookstore_qa.txt')
outpath = 'test.qa.txt'


def parse(lines):
    keys = lines[0].split(':')[1].split('|')
    lines = lines[1:]
    middle = OrderedDict()
    prefix = OrderedDict()
    postfix = OrderedDict()
    for line in lines:
        name, entities = line.split(':')
        entities = entities.strip().split('|')
        if name.startswith('prefix_'):
            prefix[name] = entities
        elif name.startswith('postfix_'):
            postfix[name] = entities
        else:
            middle[name] = entities

    return middle, prefix, postfix


def gen(outpath=outpath, path=template_path):
    with open(path, 'r') as f:
        lines = f.readlines()[1:]
    middle, prefix, postfix = parse(lines)

    res = list()
    for key in middle.keys():
        middles = middle[key]
        prefixs = prefix['prefix_' + key]
        postfixs = postfix['postfix_' + key]

        for m in middles:
            for pre in prefixs:
                query = pre + m
                label = 'api_call_qa_location_' + key + ':' + m
                line = query.strip() + '\t' + label.strip() + '\t' + 'placeholder'
                res.append(line)
            for post in postfixs:
                query = m + post
                label = 'api_call_qa_location_' + key + ':' + m
                line = query.strip() + '\t' + label.strip() + '\t' + 'placeholder'
                res.append(line)
    with open(outpath, 'w') as f:
        for line in res:
            f.write(line + '\n\n')


def main():
    gen()


if __name__ == '__main__':
    main()
