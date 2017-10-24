import sys

import os as os
import json
import numpy as np
from functools import reduce

parentdir = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))
sys.path.insert(0, parentdir)

grandfatherdir = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import data_utils

# temporary paths
DATA_DIR = 'data/memn2n/train/tree'
CANDID_PATH = 'data/memn2n/train/tree/candidates.txt'

MULTI_DATA_DIR = 'data/memn2n/train/tree/multi_tree'
MULTI_CANDID_PATH = 'data/memn2n/train/tree/multi_tree/candidates.txt'

VOCAB_PATH = 'src/memory/dmn/data/vocab/vocab.txt'

# can be sentence or word
input_mask_mode = "sentence"


def get_candidates_word_dict():
    with open(os.path.join(grandfatherdir, VOCAB_PATH), 'r') as f:
        vocab = json.load(f)
    w2idx = dict((c, i + 1) for i, c in enumerate(vocab))

    candidates, candid2idx, idx2candid = data_utils.load_candidates(
        candidates_f=os.path.join(grandfatherdir, CANDID_PATH))

    return idx2candid, w2idx


def load_raw_data():
    candidates, candid2idx, idx2candid = data_utils.load_candidates(
        candidates_f=os.path.join(grandfatherdir, CANDID_PATH))

    train_data, test_data, val_data = data_utils.load_dialog(
        data_dir=os.path.join(grandfatherdir, DATA_DIR),
        candid_dic=candid2idx, dmn=True)
    # print(len(train_data))
    # print(os.path.join(grandfatherdir, DATA_DIR))
    return train_data, test_data, val_data, candidates, candid2idx, idx2candid


def process_data(data_raw, floatX, w2idx, split_sentences=True):
    questions = []
    inputs = []
    answers = []
    input_masks = []
    relevant_labels = []

    for data in data_raw:
        inp, question, answer = data
        # print(inp)
        if len(inp) == 0:
            inp = [['此', '乃', '空', '文']]
        if split_sentences:
            inp_vector = [[w2idx[w] for w in s] for s in inp]
            inputs.append(inp_vector)
        else:
            inp = reduce(lambda x, y: x + ['.'] + y, inp)
            inp_vector = [w2idx[w] for w in inp]
            inputs.append(np.vstack(inp_vector).astype(floatX))

        question = question if len(question) else ['空']
        q_vector = [w2idx[w] for w in question]
        # print(q_vector)
        questions.append(np.vstack(q_vector).astype(floatX))

        answers.append(answer)

        relevant_labels.append([0])

        if not split_sentences:
            if input_mask_mode == 'word':
                input_masks.append(
                    np.array([index for index, w in enumerate(inp)], dtype=np.int32))
            elif input_mask_mode == 'sentence':
                input_masks.append(
                    np.array([index for index, w in enumerate(inp) if w == '.'], dtype=np.int32))
            else:
                raise Exception("invalid input_mask_mode")

    return inputs, questions, answers, input_masks, relevant_labels


def get_sentence_lens(inputs):
    lens = np.zeros((len(inputs)), dtype=int)
    sen_lens = []
    max_sen_lens = []
    for i, t in enumerate(inputs):
        sentence_lens = np.zeros((len(t)), dtype=int)
        for j, s in enumerate(t):
            sentence_lens[j] = len(s)
        lens[i] = len(t)

        sen_lens.append(sentence_lens)
        max_sen_lens.append(np.max(sentence_lens))

    return lens, sen_lens, max(max_sen_lens)


def get_lens(inputs, split_sentences=False):
    lens = np.zeros((len(inputs)), dtype=int)
    for i, t in enumerate(inputs):
        lens[i] = t.shape[0]
    return lens


def pad_inputs(inputs, lens, max_len, mode="", sen_lens=None, max_sen_len=None):
    if mode == "mask":
        padded = [np.pad(inp, (0, max_len - lens[i]), 'constant',
                         constant_values=0) for i, inp in enumerate(inputs)]
        return np.vstack(padded)

    elif mode == "split_sentences":
        padded = np.zeros((len(inputs), max_len, max_sen_len))
        for i, inp in enumerate(inputs):
            padded_sentences = \
                [np.pad(s, (0, max_sen_len - sen_lens[i][j]),
                        'constant', constant_values=0) for j, s in enumerate(inp)]
            # trim array according to max allowed inputs
            if len(padded_sentences) > max_len:
                padded_sentences = padded_sentences[(
                    len(padded_sentences) - max_len):]
                lens[i] = max_len
            padded_sentences = np.vstack(padded_sentences)
            padded_sentences = np.pad(padded_sentences, ((
                0, max_len - lens[i]), (0, 0)), 'constant', constant_values=0)
            padded[i] = padded_sentences
        return padded

    padded = [np.pad(np.squeeze(inp, axis=1), (0, max_len - lens[i]),
                     'constant', constant_values=0) for i, inp in enumerate(inputs)]
    return np.vstack(padded)


def load_data(config, split_sentences=True):

    # fix vocab
    with open(os.path.join(grandfatherdir, VOCAB_PATH), 'r') as f:
        vocab = json.load(f)
    w2idx = dict((c, i + 1) for i, c in enumerate(vocab))
    idx2w = dict((i + 1, c) for i, c in enumerate(vocab))

    word_embedding = np.random.uniform(
        -config.embedding_init,
        config.embedding_init,
        (len(vocab), config.embed_size))

    train_data, val_data, test_data, candidates, candid2idx, idx2candid = load_raw_data()

    train_data = process_data(train_data, config.floatX, w2idx)
    val_data = process_data(val_data, config.floatX, w2idx)
    test_data = process_data(test_data, config.floatX, w2idx)

    # print(len(train_data[0]))

    for t, v in zip(train_data, val_data):
        t += v

    inputs, questions, answers, input_masks, rel_labels = train_data if config.train_mode else test_data

    # print(len(train_data[0]))

    if split_sentences:
        # print(inputs)
        input_lens, sen_lens, max_sen_len = get_sentence_lens(inputs)
        max_mask_len = max_sen_len
    else:
        input_lens = get_lens(inputs)
        mask_lens = get_lens(input_masks)
        max_mask_len = np.max(mask_lens)

    q_lens = get_lens(questions)
    max_q_len = np.max(q_lens)

    max_input_len = min(np.max(input_lens), config.max_allowed_inputs)

    if split_sentences:
        inputs = pad_inputs(inputs, input_lens, max_input_len,
                            "split_sentences", sen_lens, max_sen_len)
        input_masks = np.zeros(len(inputs))
    else:
        inputs = pad_inputs(inputs, input_lens, max_input_len)
        input_masks = pad_inputs(input_masks, mask_lens, max_mask_len, "mask")

    questions = pad_inputs(questions, q_lens, max_q_len)

    answers = np.stack(answers)

    rel_labels = np.zeros((len(rel_labels), len(rel_labels[0])))

    for i, tt in enumerate(rel_labels):
        rel_labels[i] = np.array(tt, dtype=int)

    candidate_size = len(candidates)

    if config.train_mode:
        num_train = int(len(questions) * 0.8)
        print(num_train)
        train = questions[:num_train], inputs[:num_train], \
            q_lens[:num_train],  \
            input_lens[:num_train], input_masks[:num_train], \
            answers[:num_train], rel_labels[:num_train]

        valid = questions[num_train:], inputs[num_train:], \
            q_lens[num_train:], input_lens[num_train:], \
            input_masks[num_train:], \
            answers[num_train:], rel_labels[num_train:]
        return train, valid, word_embedding, max_q_len, max_input_len, max_mask_len, \
            rel_labels.shape[1], len(
                vocab), candidate_size, candid2idx, idx2candid, w2idx, idx2w

    else:
        test = questions, inputs, q_lens, input_lens, input_masks, answers, rel_labels
        return test, word_embedding, max_q_len, max_input_len, max_mask_len, \
            rel_labels.shape[1], len(vocab), candidate_size


def main():
    from dmn_plus import Config
    config = Config()
    train, valid, word_embedding, max_q_len, max_input_len, max_mask_len, \
        rel_labels, vocab, candidate_size = load_data(config)
    print(len(train[0]))
    print(len(valid[0]))


if __name__ == '__main__':
    main()
