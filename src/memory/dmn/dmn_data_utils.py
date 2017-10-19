from __future__ import division
from __future__ import print_function

import sys

import os as os
import json
import numpy as np
from functools import reduce

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)

import data_utils

# temporary paths
DATA_DIR = 'data/tree'
CANDID_PATH = 'data/tree/candidates.txt'

# can be sentence or word
input_mask_mode = "sentence"

# adapted from https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano/


def transform(data):
    def func(line):
        (story, query, answer) = line
        # print(story)
        if len(story):
            story = reduce(lambda x, y: x + ['.'] + y, story)
            story = ' '.join(story)
        else:
            story = '此 乃 空 文 . '
        query = ' '.join(query)
        # answer = ' '.join(answer)
        return {'C': story, 'Q': query, 'A': answer, 'S': []}

    target_data = list(map(func, data))
    return target_data


def data_transform():
    source_train_data, source_test_data, source_val_data, candidate_size = get_source_data()

    target_train_data = transform(source_train_data)
    target_test_data = transform(source_test_data)
    target_val_data = transform(source_val_data)

    return target_train_data, target_val_data, target_test_data, candidate_size


def get_source_data():
    candidates, candid2idx, idx2candid = data_utils.load_candidates(
        candidates_f=CANDID_PATH)
    candidate_size = len(candidates)
    # print(candidates)
    # print(idx2candid)
    train_data, test_data, val_data = data_utils.load_dialog(
        data_dir=DATA_DIR,
        candid_dic=candid2idx, dmn=True)
    # print(train_data[:5])
    return train_data, test_data, val_data, candidate_size


def create_vector(word, word2vec, word_vector_size, silent=True):
    # if the word is missing from Glove, create some fake vector and store in glove!
    vector = np.random.uniform(0.0, 1.0, (word_vector_size,))
    word2vec[word] = vector
    if (not silent):
        print("utils.py::create_vector => %s is missing" % word)
    return vector


def process_word(word, word2vec, vocab, ivocab, word_vector_size,
                 to_return="word2vec", silent=True):
    if not word in word2vec:
        create_vector(word, word2vec, word_vector_size, silent)
    if not word in vocab:
        next_index = len(vocab)
        vocab[word] = next_index
        ivocab[next_index] = word

    if to_return == "word2vec":
        return word2vec[word]
    elif to_return == "index":
        return vocab[word]
    elif to_return == "onehot":
        raise Exception("to_return = 'onehot' is not implemented yet")


def process_input(data_raw, floatX, word2vec, vocab, ivocab,
                  embed_size, split_sentences=False):
    questions = []
    inputs = []
    answers = []
    input_masks = []
    relevant_labels = []
    for x in data_raw:
        if split_sentences:
            inp = x["C"].lower().split(' . ')
            inp = [w for w in inp if len(w) > 0]
            inp = [i.split() for i in inp]
            # print(inp)
            # print('2:',inp)
        else:
            inp = x["C"].lower().split(' ')
            inp = [w for w in inp if len(w) > 0]

        # print(x["Q"])
        q = x["Q"].lower().split(' ') if len(x["Q"]) else ['空']
        # print(q)
        q = [w for w in q if len(w) > 0]

        # a = x["A"].lower().split(' ')
        # a = [w for w in a if len(w) > 0]

        if split_sentences:
            inp_vector = [[process_word(word=w,
                                        word2vec=word2vec,
                                        vocab=vocab,
                                        ivocab=ivocab,
                                        word_vector_size=embed_size,
                                        to_return="index") for w in s] for s in inp]
        else:
            inp_vector = [process_word(word=w,
                                       word2vec=word2vec,
                                       vocab=vocab,
                                       ivocab=ivocab,
                                       word_vector_size=embed_size,
                                       to_return="index") for w in inp]

        q_vector = [process_word(word=w,
                                 word2vec=word2vec,
                                 vocab=vocab,
                                 ivocab=ivocab,
                                 word_vector_size=embed_size,
                                 to_return="index") for w in q]

        # a_vector = [process_word(word=w,
        #                          word2vec=word2vec,
        #                          vocab=vocab,
        #                          ivocab=ivocab,
        #                          word_vector_size=embed_size,
        #                          to_return="index") for w in a]

        if split_sentences:
            inputs.append(inp_vector)
        else:
            inputs.append(np.vstack(inp_vector).astype(floatX))

        questions.append(np.vstack(q_vector).astype(floatX))

        # answers.append(np.vstack(a_vector).astype(floatX))

        # NOTE: here answer is a label index.
        answers.append(x['A'])

        if not split_sentences:
            if input_mask_mode == 'word':
                input_masks.append(
                    np.array([index for index, w in enumerate(inp)], dtype=np.int32))
            elif input_mask_mode == 'sentence':
                input_masks.append(
                    np.array([index for index, w in enumerate(inp) if w == '.'], dtype=np.int32))
            else:
                raise Exception("invalid input_mask_mode")

        relevant_labels.append(x["S"])

    return inputs, questions, answers, input_masks, relevant_labels


def create_embedding(word2vec, ivocab, embed_size):
    embedding = np.zeros((len(ivocab), embed_size))
    for i in range(len(ivocab)):
        word = ivocab[i]
        embedding[i] = word2vec[word]
    return embedding


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
            padded_sentences = [np.pad(s, (0, max_sen_len - sen_lens[i][j]),
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
    vocab = {}
    ivocab = {}

    train_data, val_data, test_data, candidate_size = data_transform()
    # print(train_data[:5])

    if config.word2vec_init:
        # remain to implement, to give a word2vec in advance
        # assert config.embed_size == 100
        # word2vec = load_glove(config.embed_size)
        pass
    else:
        word2vec = {}

    process_word(word="<eos>",
                 word2vec=word2vec,
                 vocab=vocab,
                 ivocab=ivocab,
                 word_vector_size=config.embed_size,
                 to_return="index")

    print('==> get train inputs')
    train_data = process_input(train_data, config.floatX,
                               word2vec, vocab, ivocab,
                               config.embed_size, split_sentences)

    print('==> get validate inputs')
    val_data = process_input(val_data, config.floatX,
                             word2vec, vocab, ivocab,
                             config.embed_size, split_sentences)

    print('==> get test inputs')
    test_data = process_input(test_data, config.floatX,
                              word2vec, vocab, ivocab,
                              config.embed_size, split_sentences)

    if config.word2vec_init:
        # remain to implement, to give a word2vec in advance
        # assert config.embed_size == 100
        # word_embedding = create_embedding(word2vec, ivocab, config.embed_size)
        pass
    else:
        word_embedding = np.random.uniform(-config.embedding_init,
                                           config.embedding_init,
                                           (len(ivocab), config.embed_size))

    inputs, questions, answers, input_masks, rel_labels = train_data \
        if config.train_mode else test_data

    # print('inputs:', inputs[:5])
    # print('questions:', questions[:5])
    # print('answers:', answers[:5])
    # print('input_masks:', input_masks[:5])
    # print('rel_labels:', rel_labels[:5])

    if split_sentences:
        input_lens, sen_lens, max_sen_len = get_sentence_lens(inputs)
        max_mask_len = max_sen_len
    else:
        input_lens = get_lens(inputs)
        mask_lens = get_lens(input_masks)
        max_mask_len = np.max(mask_lens)

    q_lens = get_lens(questions)
    max_q_len = np.max(q_lens)

    # a_lens = get_lens(answers)
    # max_a_len = np.max(a_lens)

    max_input_len = min(np.max(input_lens), config.max_allowed_inputs)

    # print('max_input_len:', np.max(input_lens))
    # print('max_sen_len:', max_sen_len)
    # print('max_q_len:', max_q_len)

    # pad out arrays to max
    if split_sentences:
        inputs = pad_inputs(inputs, input_lens, max_input_len,
                            "split_sentences", sen_lens, max_sen_len)
        input_masks = np.zeros(len(inputs))
    else:
        inputs = pad_inputs(inputs, input_lens, max_input_len)
        input_masks = pad_inputs(input_masks, mask_lens, max_mask_len, "mask")

    questions = pad_inputs(questions, q_lens, max_q_len)
    # answers = pad_inputs(answers, a_lens, max_a_len)

    answers = np.stack(answers)

    rel_labels = np.zeros((len(rel_labels), len(rel_labels[0])))

    for i, tt in enumerate(rel_labels):
        rel_labels[i] = np.array(tt, dtype=int)

    # print(len(inputs[0]))
    # print(len(inputs[0][0]))
    # print(len(questions[0]))

    # print('inputs:', inputs[:5])
    # print('questions:', questions[:5])
    # print('answers:', answers[:5])
    # print('input_masks:', input_masks[:5])
    # print('rel_labels:', rel_labels[:5])

    if config.train_mode:
        train = questions[:config.num_train], inputs[:config.num_train], \
            q_lens[:config.num_train],  \
            input_lens[:config.num_train], input_masks[:config.num_train], \
            answers[:config.num_train], rel_labels[:config.num_train]

        valid = questions[config.num_train:], inputs[config.num_train:], \
            q_lens[config.num_train:], input_lens[config.num_train:], \
            input_masks[config.num_train:], \
            answers[config.num_train:], rel_labels[config.num_train:]
        return train, valid, word_embedding, max_q_len, max_input_len, max_mask_len, \
            rel_labels.shape[1], len(vocab), candidate_size

    else:
        test = questions, inputs, q_lens, input_lens, input_masks, answers, rel_labels
        return test, word_embedding, max_q_len, max_input_len, max_mask_len, \
            rel_labels.shape[1], len(vocab), candidate_size


def main():
    from dmn_plus import Config
    config = Config()
    data, _, _, _, _, _, _, _ = load_data(config)
    p = np.random.permutation(len(data[0]))
    print(p)
    qp, ip, ql, il, im, a, r = data
    qp, ip, ql, il, im, a, r = qp[p], ip[p], ql[p], il[p], im[p], a[p], r[p]


if __name__ == '__main__':
    main()
