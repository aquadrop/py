import sys

import os
import json
import pickle
import numpy as np
from functools import reduce

parentdir = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))
sys.path.insert(0, parentdir)

grandfatherdir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, grandfatherdir)

import dmn.data_utils as data_utils
from dmn.dmn_plus2 import Config
# from dmn.vector_helper import getVector
from utils.embedding_util import ff_embedding

# temporary paths
# DATA_DIR = 'data/memn2n/train/tree/origin'
# CANDID_PATH = 'data/memn2n/train/tree/origin/candidates.txt'

DATA_DIR = 'data/memn2n/train/tree/origin/'
CANDID_PATH = 'data/memn2n/train/tree/origin/candidates.txt'

MULTI_DATA_DIR = 'data/memn2n/train/multi_tree'
MULTI_CANDID_PATH = 'data/memn2n/train/multi_tree/candidates.txt'

VOCAB_PATH = 'data/char_table/vocab.txt'

# can be sentence or word
input_mask_mode = "sentence"

from gensim.models.wrappers import FastText

config = Config()
if config.word2vec_init:
    print('loading fasttext model..')
    model = FastText.load_fasttext_format('/opt/fasttext/model/wiki.zh.bin')


def ff_embedding_local(word):
    return model[word]


def get_candidates_word_dict():
    with open(os.path.join(grandfatherdir, VOCAB_PATH), 'r') as f:
        vocab = json.load(f)
    w2idx = dict((c, i + 1) for i, c in enumerate(vocab))

    candidates, candid2idx, idx2candid = data_utils.load_candidates(
        candidates_f=os.path.join(grandfatherdir, CANDID_PATH))

    return idx2candid, w2idx


def load_raw_data(data_path, candid_path, word_vector=False, split_sentences=True):
    print('Load raw data2.....')
    candidates, candid2idx, idx2candid = data_utils.load_candidates(
        candidates_f=candid_path)
    candidate_size = len(candidates)

    char = 2 if word2vec_init else 0
    train_data, test_data, val_data = data_utils.load_dialog(
        data_dir=data_path,
        candid_dic=candid2idx, char=char)

    train_data += test_data
    train_data += val_data

    inputs = []
    questions = []
    answers = []
    relevant_labels = []
    input_masks = []
    for data in train_data:
        inp, question, answer = data
        if len(inp) == 0:
            inp = [[' ']]
        inputs.append(inp)
        if len(question) == 0:
            question = [' ']
        questions.append(question)
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

    answers = [np.sum(np.eye(candidate_size)[np.asarray(a)], axis=0) for a in
               answers] if config.multi_label else answers
    answers = np.stack(answers)

    relevant_labels = np.zeros((len(relevant_labels), len(relevant_labels[0])))
    for i, tt in enumerate(relevant_labels):
        relevant_labels[i] = np.array(tt, dtype=int)

    num_train = int(len(questions) * 0.8)
    train = questions[:num_train], inputs[:num_train], \
        q_lens[:num_train], sen_lens[:num_train],\
        input_lens[:num_train], input_masks[:num_train], \
        answers[:num_train], relevant_labels[:num_train]
    valid = questions[num_train:], inputs[num_train:], \
        q_lens[num_train:], sen_lens[num_train:], input_lens[num_train:], \
        input_masks[num_train:], \
        answers[num_train:], relevant_labels[num_train:]
    print('inputs:', inputs[:3])
    print('questions:', questions[:3])

    return train, valid, max_q_len, max_input_len, max_sen_len, max_mask_len, \
        relevant_labels.shape[1], candidate_size, candid2idx, idx2candid


def vectorize_data(data, config, max_input_len, max_sen_len, max_q_len):
    questions, inputs, q_lens, sen_lens, input_lens, \
        input_masks, answers, relevant_labels = data
    # print(questions)
    if config.split_sentences:
        inputs = pad_inputs2(inputs, input_lens, max_input_len,
                             "split_sentences", sen_lens, max_sen_len)
        input_masks = np.zeros(len(inputs))
    else:
        inputs = pad_inputs2(
            inputs, input_lens, max_input_len)
        input_masks = pad_inputs2(
            input_masks, input_lens, max_input_len, "mask")

    questions = pad_inputs2(
        questions, q_lens, max_q_len)

    w2idx = {}
    idx2w = {}
    vocab_size = 0
    if not config.word_vector:
        with open(config.vocab_path, 'r') as f:
            vocab = json.load(f)
        w2idx = dict((c, i + 1) for i, c in enumerate(vocab))
        idx2w = dict((i + 1, c) for i, c in enumerate(vocab))
        vocab_size = len(vocab)

    # embedding
    # print('inputs embedding...')
    inputs_embeddings = []
    for inp in inputs:
        inp_embeddings = []
        for line in inp:
            line_embeddings = []
            for word in line:
                if config.word_vector:
                    line_embeddings.append(
                        ff_embedding_local(word))
                else:
                    line_embeddings.append(w2idx.get(word, 3))
            inp_embeddings.append(line_embeddings)
        inputs_embeddings.append(inp_embeddings)
    inputs_embeddings = np.asarray(inputs_embeddings, dtype=np.float32)

    # print('questions embedding...')
    questions_embeddings = []
    for question in questions:
        question_embedding = []
        # print(question)
        for word in question:
            if config.word_vector:
                question_embedding.append(
                    ff_embedding_local(word))
            else:
                question_embedding.append(w2idx.get(word, 3))
        questions_embeddings.append(question_embedding)
    questions_embeddings = np.asarray(
        questions_embeddings, dtype=np.float32)

    # print(question_embedding[3])
    data = questions_embeddings, inputs_embeddings, q_lens, input_lens, \
        input_masks, answers, relevant_labels
    return data, w2idx, idx2w, vocab_size


def process_data(data_raw, floatX, w2idx, split_sentences=True, memory_size=config.max_memory_size):
    questions = []
    inputs = []
    answers = []
    input_masks = []
    relevant_labels = []

    for data in data_raw:
        inp, question, answer = data
        inp = inp[-memory_size:]
        # print(inp)
        if len(inp) == 0:
            inp = [['']]
        if split_sentences:
            inp_vector = [[w2idx.get(w, 0) for w in s] for s in inp]
            inputs.append(inp_vector)
        else:
            inp = reduce(lambda x, y: x + ['.'] + y, inp)
            inp_vector = [w2idx.get(w, 0) for w in inp]
            inputs.append(np.vstack(inp_vector).astype(floatX))

        question = question if len(question) else ['']
        q_vector = [w2idx.get(w, 0) for w in question]
        # print(q_vector)
        questions.append(np.vstack(q_vector).astype(floatX))

        answers.append(answer)
        # print(answer)

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
    # print(inputs[0])
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
        lens[i] = len(t)
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
                0, max_len - lens[i]), (0, 0)), 'constant',
                constant_values=0)
            padded[i] = padded_sentences
        return padded

    padded = [np.pad(np.squeeze(inp, axis=1), (0, max_len - lens[i]),
                     'constant', constant_values=0) for i, inp in enumerate(inputs)]
    return np.vstack(padded)


def pad_inputs2(inputs, lens, max_len, mode="", sen_lens=None, max_sen_len=None):
    if mode == "mask":
        padded = [np.pad(inp, (0, max_len - lens[i]), 'constant',
                         constant_values=0) for i, inp in enumerate(inputs)]
        return np.vstack(padded)

    elif mode == "split_sentences":
        padded = np.zeros((len(inputs), max_len, max_sen_len), dtype='<U3')
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
                0, max_len - lens[i]), (0, 0)), 'constant',
                constant_values=0)
            padded[i] = padded_sentences
        return padded

    padded = [np.pad(inp, (0, max_len - lens[i]),
                     'constant', constant_values=0) for i, inp in enumerate(inputs)]
    # print(padded)
    return np.asarray(padded)


def process_word_core(word, w2idx, idx2w, word_embedding, word2vec, updated_embedding, init=False):
    def get_next_index(w2idx):
        words = list(w2idx.keys())
        # words.sort()
        for w in words:
            if w.startswith('reserved_'):
                next_index = w2idx[w]
                del w2idx[w]
                return next_index, w
        return 0, 'unk'

    # print(init)
    if not word in w2idx:
        if init:
            next_index = len(w2idx)
            w2idx[word] = next_index
            idx2w[next_index] = word
            if word.startswith('reserved_'):
                embedding = word2vec['unk']
            else:
                embedding = ff_embedding_local(word)
            word2vec[word] = embedding
            word_embedding.append(embedding)
        else:
            print(word)
            next_index, w = get_next_index(w2idx)
            print('next_index:{0},word:{1}'.format(next_index, w))
            if next_index == 0 and w == 'unk':
                print('****VOCAB SIZE OVERFLOW****')
                w2idx[word] = next_index
            else:
                w2idx[word] = next_index
                idx2w[next_index] = word
                # embedding = word2vec[w]
                new_embedding = ff_embedding_local(word)
                # index = word_embedding.index(embedding)
                word_embedding[next_index] = new_embedding
                updated_embedding[next_index] = new_embedding
                del word2vec[w]
                word2vec[word] = new_embedding


def process_word(data, w2idx, idx2w, word_embedding, word2vec, updated_embedding, init=False):
    for d in data:
        inp, question, _ = d
        for i in inp:
            for w in i:
                process_word_core(
                    w, w2idx, idx2w, word_embedding, word2vec, updated_embedding, init)
        for w in question:
            process_word_core(w, w2idx, idx2w, word_embedding,
                              word2vec, updated_embedding, init)


def vectorize_data(config, data, metadata, split_sentences=True):
    data = process_data(data, config.floatX, metadata['w2idx'])
    inputs, questions, answers, input_masks, rel_labels = data
    candidates = metadata['candidates']
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
    max_q_len = metadata['max_q_len']

    max_input_len = metadata['max_input_len']
    max_mask_len = metadata['max_mask_len']
    max_sen_len = metadata['max_sen_len']

    if split_sentences:
        inputs = pad_inputs(inputs, input_lens, max_input_len,
                            "split_sentences", sen_lens, max_sen_len)
        input_masks = np.zeros(len(inputs))
    else:
        inputs = pad_inputs(inputs, input_lens, max_input_len)
        input_masks = pad_inputs(input_masks, mask_lens, max_mask_len, "mask")

    questions = pad_inputs(questions, q_lens, max_q_len)

    candidate_size = len(candidates)

    answers = [np.sum(np.eye(candidate_size)[np.asarray(a)], axis=0) for a in
               answers] if config.multi_label else answers

    answers = np.stack(answers)

    rel_labels = np.zeros((len(rel_labels), len(rel_labels[0])))

    for i, tt in enumerate(rel_labels):
        rel_labels[i] = np.array(tt, dtype=int)

    candidate_size = len(candidates)

    # print(word2vec)

    data = questions[:], inputs[:], \
        q_lens[:], \
        input_lens[:], input_masks[:], \
        answers[:], rel_labels[:]

    # print(type(self.word_embedding))
    return data


def get_lens_info(data, metadata, split_sentences=True):
    data = process_data(data, config.floatX, metadata['w2idx'])
    inputs, questions, answers, input_masks, rel_labels = data
    rel_labels = np.array(rel_labels)
    candidates = metadata['candidates']
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

    return max_mask_len, max_q_len, max_input_len, rel_labels.shape[1]


def load_data(config, split_sentences=True):
    w2idx = {}
    idx2w = {}
    word_embedding = []
    word2vec = {}

    updated_embedding = {}

    metadata = dict()
    metadata['word_embedding'] = word_embedding
    metadata['updated_embedding'] = updated_embedding
    metadata['word2vec'] = word2vec
    data_dir = config.data_dir
    candid_path = config.candid_path

    train_data, val_data, test_data, candidates, candid2idx, idx2candid = \
        load_raw_data(data_dir, candid_path,
                      word2vec_init=config.word_vector)

    metadata['candidates'] = candidates
    metadata['candid2idx'] = candid2idx
    metadata['idx2candid'] = idx2candid
    metadata['candidate_size'] = len(candidates)
    if config.word2vec_init:
        print('Process word vector.')

        if os.path.exists(config.metadata_path):
            print('Updata.....')
            with open(config.metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            w2idx = metadata['w2idx']
            idx2w = metadata['idx2w']
            word_embedding = metadata['word_embedding']
            word2vec = metadata['word2vec']

            process_word(data=train_data, w2idx=w2idx, idx2w=idx2w,
                         word_embedding=word_embedding, word2vec=word2vec, updated_embedding=updated_embedding)
            process_word(data=val_data, w2idx=w2idx, idx2w=idx2w,
                         word_embedding=word_embedding, word2vec=word2vec, updated_embedding=updated_embedding)
            process_word(data=test_data, w2idx=w2idx, idx2w=idx2w,
                         word_embedding=word_embedding, word2vec=word2vec, updated_embedding=updated_embedding)
        else:
            print('Init.....')
            process_word_core('unk', w2idx, idx2w,
                              word_embedding, word2vec, updated_embedding, init=True)
            process_word(data=train_data, w2idx=w2idx, idx2w=idx2w,
                         word_embedding=word_embedding, word2vec=word2vec, updated_embedding=updated_embedding,
                         init=True)
            process_word(data=val_data, w2idx=w2idx, idx2w=idx2w,
                         word_embedding=word_embedding, word2vec=word2vec, updated_embedding=updated_embedding,
                         init=True)
            process_word(data=test_data, w2idx=w2idx, idx2w=idx2w,
                         word_embedding=word_embedding, word2vec=word2vec, updated_embedding=updated_embedding,
                         init=True)
            current_size = len(w2idx)
            for i in range(config.vocab_size - current_size):
                process_word_core('reserved_' + str(i), w2idx,
                                  idx2w, word_embedding, word2vec, updated_embedding, init=True)
        vocab_size = len(w2idx)
        # word_embedding = np.asarray(word_embedding)
    else:
        print('Process char level.....')

        with open(os.path.join(grandfatherdir, config.vocab_path), 'r') as f:
            vocab = json.load(f)
        w2idx = dict((c, i + 1) for i, c in enumerate(vocab))
        idx2w = dict((i + 1, c) for i, c in enumerate(vocab))
        vocab_size = len(vocab)
        metadata['w2idx'] = w2idx
        metadata['idx2w'] = idx2w
        metadata['vocab_size'] = vocab_size
        max_mask_len, max_q_len, max_input_len, num_supporting_facts = get_lens_info(
            train_data, metadata, split_sentences)
        max_mask_len2, max_q_len2, max_input_len2, _ = get_lens_info(
            val_data, metadata, split_sentences)
        max_mask_len3, max_q_len3, max_input_len3, _ = get_lens_info(
            test_data, metadata, split_sentences)
        max_mask_len = max([max_mask_len, max_mask_len2, max_mask_len3])
        max_q_len = max([max_q_len, max_q_len2, max_q_len3])
        max_input_len = max([max_input_len, max_input_len2, max_input_len3])
        metadata['max_mask_len'] = max_mask_len
        metadata['max_q_len'] = max_q_len
        metadata['max_input_len'] = max_input_len
        metadata['num_supporting_facts'] = num_supporting_facts
        metadata['max_sen_len'] = max_mask_len
        return train_data, val_data, test_data, metadata
        # word_embedding = np.random.uniform(
        #     -config.embedding_init,
        #     config.embedding_init,
        #     (vocab_size, config.embed_size))

    # train_data = process_data(train_data, config.floatX, w2idx)
    # val_data = process_data(val_data, config.floatX, w2idx)
    # test_data = process_data(test_data, config.floatX, w2idx)

    # print(len(train_data[0]))


def main():
    from dmn_plus2 import Config
    config = Config()
    train, valid, word_embedding, word2vec, updated_embedding, max_q_len, max_input_len, max_mask_len, \
        rel_labels, vocab_size, candidate_size, candid2idx, idx2candid, w2idx, idx2w = load_data(
            config)
    print(len(train[0]))
    print(len(valid[0]))


if __name__ == '__main__':
    main()
