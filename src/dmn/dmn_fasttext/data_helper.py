from __future__ import absolute_import

import sys

import os
import json
import pickle
import time
import numpy as np
from collections import OrderedDict
from functools import reduce

parentdir = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))
sys.path.insert(0, parentdir)

grandfatherdir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, grandfatherdir)
sys.path.append(grandfatherdir)
from utils.query_util import tokenize
from utils.translator import Translator
from dmn.dmn_fasttext.config import Config
from gensim.models.wrappers import FastText
from dmn.dmn_fasttext.vector_helper import getVector

# EMPTY='EMPTY'
# PAD='PAD'
# NONE=''
# UNK='UNK'

config = Config()

# if config.word:
#     print('loading fasttext model...')
#     model = FastText.load_fasttext_format('/opt/fasttext/model/skipgram.bin')

translator = Translator()


# def ff_embedding_local(word):
#     if model.__contains__(word.strip()):
#         return model[word]
#     else:
#         print(word)
#         return model[UNK]

translator = Translator()


def load_candidates(candidates_f):
    candidates, candid2idx, idx2candid = [], {}, {}
    with open(candidates_f) as f:
        for i, line in enumerate(f):
            candid2idx[line.strip()] = i
            idx2candid[i] = line.strip()
            candidates.append(line.strip())

    return candidates, candid2idx, idx2candid


def load_dialog(sentences, data_dir, candid_dic, char=1):
    train_file = os.path.join(data_dir, 'train.txt')
    test_file = os.path.join(data_dir, 'test.txt')
    val_file = os.path.join(data_dir, 'val.txt')

    train_data = get_dialogs(sentences, train_file, candid_dic, char)
    test_data = get_dialogs(sentences, test_file, candid_dic, char)
    val_data = get_dialogs(sentences, val_file, candid_dic, char)
    return train_data, test_data, val_data


def get_dialogs(sentences, f, candid_dic,  char=1):
    '''Given a file name, read the file, retrieve the dialogs, and then convert the sentences into a single dialog.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    with open(f) as f:
        return parse_dialogs_per_response(sentences, f.readlines(), candid_dic, char)


def parse_dialogs_per_response(sentences, lines, candid_dic, char=1):
    '''
        Parse dialogs provided in the babi tasks format
    '''
    data = []
    context = []
    u = None
    r = None
    # print(candid_dic)
    for line in lines:
        line = line.strip()
        if line:
            if '\t' in line:
                # print(line)
                try:
                    u, r, salt = line.split('\t')
                except:
                    print(line)
                    exit(-1)
                if config.multi_label:
                    a = [candid_dic[single_r] for single_r in r.split(",")]
                else:
                    if r not in candid_dic:
                        continue
                    a = candid_dic[r]
                u = tokenize(u, char=char)
                if config.fix_vocab:
                    r = translator.en2cn(r)
                r = tokenize(r, char=char)
                placeholder = salt == 'placeholder'
                if config.fix_vocab:
                    salt = translator.en2cn(salt)
                salt = tokenize(salt, char=char)

                sentences.add(','.join(u))
                sentences.add(','.join(r))
                sentences.add(','.join(salt))

                # print(u)
                # temporal encoding, and utterance/response encoding
                # data.append((context[:],u[:],candid_dic[' '.join(r)]))
                data.append((context[:], u[:], a))
                context.append(u)
                # r = r if placeholder == 'placeholder' else r + salt
                context.append(r)
                # if salt != 'placeholder':
                #     context.append(salt)
        else:
            # clear context
            context = []
    # print(data)
    sentences.add(config.EMPTY)
    return data


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


def load_raw_data(config):
    print('Load raw data.....')
    candidates, candid2idx, idx2candid = load_candidates(
        candidates_f=config.candid_path)
    candidate_size = len(candidates)

    char = 2 if config.word else 1
    sentences = set()
    train_data, test_data, val_data = load_dialog(sentences,
                                                  data_dir=config.data_dir,
                                                  candid_dic=candid2idx,  char=char)

    train_data += test_data
    train_data += val_data

    inputs = []
    questions = []
    answers = []
    relevant_labels = []
    input_masks = []
    for data in train_data:
        inp, question, answer = data
        inp = inp[-config.max_memory_size:]
        if len(inp) == 0:
            inp = [[config.EMPTY]]
        inputs.append(inp)
        if len(question) == 0:
            question = [config.EMPTY]
        questions.append(question)
        answers.append(answer)
        relevant_labels.append([0])

    # if not config.split_sentences:
    #     if input_mask_mode == 'word':
    #         input_masks.append(
    #             np.array([index for index, w in enumerate(inp)], dtype=np.int32))
    #     elif input_mask_mode == 'sentence':
    #         input_masks.append(
    #             np.array([index for index, w in enumerate(inp) if w == '.'], dtype=np.int32))
    #     else:
    #         raise Exception("invalid input_mask_mode")

    if config.split_sentences:
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

    if not config.word:
        with open(config.vocab_path, 'r') as f:
            vocab = json.load(f)
        w2idx = dict((c, i + 1) for i, c in enumerate(vocab))
        idx2w = dict((i + 1, c) for i, c in enumerate(vocab))
        vocab_size = len(vocab)
    else:
        w2idx = None
        idx2w = None
        vocab_size = 0

    return train, valid, max_q_len, max_input_len, max_sen_len, max_mask_len, \
        relevant_labels.shape[1], candidate_size, candid2idx, \
        idx2candid, w2idx, idx2w, vocab_size, sentences


def pad_inputs(inputs, lens, max_len, mode="", sen_lens=None, max_sen_len=None, dtype='<U3'):
    if mode == "mask":
        padded = [np.pad(inp, (0, max_len - lens[i]), 'constant',
                         constant_values=0) for i, inp in enumerate(inputs)]
        return np.vstack(padded)

    elif mode == "split_sentences":
        padded = np.zeros((len(inputs), max_len, max_sen_len), dtype=dtype)
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


def vectorize_data(data, config, metadata):
    questions, inputs, q_lens, sen_lens, input_lens, \
        input_masks, answers, relevant_labels = data

    sentences_embedding = metadata['sentences_embedding']
    # q_embedding = sentences_embedding['questions_embedding']
    # a_embedding = sentences_embedding['answers_embedding']
    max_input_len = metadata['max_input_len']

    inputs_embeddings = []
    for inp in inputs:
        inp_embeddings = []
        for line in inp:
            line = ','.join(line)
            embedding = sentences_embedding[line]
            inp_embeddings.append(embedding)
        for _ in range(max_input_len - len(inp)):
            inp_embeddings.append(sentences_embedding[config.EMPTY])
        inputs_embeddings.append(inp_embeddings)
    inputs_embeddings = np.asarray(inputs_embeddings, dtype=np.float32)

    questions_embeddings = []
    for question in questions:
        question = ','.join(question)
        questions_embeddings.append(sentences_embedding[question])
    questions_embeddings = np.asarray(
        questions_embeddings, dtype=np.float32)

    data = questions_embeddings, inputs_embeddings, q_lens, input_lens, \
        input_masks, answers, relevant_labels
    return data


def sentence_embedding_core(config, sentences, w2idx):
    split_sentences = [sen.split(',') for sen in sentences]
    sen_lens = [len(split_sentence) for split_sentence in split_sentences]
    max_len = max(sen_lens)

    # print('split_sentences:', split_sentences[:2])


    pad_sentences = pad_inputs(split_sentences, sen_lens, max_len)
    # with open('debug.txt','a') as f:
    #     print('split_sentences ==>',split_sentences[:4],file=f)
    #     print('pad_sentences ==>',pad_sentences[:4],file=f)

    sentences_embedding = OrderedDict()
    for sen in pad_sentences:
        sen = sen.tolist()
        sen = list(map(lambda x: [x, config.PAD][x == config.NONE], sen))
        # print(sen)
        if config.word:
            # sen_embedding = [ff_embedding_local(word) for word in sen]
            sen_embedding = [getVector(word) for word in sen]
        else:
            sen_embedding = [w2idx.get(word, 3) for word in sen]
        sen = [i for i in sen if i!=config.PAD]
        join_sen = ','.join(sen)
        sentences_embedding[join_sen] = sen_embedding
    if config.word:
        # inp_empty_embedding = [ff_embedding_local(PAD) for _ in range(max_len)]
        inp_empty_embedding = [getVector(config.PAD) for _ in range(max_len)]
    else:
        inp_empty_embedding = [3 for _ in range(max_len)]
    sentences_embedding[config.EMPTY] = inp_empty_embedding
    sentences_embedding[config.PAD] = [getVector(config.PAD)] * max_len
    return sentences_embedding


def sentence_embedding(config, sentences, w2idx):
    sentences_embedding = sentence_embedding_core(
        config, sentences, w2idx)
    return sentences_embedding
