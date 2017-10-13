import argparse
import pickle as pkl
import gensim
import sys
import os
import time

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)

from utils.query_util import tokenize

import numpy as np
import tensorflow as tf
from sklearn import metrics

import data_utils
import memn2n_lstm as memn2n
import memn2n2 as memn2n2
import config as config
import heapq
import operator
import memory.config as config
dir_path = os.path.dirname(os.path.realpath(__file__))

grandfatherdir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

if config.MULTILABEL >= 1:
    DATA_DIR = grandfatherdir + '/data/memn2n/train/multi_tree'
else:
    DATA_DIR = grandfatherdir + '/data/memn2n/train/tree'
if config.MULTILABEL >= 1:
    P_DATA_DIR = grandfatherdir + '/data/memn2n/processed/multiple/'
    CKPT_DIR = grandfatherdir + '/model/memn2n/ckpt_mlt'
else:
    P_DATA_DIR = grandfatherdir + '/data/memn2n/processed/'
    CKPT_DIR = grandfatherdir + '/model/memn2n/ckpt2'
W2V_DIR = grandfatherdir + '/model/w2v/'
HOPS = config.HOPS
BATCH_SIZE = config.BATCH_SIZE
EMBEDDING_SIZE = config.EMBEDDING_SIZE

'''
    dictionary of models
        select model from here
model = {
        'memn2n' : memn2n.MemN2NDialog
        }# add models, as you implement

'''

'''
    run prediction on dataset

'''


def batch_predict(model, S, Q, n, batch_size):
    preds = []
    loss = []
    for start in range(0, n, batch_size):
        end = start + batch_size
        s = S[start:end]
        q = Q[start:end]
        pred, top_prob = model.predict(s, q)
        # print(pred.indices, top_prob.values)
        preds += list(pred.indices)
    return preds


'''
    preprocess data

'''


def prepare_data(args):
    # get candidates (restaurants)
    candidates, candid2idx, idx2candid = data_utils.load_candidates(
        candidates_f=os.path.join(DATA_DIR, 'candidates.txt'))
    # get data
    train, test, val = data_utils.load_dialog(
        data_dir=DATA_DIR,
        candid_dic=candid2idx)
    ##
    # get metadata
    metadata = data_utils.build_vocab(train + test + val, candidates)

    ###
    # write data to file
    data_ = {
        'candidates': candidates,
        'train': train,
        'test': test,
        'val': val
    }
    with open(P_DATA_DIR + 'data.pkl', 'wb') as f:
        pkl.dump(data_, f)

    ###
    # save metadata to disk
    metadata['candid2idx'] = candid2idx
    metadata['idx2candid'] = idx2candid

    # # build embeddings
    # w2idx = metadata['w2idx']
    # print('Loading word2vec...')
    # w2v_model = gensim.models.Word2Vec.load(
    #     os.path.join(W2V_DIR, 'dic_18_unk_short.bin'))

    # embeddings = list()
    # for word, _ in w2idx.items():
    #     if w2v_model.__contains__(word.strip()):
    #         vector = w2v_model.__getitem__(word.strip())
    #     else:
    #         # print('unk:', word)
    #         vector = w2v_model.__getitem__('unk')
    #     # print(type(vector))
    #     embeddings.append(vector)
    # embeddings = np.asarray(embeddings)
    # metadata['embeddings'] = embeddings

    with open(P_DATA_DIR + 'metadata.pkl', 'wb') as f:
        pkl.dump(metadata, f)


'''
    parse arguments

'''


def parse_args(args):
    parser = argparse.ArgumentParser(
        description='Train Model for Goal Oriented Dialog Task : bAbI(6)')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-i', '--infer', action='store_true',
                       help='perform inference in an base session')
    group.add_argument('--ui', action='store_true',
                       help='interact through web app(flask); do not call this from cmd line')
    group.add_argument('-t', '--train', action='store_true',
                       help='train model')
    group.add_argument('-d', '--prep_data', action='store_true',
                       help='prepare data')
    # parser.add_argument('--task_id', required=False, type=int, default=1,
    #                     help='Task Id in bAbI (6) tasks {1-6}')
    parser.add_argument('--batch_size', required=False, type=int, default=16,
                        help='you know what batch size means!')
    parser.add_argument('--epochs', required=False, type=int, default=200,
                        help='num iteration of training over train set')
    parser.add_argument('--eval_interval', required=False, type=int, default=5,
                        help='num iteration of training over train set')
    parser.add_argument('--log_file', required=False, type=str, default='log.txt',
                        help='enter the name of the log file')
    args = vars(parser.parse_args(args))
    return args


class InteractiveSession():
    def __init__(self, model, idx2candid, w2idx, n_cand, memory_size):
        self.context = []
        self.u = None
        self.r = None
        self.nid = 1
        self.model = model
        self.idx2candid = idx2candid
        self.w2idx = w2idx
        self.n_cand = model._candidates_size
        self.memory_size = memory_size
        self.model = model

    def reply(self, msg):
        line = msg.strip().lower()
        if line == 'clear':
            self.context = []
            self.nid = 1
            reply_msg = 'memory cleared!'
        else:
            if config.MULTILABEL >= 1:
                u = tokenize(line)
                print('context:', self.context)
                data = [(self.context, u, -1)]
                print('data:', data)
                s, q, a = data_utils.vectorize_data(data,
                                                    self.w2idx,
                                                    self.model._sentence_size,
                                                    1,
                                                    self.n_cand,
                                                    self.memory_size)
                preds, top_probs = self.model.predict(s, q)
                # preds = preds.indices[0]
                preds = preds.indices[0].tolist()
                top_probs = top_probs.values[0]
                print(top_probs)
                r = []
                for i, pred in enumerate(preds):
                    r.append(self.idx2candid[pred])
                reply_msg = ','.join(r)
                r = tokenize(reply_msg)
                u.append('$u')
                # u.append('#' + str(self.nid))
                r.append('$r')
                # r.append('#' + str(self.nid))
                self.context.append(u)
                self.context.append(r)
                print('context:', self.context)
                self.nid += 1
            else:
                u = data_utils.tokenize(line)
                data = [(self.context, u, -1)]
                s, q, a = data_utils.vectorize_data(data,
                                                    self.w2idx,
                                                    self.model._sentence_size,
                                                    1,
                                                    self.n_cand,
                                                    self.memory_size)
                preds = self.model.predict(s, q)
                r = self.idx2candid[preds[0]]
                reply_msg = r
                r = data_utils.tokenize(r)
                u.append('$u')
                # u.append('#' + str(self.nid))
                r.append('$r')
                # r.append('#' + str(self.nid))
                self.context.append(u)
                self.context.append(r)
                self.nid += 1

        return reply_msg


def main(args):
    # parse args
    args = parse_args(args)

    # prepare data
    if args['prep_data']:
        print('\n>> Preparing Data\n')
        prepare_data(args)
        sys.exit()

    # ELSE
    # read data and metadata from pickled files
    with open(P_DATA_DIR + 'metadata.pkl', 'rb') as f:
        metadata = pkl.load(f)
    with open(P_DATA_DIR + 'data.pkl', 'rb') as f:
        data_ = pkl.load(f)

    # read content of data and metadata
    candidates = data_['candidates']
    candid2idx, idx2candid = metadata['candid2idx'], metadata['idx2candid']

    # get train/test/val data
    train, test, val = data_['train'], data_['test'], data_['val']

    # gather more information from metadata
    sentence_size = metadata['sentence_size']
    w2idx = metadata['w2idx']  # is a list
    idx2w = metadata['idx2w']
    memory_size = metadata['memory_size']
    vocab_size = metadata['vocab_size']
    n_cand = metadata['n_cand']
    candidate_sentence_size = metadata['candidate_sentence_size']
    # embeddings = metadata['embeddings']

    # vectorize candidates
    candidates_vec = data_utils.vectorize_candidates(
        candidates, w2idx, candidate_sentence_size)

    print('---- memory config ----')
    print('embedding size:', EMBEDDING_SIZE)
    print('batch_size:', BATCH_SIZE)
    print('memory_size:', memory_size)
    print('vocab_size:', vocab_size)
    print('candidate_size:', n_cand)
    print('candidate_sentence_size:', candidate_sentence_size)
    print('hops:', HOPS)
    print('---- end ----')
    ###
    # create model
    # model = model['memn2n'](  # why?
    model = memn2n.MemN2NDialog(
        batch_size=BATCH_SIZE,
        vocab_size=vocab_size,
        candidates_size=n_cand,
        sentence_size=sentence_size,
        embedding_size=EMBEDDING_SIZE,
        candidates_vec=candidates_vec,
        hops=HOPS
    )

    # model = memn2n2.MemN2NDialog(
    #     batch_size=BATCH_SIZE,
    #     vocab_size=vocab_size,
    #     candidates_size=n_cand,
    #     sentence_size=sentence_size,
    #     embedding_size=EMBEDDING_SIZE,
    #     candidates_vec=candidates_vec,
    #     embeddings=embeddings,
    #     hops=HOPS
    # )

    # gather data in batches
    train, val, test, batches = data_utils.get_batches(
        train, val, test, metadata, batch_size=BATCH_SIZE)

    # for t in train['q']:
    #     print(recover_sentence(t, idx2w))

    if args['train']:
        # training starts here
        epochs = args['epochs']
        eval_interval = args['eval_interval']
        #
        # training and evaluation loop
        print('\n>> Training started!\n')
        # write log to file
        log_handle = open(dir_path + '/../../log/' + args['log_file'], 'w')
        cost_total = 0.
        best_cost = 100
        # best_validation_accuracy = 0.
        lowest_val_acc = 0.8
        total_begin = time.clock()
        begin = time.clock()
        for i in range(epochs + 1):

            for start, end in batches:
                s = train['s'][start:end]
                q = train['q'][start:end]
                # print(len(q))
                a = train['a'][start:end]
                if config.MULTILABEL >= 1:
                    # convert to one hot
                    one_hot = np.zeros((end - start, n_cand))
                    for aa in range(end - start):
                        for index in a[aa]:
                            one_hot[aa][index] = 1
                    a = one_hot
                cost_total += model.batch_fit(s, q, a)
            if config.MULTILABEL >= 1:
                if i % 1 == 0 and i:
                    print('stage...', i, cost_total)
                    if cost_total < best_cost:
                        print('saving model...', i, '++',
                              str(best_cost) + '-->' + str(cost_total))
                        best_cost = cost_total
                        model.saver.save(model._sess, CKPT_DIR + '/memn2n_model.ckpt',
                                         global_step=i)
            else:
                if i % 1 == 0 and i:
                    print('stage...', i)
                    if i % eval_interval == 0 and i:
                        train_preds = batch_predict(model, train['s'], train['q'], len(
                            train['s']), batch_size=BATCH_SIZE)
                        # for i in range(len(train['q'])):
                        #     if train_preds[i] != train['a'][i]:
                        #         print(recover_sentence(train['q'][i], idx2w),
                        #               recover_cls(train_preds[i], idx2candid),
                        #               recover_cls(train['a'][i], idx2candid))
                        val_preds = batch_predict(model, val['s'], val['q'], len(
                            val['s']), batch_size=BATCH_SIZE)
                        train_acc = metrics.accuracy_score(
                            np.array(train_preds), train['a'])
                        val_acc = metrics.accuracy_score(val_preds, val['a'])
                        end = time.clock()
                        print('Epoch[{}] : <ACCURACY>\n\ttraining : {} \n\tvalidation : {}'.
                              format(i, train_acc, val_acc))
                        print('time:{}'.format(end - begin))
                        log_handle.write('{} {} {} {}\n'.format(i, train_acc, val_acc,
                                                                cost_total / (eval_interval * len(batches))))
                        cost_total = 0.  # empty cost
                        begin = end
                        #
                        # save the best model, to disk
                        # if val_acc > best_validation_accuracy:
                        # best_validation_accuracy = val_acc
                        if train_acc > lowest_val_acc:
                            print('saving model...', train_acc, lowest_val_acc)
                            lowest_val_acc = train_acc
                            model.saver.save(model._sess, CKPT_DIR + '/memn2n_model.ckpt',
                                             global_step=i)
        # close file
        total_end = time.clock()
        print('Total time: {} minutes.'.format((total_end - total_begin) / 60))
        log_handle.close()

    else:  # inference
        ###
        # restore checkpoint
        ckpt = tf.train.get_checkpoint_state(CKPT_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            print('\n>> restoring checkpoint from', ckpt.model_checkpoint_path)
            model.saver.restore(model._sess, ckpt.model_checkpoint_path)
        # base(model, idx2candid, w2idx, sentence_size, BATCH_SIZE, n_cand, memory_size)

        # create an base session instance
        isess = InteractiveSession(
            model, idx2candid, w2idx, n_cand, memory_size)

        if args['infer']:
            query = ''
            while query != 'exit':
                query = input('>> ')
                print('>> ' + isess.reply(query))
        elif args['ui']:
            return isess


def recover_sentence(sentence_idx, idx2w):
    sentence = [idx2w[idx - 1] for idx in sentence_idx if idx != 0]
    return ','.join(sentence)


def recover_cls(idx, idx2cls):
    if not isinstance(idx, np.int64):
        idx = idx[0]
    result = idx2cls[idx]
    return result


def launch_multiple_session():
    return


# _______MAIN_______
if __name__ == '__main__':
    main(sys.argv[1:])
