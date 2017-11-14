from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

import time
import pickle
import argparse
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parentdir = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))
grandfatherdir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, parentdir)
sys.path.insert(0, grandfatherdir)

from utils.query_util import tokenize
from utils.translator import Translator

import dmn_data_utils2 as dmn_data_utils
from dmn_plus2 import Config, DMN_PLUS

translator = Translator()

EPOCH = 5

def prepare_data(args, config):
    # train, valid, word_embedding, word2vec, updated_embedding, max_q_len, max_input_len, max_sen_len, \
    #     num_supporting_facts, vocab_size, candidate_size, candid2idx, \
    #     idx2candid, w2idx, idx2w = dmn_data_utils.load_data(
    #         config, split_sentences=True)
    train_data, val_data, test_data, metadata = dmn_data_utils.load_data(
            config, split_sentences=True)
    # metadata = dict()
    data = dict()
    data['train'] = train_data
    data['valid'] = val_data

    # metadata['word_embedding'] = word_embedding
    # metadata['updated_embedding'] = updated_embedding
    # metadata['word2vec'] = word2vec
    #
    # metadata['max_q_len'] = max_q_len
    # metadata['max_input_len'] = max_input_len
    # metadata['max_sen_len'] = max_sen_len
    # metadata['num_supporting_facts'] = num_supporting_facts
    # metadata['vocab_size'] = vocab_size
    # metadata['candidate_size'] = candidate_size
    # metadata['candid2idx'] = candid2idx
    # metadata['idx2candid'] = idx2candid
    # metadata['w2idx'] = w2idx
    # metadata['idx2w'] = idx2w

    # print('after.')
    # print('updated_embedding:', updated_embedding)

    with open(config.metadata_path, 'wb') as f:
        pickle.dump(metadata, f, protocol=4)
    with open(config.data_path, 'wb') as f:
        pickle.dump(data, f, protocol=4)


def parse_args(args):
    parser = argparse.ArgumentParser(description='DMN-PLUS')
    parser.add_argument("-r", "--restore", action='store_true',
                        help="restore previously trained weights")
    parser.add_argument("-s", "--strong_supervision",
                        help="use labelled supporting facts (default=false)")
    # parser.add_argument("-t", "--dmn_type",
    #                     help="specify type of dmn (default=original)")
    parser.add_argument("-l", "--l2_loss", type=float,
                        default=0.001, help="specify l2 loss constant")
    parser.add_argument("-n", "--num_runs", type=int,
                        help="specify the number of model runs")
    parser.add_argument("-p", "--infer", action='store_true', help="predict")
    parser.add_argument("-t", "--train", action='store_true', help="train")
    parser.add_argument("-d", "--prep_data",
                        action='store_true', help="prepare data")

    args = vars(parser.parse_args(args))
    return args

def _check_restore_parameters(sess, saver, model_path):
    """ Restore the previously trained parameters if there are any. """
    print("--checking directory:", model_path)
    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt and ckpt.model_checkpoint_path:
        print("Loading parameters for the model")
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("Initializing fresh parameters for the model")

def main(args):
    args = parse_args(args)
    # print(args)

    config = Config()
    args['train'] = 'yeah'
    if args['prep_data']:
        print('\n>> Preparing Data\n')
        begin = time.clock()
        prepare_data(args, config)
        end = time.clock()
        print('>> Preparing Data Time:{}'.format(end - begin))
        sys.exit()

    if args['train']:

        print('Load metadata and data files (training mode)')
        with open(config.data_path, 'rb') as f:
            data = pickle.load(f)

        with open(config.metadata_path, 'rb') as f:
            metadata = pickle.load(f)

        train = data['train']
        valid = data['valid']
        train = dmn_data_utils.vectorize_data(config, train, metadata)
        valid = dmn_data_utils.vectorize_data(config, valid, metadata)

        model = DMN_PLUS(config, metadata)
        print('Training DMN-PLUS start')

        print('==> initializing variables')
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:

            # sum_dir = 'summaries/train/' + time.strftime("%Y-%m-%d %H %M")
            # if not os.path.exists(sum_dir):
            #     os.makedirs(sum_dir)
            # train_writer = tf.summary.FileWriter(sum_dir, session.graph)
            train_writer = None

            session.run(init)

            _check_restore_parameters(session, saver, model_path=config.ckpt_path)

            best_val_epoch = 0
            prev_epoch_loss = float('inf')
            best_val_loss = float('inf')
            best_val_accuracy = 0.0

            best_train_epoch = 0
            best_train_loss = float('inf')
            best_train_accuracy = 0.8

            if config.word2vec_init:
                session.run(model.embedding_init, feed_dict={
                                     model.embedding_placeholder: model.word_embedding})

            print('==> starting training')
            for epoch in range(config.max_epochs):
                if not (epoch % EPOCH == 0 and epoch > 1):
                    print('Epoch {}'.format(epoch))
                    _ = model.run_epoch(session, train, epoch, train_writer,
                                        train_op=model.train_step, train=True)
                    # _ = model.run_epoch(session, model.valid, epoch, train_writer,
                    #                     train_op=model.train_step, train=True)
                else:
                    print('Epoch {}'.format(epoch))
                    start = time.time()
                    train_loss, train_accuracy, train_error = model.run_epoch(
                        session, train, epoch, train_writer,
                        train_op=model.train_step, train=True, display=True)
                    valid_loss, valid_accuracy, valid_error = model.run_epoch(
                        session, valid, display=True)
                    # print('Training error:')
                    if train_accuracy > 0.99:
                        for e in train_error:
                            print(e)
                    # print('Validation error:')
                    print('Training loss: {}'.format(train_loss))
                    print('Validation loss: {}'.format(valid_loss))
                    print('Training accuracy: {}'.format(train_accuracy))
                    print('Vaildation accuracy: {}'.format(valid_accuracy))

                    if train_accuracy > best_train_accuracy:
                        print('Saving weights and updating best_train_loss:{} -> {},\
                               best_train_accuracy:{} -> {}'.format(best_train_loss, train_loss,\
                                                                    best_train_accuracy, train_accuracy))
                        best_train_accuracy = train_accuracy
                        saver.save(
                            session, config.ckpt_path + 'dmn.weights')
                        best_train_loss = train_loss
                        best_train_epoch = epoch
                    print('best_train_loss: {}'.format(best_train_loss))
                    print('best_train_epoch: {}'.format(best_train_epoch))
                    print('best_train_accuracy: {}'.format(best_train_accuracy))

                    # anneal
                    if train_loss > prev_epoch_loss * model.config.anneal_threshold:
                        model.config.lr /= model.config.anneal_by
                        print('annealed lr to %f' % model.config.lr)

                    prev_epoch_loss = train_loss

                    # if epoch - best_val_epoch > config.early_stopping:
                    #     break
                    print('Total time: {}'.format(time.time() - start))

            print('Best train accuracy:', best_train_accuracy)

    else:  # inference
        config.train_mode = False
        model = DMN_PLUS(config)
        print('Predict start')
        print('==> initializing variables')
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(init)

            # restore checkpoint
            ckpt = tf.train.get_checkpoint_state(config.ckpt_path)
            if ckpt and ckpt.model_checkpoint_path:
                print('\n>> restoring checkpoint from',
                      ckpt.model_checkpoint_path)
            saver.restore(session, ckpt.model_checkpoint_path)

            isess = InteractiveSession(session, model, config)

            query = ''
            while query != 'exit':
                query = input('>> ')
                print('>> ' + isess.reply(query))


class InteractiveSession():
    def __init__(self, session, model, config):
        self.context = [[' ']]
        self.u = None
        self.r = None
        self.model = model
        self.session = session
        self.config = config
        self.idx2candid = self.model.idx2candid
        self.w2idx = self.model.w2idx

    def reply(self, msg):
        line = msg.strip().lower()
        if line == 'clear':
            self.context = []
            replp_msg = 'memory cleared!'
        else:
            inputs = []
            questions = []

            q = tokenize(line, char=2)
            q_vector = [self.w2idx[w] for w in q]
            inp_vector = [[self.w2idx[w] for w in s] for s in self.context]

            inputs.append(inp_vector)
            questions.append(np.vstack(q_vector).astype(np.float32))

            input_lens, sen_lens, max_sen_len = dmn_data_utils.get_sentence_lens(
                inputs)

            q_lens = dmn_data_utils.get_lens(questions)
            # max_q_len = np.max(q_lens)

            # max_input_len = min(np.max(input_lens),
            #                     self.config.max_allowed_inputs)

            max_input_len = self.model.max_input_len
            max_sen_len = self.model.max_sen_len
            max_q_len = self.model.max_q_len

            inputs = dmn_data_utils.pad_inputs(inputs, input_lens, max_input_len,
                                               "split_sentences", sen_lens, max_sen_len)

            # inputs = [inputs[0] for _ in range(self.config.batch_size)]
            inputs = np.asarray(inputs)

            questions = dmn_data_utils.pad_inputs(
                questions, q_lens, max_q_len)
            # questions = [questions[0] for _ in range(self.config.batch_size)]
            questions = np.asarray(questions)

            preds = self.model.predict(self.session,
                                       inputs, input_lens, max_sen_len, questions, q_lens)
            preds = preds[0].tolist()
            print(preds)
            r = self.idx2candid[preds[0]]
            reply_msg = r
            r = translator.en2cn(r)
            r = tokenize(r, char=2)
            self.context.append(r)

        return reply_msg


def update_embedding(sess, ckpt, up_embedding):
    saver = tf.train.import_meta_graph(ckpt)
    # We can now access the default graph where all our metadata has been loaded
    graph = tf.get_default_graph()
    embeddings = graph.get_tensor_by_name('embedding/embeddings:0')
    embeddings = embeddings.eval(session=sess)
    # print(type(embeddings))
    for k, v in up_embedding.items():
        embeddings[k] = v

    return embeddings


if __name__ == '__main__':
    main(sys.argv[1:])
    # update_embedding('/home/ecovacs/work/memory_py/model/dmn/ckpt2/dmn.weights.meta')
