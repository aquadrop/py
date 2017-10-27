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


def prepare_data(args, config):
    train, valid, word_embedding, max_q_len, max_input_len, max_sen_len, \
        num_supporting_facts, vocab_size, candidate_size, candid2idx, \
        idx2candid, w2idx, idx2w = dmn_data_utils.load_data(
            config, split_sentences=True)

    metadata = dict()
    data=dict()
    data['train'] = train
    data['valid'] = valid
    metadata['word_embedding'] = word_embedding
    metadata['max_q_len'] = max_q_len
    metadata['max_input_len'] = max_input_len
    metadata['max_sen_len'] = max_sen_len
    metadata['num_supporting_facts'] = num_supporting_facts
    metadata['vocab_size'] = vocab_size
    metadata['candidate_size'] = candidate_size
    metadata['candid2idx'] = candid2idx
    metadata['idx2candid'] = idx2candid
    metadata['w2idx'] = w2idx
    metadata['idx2w'] = idx2w

    with open(config.metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    with open(config.data_path, 'wb') as f:
        pickle.dump(data, f)


def parse_args(args):
    parser = argparse.ArgumentParser(description='DMN-PLUS')
    parser.add_argument("-r", "--restore",action='store_true',
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


def main(args):
    args = parse_args(args)
    # print(args)

    config = Config()

    if args['prep_data']:
        print('\n>> Preparing Data\n')
        prepare_data(args, config)
        sys.exit()


    if args['train']:
        model = DMN_PLUS(config)
        print('Training DMN-PLUS start')

        best_overall_val_loss = float('inf')

        print('==> initializing variables')
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:

            sum_dir = 'summaries/train/' + time.strftime("%Y-%m-%d %H %M")
            if not os.path.exists(sum_dir):
                os.makedirs(sum_dir)
            train_writer = tf.summary.FileWriter(sum_dir, session.graph)

            session.run(init)

            best_val_epoch = 0
            prev_epoch_loss = float('inf')
            best_val_loss = float('inf')
            best_val_accuracy = 0.0

            if args['restore']:
                print('==> restoring weights')
                saver.restore(session, config.ckpt_path+'dmn.weights')

            print('==> starting training')
            for epoch in range(config.max_epochs):
                print('Epoch {}'.format(epoch))
                start = time.time()

                train_loss, train_accuracy, train_error = model.run_epoch(
                    session, model.train, epoch, train_writer,
                    train_op=model.train_step, train=True)
                valid_loss, valid_accuracy, valid_error = model.run_epoch(
                    session, model.valid)
                print('Training error:')
                for e in train_error:
                    print(e)
                # print('Validation error:')
                print('Training loss: {}'.format(train_loss))
                print('Validation loss: {}'.format(valid_loss))
                print('Training accuracy: {}'.format(train_accuracy))
                print('Vaildation accuracy: {}'.format(valid_accuracy))

                if valid_loss < best_val_loss:
                    best_val_loss = valid_loss
                    best_val_epoch = epoch
                    if best_val_loss < best_overall_val_loss:
                        print('Saving weights')
                        best_overall_val_loss = best_val_loss
                        best_val_accuracy = valid_accuracy
                        saver.save(session, config.ckpt_path+'dmn.weights')

                # anneal
                if train_loss > prev_epoch_loss * model.config.anneal_threshold:
                    model.config.lr /= model.config.anneal_by
                    print('annealed lr to %f' % model.config.lr)

                prev_epoch_loss = train_loss

                # if epoch - best_val_epoch > config.early_stopping:
                #     break
                print('Total time: {}'.format(time.time() - start))

            print('Best validation accuracy:', best_val_accuracy)

    else:  # inference
        config.train_mode=False
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
        self.context = [['此', '乃', '空', '文']]
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

            q = tokenize(line)
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
            r = tokenize(r)
            self.context.append(r)

        return reply_msg


if __name__ == '__main__':
    main(sys.argv[1:])
