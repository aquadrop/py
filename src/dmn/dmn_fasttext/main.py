import tensorflow as tf
import numpy as np

import time
import pickle
import argparse
import os
import sys
from tqdm import tqdm

import data_helper
from config import Config
from dmn import DMN_PLUS
from dmn_session import DmnSession, DmnInfer

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def prepare_data(config):
    train, valid, max_q_len, max_input_len, max_sen_len, max_mask_len, \
        num_supporting_facts, candidate_size, candid2idx, idx2candid, \
        w2idx, idx2w, vocab_size, sentences = \
        data_helper.load_raw_data(config)

    sentences = list(sentences)
    sentences.sort()
    # print('origin sentences:', sentences[:2])

    # embedding
    sentences_embedding, max_len = data_helper.sentence_embedding(
        config, sentences,  w2idx)

    # debug
    questions, inputs, q_lens, sen_lens, input_lens, input_masks, answers, relevant_labels=train
    with open('debug.txt','a') as f:
        print('inputs ==>\n', inputs[:4], file=f)
        print('questions ==>\n',questions[:4],file=f)
        print('answers ==>\n', answers[:4], file=f)
        print('sentences ==>\n',sentences[:10],file=f)
        print('sentences embedding ==>\n',sentences_embedding,file=f)


    metadata = dict()
    data = dict()
    data['train'] = train
    data['valid'] = valid
    # data['sentences'] = sentences
    metadata['sentences_embedding'] = sentences_embedding
    metadata['max_input_len'] = max_input_len
    metadata['max_q_len'] = max_len
    metadata['max_sen_len'] = max_len
    metadata['max_mask_len'] = max_mask_len
    metadata['num_supporting_facts'] = num_supporting_facts
    metadata['candidate_size'] = candidate_size
    metadata['candid2idx'] = candid2idx
    metadata['idx2candid'] = idx2candid
    metadata['w2idx'] = w2idx
    metadata['idx2w'] = idx2w
    metadata['vocab_size'] = vocab_size

    with open(config.metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    with open(config.data_path, 'wb') as f:
        pickle.dump(data, f)


def parse_args(args):
    parser = argparse.ArgumentParser(description='DMN-PLUS')
    parser.add_argument("-r", "--restore", action='store_true',
                        help="restore previously trained weights")
    parser.add_argument("-s", "--strong_supervision",
                        help="use labelled supporting facts (default=false)")
    parser.add_argument("-l", "--l2_loss", type=float,
                        default=0.001, help="specify l2 loss constant")
    parser.add_argument("-n", "--num_runs", type=int,
                        help="specify the number of model runs")
    parser.add_argument(
        "-p", "--infer", action='store_true', help="predict")
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


def run_epoch(model, config, session, data, metadata, num_epoch=0, train_writer=None,
              train_op=None, verbose=2, train=False, display=False):
    dp = config.dropout
    if train_op is None:
        train_op = tf.no_op()
        dp = 1

    total_steps = len(data[0]) // config.batch_size

    total_loss = []
    accuracy = 0
    error = []

    # shuffle data
    # p = np.random.permutation(len(data[0]))
    qp, ip, ql, sl, il, im, a, r = data
    # qp, ip, ql, sl, il, im, a, r = qp[p], ip[p], ql[p], sl[p], il[p], im[p], a[p], r[p]
    # ql, sl, il, im, a, r = ql[p], sl[p], il[p], im[p], a[p], r[p]

    if not display:
        for step in tqdm(range(total_steps)):
            # index = range(step * config.batch_size,
            #               (step + 1) * config.batch_size)
            b = step * config.batch_size
            e = (step + 1) * config.batch_size
            batch_size_data = qp[b:e], ip[b:e], ql[b:
                                                   e], sl[b:e], il[b:e], im[b:e], a[b:e], r[b:e]
            qp_vc, ip_vc, ql_vc, il_vc, im_vc, a_vc, r_vc = data_helper.vectorize_data(
                batch_size_data, config, metadata)
            # print(qp[:2])
            feed = {model.question_placeholder: qp_vc,
                    model.input_placeholder: ip_vc,
                    model.question_len_placeholder: ql_vc,
                    model.input_len_placeholder: il_vc,
                    model.answer_placeholder: a_vc,
                    model.rel_label_placeholder: r_vc,
                    model.dropout_placeholder: dp}

            _ = session.run([train_op], feed_dict=feed)
        return display

    for step in range(total_steps):
        # index = range(step * config.batch_size,
        #               (step + 1) * config.batch_size)
        b = step * config.batch_size
        e = (step + 1) * config.batch_size
        batch_size_data = qp[b:e], ip[b:e], ql[b:
                                               e], sl[b:e], il[b:e], im[b:e], a[b:e], r[b:e]
        qp_vc, ip_vc, ql_vc, il_vc, im_vc, a_vc, r_vc = data_helper.vectorize_data(
            batch_size_data, config, metadata)
        feed = {model.question_placeholder: qp_vc,
                model.input_placeholder: ip_vc,
                model.question_len_placeholder: ql_vc,
                model.input_len_placeholder: il_vc,
                model.answer_placeholder: a_vc,
                model.rel_label_placeholder: r_vc,
                model.dropout_placeholder: dp}

        loss, pred, output, _ = session.run(
            [model.calculate_loss, model.pred, model.output, train_op], feed_dict=feed)

        # if train_writer is not None:
        #     train_writer.add_summary(
        #         summary, num_epoch * total_steps + step)

        answers = a[step *
                    config.batch_size:(step + 1) * config.batch_size]
        questions = qp[step *
                       config.batch_size:(step + 1) * config.batch_size]

        # print(pred)

        if config.multi_label:
            pass
            # multi target
            # correct = 0
            # pred = pred.indices.tolist()
            #
            # for i in range(config.batch_size):
            #     predicts = pred[i]
            #     labels = answers[i]
            #     labels = [idx for idx, i in enumerate(
            #         labels) if int(i) == 1]
            #     while len(predicts) > len(labels):
            #         predicts.pop()
            #     if set(predicts) == set(labels):
            #         correct += 1
            #     else:
            #         Q = ''.join([self.idx2w.get(idx, '')
            #                      for idx in questions[i].astype(np.int32).tolist()])
            #         Q = Q.replace('unk', '')
            #         labels.sort()
            #         predicts.sort()
            #         A = ','.join([self.idx2candid[a] for a in labels])
            #         P = ','.join([self.idx2candid[p] for p in predicts])
            #         error.append((Q, A, P))
            # accuracy += correct / float(len(answers))
        else:
            # print(pred)
            accuracy += np.sum(pred == answers) / float(len(answers))

            # for Q, A, P in zip(questions, answers, pred):
            #     # print(A, P)
            #     if A != P:
            #         Q = ''.join([self.idx2w.get(idx, '')
            #                      for idx in Q.astype(np.int32).tolist()])
            #         Q = Q.replace('unk', '')
            #         A = self.idx2candid[A]
            #         P = self.idx2candid[P]
            #         error.append((Q, A, P))

        total_loss.append(loss)
        if verbose and step % verbose == 0:
            sys.stdout.write('\r{} / {} : loss = {}'.format(
                step, total_steps, np.mean(total_loss)))
            sys.stdout.flush()

    if verbose:
        sys.stdout.write('\r')

    return np.mean(total_loss), accuracy / float(total_steps), error


def load_data(config):
    """Loads metadata and data"""
    with open(config.metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    if config.train_mode:
        with open(config.data_path, 'rb') as f:
            data = pickle.load(f)
        train = data['train']
        valid = data['valid']
    else:
        train = None
        valid = None

    return train, valid, metadata


def train(config, restore=False):
    train_data, valid_data, metadata = load_data(config)
    # sentences_embedding = metadata['sentences_embedding']

    model = DMN_PLUS(config, metadata)

    print('==> Training DMN-PLUS start\n')

    print('==> initializing variables')
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as session:
        session.run(init)

        best_train_epoch = 0
        best_train_loss = float('inf')
        best_train_accuracy = 0
        prev_epoch_loss = float('inf')

        if restore:
            print('==> restoring weights')
            saver.restore(session, config.ckpt_path + 'dmn.weights')
        else:
            if not config.word:
                word_embedding = np.random.uniform(
                    -config.embedding_init,
                    config.embedding_init,
                    (config.vocab_size, config.embed_size))
                session.run(model.embedding_init, feed_dict={
                            model.embedding_placeholder: word_embedding})
        print('==> starting training')
        for epoch in range(config.max_epochs):
            if not (epoch % config.interval_epochs == 0 and epoch > 1):
                print('Epoch {}'.format(epoch))
                _ = run_epoch(model, config, session, train_data, metadata, epoch,
                              train_op=model.train_step, train=True)
            else:
                print('Epoch {}'.format(epoch))
                start = time.time()
                train_loss, train_accuracy, train_error = run_epoch(model, config,
                                                                    session, train_data, metadata, epoch,
                                                                    train_op=model.train_step, train=True, display=True)
                valid_loss, valid_accuracy, valid_error = run_epoch(model, config,
                                                                    session, valid_data, metadata, display=True)
                if train_accuracy > 0.90:
                    for e in train_error:
                        print(e)

                print('Training loss: {}'.format(train_loss))
                print('Validation loss: {}'.format(valid_loss))
                print('Training accuracy: {}'.format(train_accuracy))
                print('Vaildation accuracy: {}'.format(valid_accuracy))

                if train_accuracy > best_train_accuracy:
                    print('Saving weights and updating best_train_loss:{} -> {},\
                            best_train_accuracy:{} -> {}'.format(best_train_loss, train_loss,
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
                if train_loss > prev_epoch_loss * config.anneal_threshold:
                    config.lr /= config.anneal_by
                    print('annealed lr to %f' % config.lr)

                prev_epoch_loss = train_loss

                # if epoch - best_val_epoch > config.early_stopping:
                #     break
                print('Total time: {}'.format(time.time() - start))
        print('==> training stop')


def inference(config):
    print('Inference start')
    di = DmnInfer()
    sess = di.get_session()

    query = ''
    while query != 'exit':
        query = input('>> ')
        print(sess.reply(query))


def main(args):
    args = parse_args(args)
    config = Config()

    if args['prep_data']:
        print('\n>> Preparing Data\n')
        begin = time.clock()
        prepare_data(config)
        end = time.clock()
        print('>> Preparing Data Time:{}'.format(end - begin))
        sys.exit()
    elif args['train']:
        train(config, args['restore'])
    elif args['inference']:
        inference(config)
    else:
        print('ERROR:Unknow Mode')


if __name__ == '__main__':
    main(sys.argv[1:])
