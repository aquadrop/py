""" 
Memory Session
"""

import pickle as pkl
import tensorflow as tf

import os
import sys

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
grandfatherdir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parentdir)
sys.path.append(grandfatherdir)

import utils.query_util as query_util
import memory.memn2n as memn2n
import memory.data_utils as data_utils


class Memn2nSession():
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


class MemInfer:
    def __init__(self, config):
        self.metadata_dir = config['metadata_dir']
        self.data_dir = config['data_dir']
        self.ckpt_dir = config['ckpt_dir']
        self.model = self._load_model()

    def _load_model(self):

        with open(self.metadata_dir, 'rb') as f:
            metadata = pkl.load(f)
        with open(self.data_dir, 'rb') as f:
            data_ = pkl.load(f)

        # read content of data and metadata
        candidates = data_['candidates']
        candid2idx, self.idx2candid = metadata['candid2idx'], metadata['idx2candid']

        # get train/test/val data
        train, test, val = data_['train'], data_['test'], data_['val']

        # gather more information from metadata
        sentence_size = metadata['sentence_size']
        self.w2idx = metadata['w2idx']
        idx2w = metadata['idx2w']
        self.memory_size = metadata['memory_size']
        vocab_size = metadata['vocab_size']
        self.n_cand = metadata['n_cand']
        candidate_sentence_size = metadata['candidate_sentence_size']

        # vectorize candidates
        candidates_vec = data_utils.vectorize_candidates(
            candidates, self.w2idx, candidate_sentence_size)
        HOPS = 3
        print('---- memory config ----')
        print('memory_size:', self.memory_size)
        print('vocab_size:', vocab_size)
        print('candidate_size:', self.n_cand)
        print('candidate_sentence_size:', candidate_sentence_size)
        print('hops:', HOPS)
        print('---- end ----')

        model = memn2n.MemN2NDialog(
            batch_size=16,
            vocab_size=vocab_size,
            candidates_size=self.n_cand,
            sentence_size=sentence_size,
            embedding_size=300,
            candidates_vec=candidates_vec,
            hops=HOPS
        )

        ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print('\n>> restoring checkpoint from', ckpt.model_checkpoint_path)
            model.saver.restore(model._sess, ckpt.model_checkpoint_path)

        return model

    def get_session(self):
        sess = Memn2nSession(
            self.model, self.idx2candid, self.w2idx, self.n_cand, self.memory_size)
        return sess

    def mem_infer(self, sess, query):
        return sess.reply(query)


def main():
    metadata_dir = os.path.join(
        grandfatherdir, 'data/memn2n/processed/metadata.pkl')
    data_dir = os.path.join(
        grandfatherdir, 'data/memn2n/processed/data.pkl')
    ckpt_dir = os.path.join(grandfatherdir, 'model/memn2n/ckpt')
    config = {"metadata_dir": metadata_dir,
              "data_dir": data_dir, "ckpt_dir": ckpt_dir}
    mi = MemInfer(config)
    sess = mi.get_session()

    reply = sess.reply("手机")
    print(reply)


if __name__ == '__main__':
    main()
