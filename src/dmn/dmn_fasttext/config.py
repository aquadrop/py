import os
import numpy as np


class Config(object):
    """Holds model hyperparams and data information."""

    batch_size = 512
    embed_size = 300
    hidden_size = 300

    max_epochs = 3456
    interval_epochs = 5
    early_stopping = 20

    dropout = 0.9
    lr = 0.001
    l2 = 0.001

    cap_grads = True
    max_grad_val = 10
    noisy_grads = True

    word = True
    embedding_init = np.sqrt(3)

    # set to zero with strong supervision to only train gates
    strong_supervision = False
    beta = 1

    # NOTE not currently used hence non-sensical anneal_threshold
    anneal_threshold = 1000
    anneal_by = 1

    num_hops = 2
    num_attention_features = 4

    max_allowed_inputs = 130
    total_num = 3000000

    floatX = np.float32

    multi_label = False
    top_k = 5
    max_memory_size = 3
    max_sen_len=10
    fix_vocab = True

    train_mode = True

    vocab_size = 7464

    split_sentences = True

    EMPTY = 'EMPTY'
    PAD = 'PAD'
    NONE = ''
    UNK = 'UNK'

    # paths
    prefix = grandfatherdir = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))

    vocab_path = os.path.join(prefix, 'data/char_table/dmn_vocab.txt')

    DATA_DIR = os.path.join(prefix, 'data/memn2n/train/tree/origin/')
    CANDID_PATH = os.path.join(
        prefix, 'data/memn2n/train/tree/origin/candidates.txt')

    MULTI_DATA_DIR = os.path.join(prefix, 'data/memn2n/train/multi_tree')
    MULTI_CANDID_PATH = os.path.join(
        prefix, 'data/memn2n/train/multi_tree/candidates.txt')

    data_dir = MULTI_DATA_DIR if multi_label else DATA_DIR
    candid_path = MULTI_CANDID_PATH if multi_label else CANDID_PATH

    metadata_path = os.path.join(
        prefix, 'model/dmn/dmn_processed/metadata_word.pkl')
    data_path = os.path.join(prefix, 'model/dmn/dmn_processed/data_word.pkl')
    ckpt_path = os.path.join(prefix, 'model/dmn/ckpt_word/')

    multi_metadata_path = os.path.join(
        prefix, 'model/dmn/dmn_processed/multi_metadata.pkl')
    multi_data_path = os.path.join(
        prefix, 'model/dmn/dmn_processed/multi_data.pkl')
    multi_ckpt_path = os.path.join(prefix, 'model/dmn/ckpt-ff-trainable/')

    metadata_path = multi_metadata_path if multi_label else metadata_path
    data_path = multi_data_path if multi_label else data_path
    ckpt_path = multi_ckpt_path if multi_label else ckpt_path


def main():
    config = Config()


if __name__ == '__main__':
    main()
