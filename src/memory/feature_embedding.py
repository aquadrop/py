
from __future__ import division  # py3 "true division"

import logging
import sys
import os
import heapq
from timeit import default_timer
from copy import deepcopy
from collections import defaultdict
import threading
import itertools
import warnings
import gensim
from gensim.utils import keep_vocab_item, call_on_class_only
from gensim.models.keyedvectors import KeyedVectors, Vocab
from gensim.models import word2vec
try:
    from queue import Queue, Empty
except ImportError:
    from Queue import Queue, Empty

from numpy import exp, log, dot, zeros, outer, random, dtype, float32 as REAL,\
    double, uint32, seterr, array, uint8, vstack, fromstring, sqrt, newaxis,\
    ndarray, empty, sum as np_sum, prod, ones, ascontiguousarray, vstack, logaddexp

from scipy.special import expit

from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from gensim.corpora.dictionary import Dictionary
from six import iteritems, itervalues, string_types
from six.moves import xrange
from types import GeneratorType
from scipy import stats
logger = logging.getLogger(__name__)
from gensim.models.word2vec_inner import train_batch_sg, train_batch_cbow
from gensim.models.word2vec_inner import score_sentence_sg, score_sentence_cbow
from gensim.models.word2vec_inner import FAST_VERSION, MAX_WORDS_IN_BATCH
#from word2vec_inner import train_batch_cbow
class feature_embedding_vocab(word2vec.Word2Vec):
    def __init__(self, sentences=None, size=100, alpha=0.025, window=5, min_count=5,
            max_vocab_size=None, sample=1e-3, seed=1, workers=3, min_alpha=0.0001,
            sg=0, hs=0, negative=5, cbow_mean=1, hashfxn=hash, iter=5, null_word=0,
            trim_rule=None, sorted_vocab=1, batch_words=MAX_WORDS_IN_BATCH):
        self.initialize_word_vectors()
        self.sg = int(sg)
        self.cum_table = None  # for negative sampling
        self.vector_size = int(size)
        self.layer1_size = int(size)
        if size % 4 != 0:
            logger.warning("consider setting layer size to a multiple of 4 for greater performance")
        self.alpha = float(alpha)
        self.min_alpha_yet_reached = float(alpha)  # To warn user if alpha increases
        self.window = int(window)
        self.max_vocab_size = max_vocab_size
        self.seed = seed
        self.random = random.RandomState(seed)
        self.min_count = min_count
        self.sample = sample
        self.workers = int(workers)
        self.min_alpha = float(min_alpha)
        self.hs = hs
        self.negative = negative
        self.cbow_mean = int(cbow_mean)
        self.hashfxn = hashfxn
        self.iter = iter
        self.null_word = null_word
        self.train_count = 0
        self.total_train_time = 0
        self.sorted_vocab = sorted_vocab
        self.batch_words = batch_words
        self.model_trimmed_post_training = False
        self.scan_vocab(sentences, trim_rule=trim_rule)
        #self.train(sentences, total_examples=self.corpus_count, epochs=self.iter,
        #           start_alpha=self.alpha, end_alpha=self.min_alpha)


class feature_embedding_train(word2vec.Word2Vec):
    def __init__(self, sentences=None, size=100, alpha=0.025, window=5, min_count=5,
            max_vocab_size=None, sample=1e-3, seed=1, workers=3, min_alpha=0.0001,
            sg=0, hs=0, negative=5, cbow_mean=1, hashfxn=hash, iter=5, null_word=0,
            trim_rule=None, sorted_vocab=1, batch_words=MAX_WORDS_IN_BATCH):
        self.initialize_word_vectors()
        self.sg = int(sg)
        self.cum_table = None  # for negative sampling
        self.vector_size = int(size)
        self.layer1_size = int(size)
        if size % 4 != 0:
            logger.warning("consider setting layer size to a multiple of 4 for greater performance")
        self.alpha = float(alpha)
        self.min_alpha_yet_reached = float(alpha)  # To warn user if alpha increases
        self.window = int(window)
        self.max_vocab_size = max_vocab_size
        self.seed = seed
        self.random = random.RandomState(seed)
        self.min_count = min_count
        self.sample = sample
        self.workers = int(workers)
        self.min_alpha = float(min_alpha)
        self.hs = hs
        self.negative = negative
        self.cbow_mean = int(cbow_mean)
        self.hashfxn = hashfxn
        self.iter = iter
        self.null_word = null_word
        self.train_count = 0
        self.total_train_time = 0
        self.sorted_vocab = sorted_vocab
        self.batch_words = batch_words
        self.model_trimmed_post_training = False
        self.build_vocab(sentences, trim_rule=trim_rule)
        self.train(sentences, total_examples=self.corpus_count, epochs=self.iter,
                   start_alpha=self.alpha, end_alpha=self.min_alpha)

#    def _do_train_job(self, sentences, alpha, inits):
#        """
#        Train a single batch of sentences. Return 2-tuple `(effective word count after
#        ignoring unknown words and sentence length trimming, total word count)`.
#        """
#        work, neu1 = inits
#        tally = 0
#        if self.sg:
#            tally += train_batch_sg(self, sentences, alpha, work)
#        else:
#            tally += train_batch_cbow(self, sentences, alpha, work, neu1)
#        return tally, self._raw_word_count(sentences)
