import sys
import os
import gensim
import numpy as np

from dmn.dmn_fasttext.config import Config

config = Config()
loop_word = 0
dim_shrink = 1
sentence_vector_dict = {}

path = '/opt/word2vec/benebot_vector/word2vec.bin'


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(
                Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class BenebotVector(metaclass=Singleton):

    model = None

    def __init__(self, model_path):
        if not self.model:
            print('load word vector model')
            self.model = gensim.models.Word2Vec.load(model_path)
        # self.dim = len(self.getVectorByWord('中国'))
        self.dim = self.model.vector_size

    def hasWord(self, word):
        return self.model.__contains__(word.strip())

    def getVectorByWord(self, word):
        result = []
        if self.model.__contains__(word.strip()):
            vector = self.model.__getitem__(word.strip())
            result = [v for v in vector]
        return result

    def getSimilarWords(self, word):
        result = self.model.most_similar(word.strip())
        return result

    def calWordSimilarity(self, word1, word2):
        result = self.model.similarity(word1.strip(), word2.strip())
        return result

    def getVectorBySentence(self, words):
        vectors = []
        for word in words:
            vector = self.getVectorByWord(word)
            if not vector:
                if len(word) == 1:
                    vector = [0.0] * self.dim
                else:
                    vector = self.getVectorBySentence(word)
            vectors.append(vector)
        result = [0.0] * self.dim
        if vectors:
            result = np.mean(vectors, axis=0)
        return result

    def getVectorByWeightSentence(self, weight_sentence):
        vectors = []
        value_sum = 0.0
        for key, value in weight_sentence.items():
            vector = self.getVectorByWord(key)
            if not vector:
                vector = np.array([0.0] * self.dim)
            vector = np.array(vector) * value
            vectors.append(vector)
            value_sum += value
        result = np.array([0.0] * self.dim)
        if vectors:
            result = np.sum(vectors, axis=0) / value_sum
        return result


bv = BenebotVector(path)


def getVector(word, embedding_dim=300):
    if word == config.PAD:
        return np.array([0.0] * embedding_dim)
    vector = bv.getVectorByWord(word)
    # print(vector)
    if vector:
        return vector
    else:
        words = [w for w in word]
        return bv.getVectorBySentence(words)

def computeSentenceSim(sent1, sent2):
    u = bv.getVectorBySentence(sent1)
    v = bv.getVectorBySentence(sent2)

    c = np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)
    return c


def embedding_lookup(sequence_num, sequence_length, embedding_dim, input_x, maintain=0):
    for sentence_index in range(sequence_num):
        sentence_str = input_x[sentence_index]
        if not (sentence_str in sentence_vector_dict):
            # print(sentence_str)
            sentence = sentence_str.split(" ")
            sentence_length = len(sentence)
            loop_count = loop_word
            if loop_count > sentence_length:
                loop_count = sentence_length
            for word_index in range(sequence_length):
                if word_index >= sentence_length:
                    loop_index = word_index - sentence_length
                    if loop_index < loop_count:
                        word_vector_tmp = np.array(
                            [getVector(sentence[loop_index], embedding_dim) * dim_shrink])
                    else:
                        word_vector_tmp = np.array([[0.0] * embedding_dim])
                else:
                    word_vector_tmp = np.array(
                        [getVector(sentence[word_index], embedding_dim) * dim_shrink])
                if word_index == 0:
                    word_vector = word_vector_tmp
                else:
                    word_vector = np.concatenate(
                        [word_vector, word_vector_tmp], 0)
            sentence_vector_tmp = np.array([word_vector])
            sentence_vector_dict[sentence_str] = sentence_vector_tmp
        sentence_vector_tmp = sentence_vector_dict.get(sentence_str)
        if maintain == 0:
            sentence_vector_dict.clear()
        if sentence_index == 0:
            sentence_vector = sentence_vector_tmp
        else:
            sentence_vector = np.concatenate(
                [sentence_vector, sentence_vector_tmp], 0)
    embedded_chars = sentence_vector.astype(np.float32)
    return embedded_chars


def main():
    while True:
        # ipt = input("input:")
        # vec = bv.getSimilarWords(ipt)
        # print(len(vec), vec)
        c = computeSentenceSim("科沃斯,机器人".split(","), "我,想,把,这里,的,书,借出去,可以,吗".split(","))
        print(c)


if __name__ == '__main__':
    main()
