import numpy as np
import sys
import os
import requests


parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)

from utils.query_util import tokenize
from utils.solr_util import solr_qa
from utils.embedding_util import ff_embedding
from qa.base import BaseKernel

THRESHOLD = 0.95
REACH = 0.9999

class Qa:
    def __init__(self, core):
        self.core = core
        self.base = BaseKernel()

    def get_responses(self, query, ):
        docs = solr_qa(self.core, query, 'g')
        # print(docs)
        best_query = None
        best_answer = None
        best_score = -1
        for index, doc in enumerate(docs):
            if index > 10:
                break
            if 'g' not in doc or 'b' not in doc:
                continue
            b = doc['b']
            g = doc['g']
            for _g in g:
                score = self.similarity(query, _g)
                if score > best_score:
                    best_score = score
                    best_query = _g
                    best_answer = b
                    if score >= REACH:
                        break
            # print(score)

        if best_score < THRESHOLD:
            print('redirecting to third party', best_score)
            return query, self.base.kernel(query), best_score
            # return query, 'api_call_base', best_score
        else:
            return best_query, np.random.choice(best_answer), best_score

    def embed(self, tokens):
        embeddings = [ff_embedding(word) for word in tokens]
        embeddings = np.asarray(embeddings)
        # print(embeddings.shape)
        embedding = np.mean(embeddings, axis=0)
        # print(embedding)

        return embedding

    def similarity(self, query1, query2):
        def cos(embed1, embed2):
            num = np.dot(embed1, embed2.T)
            denom = np.linalg.norm(embed1) * np.linalg.norm(embed2)
            cos = num / denom
            sin = 0.5 + 0.5 * cos
            return cos

        tokens1 = tokenize(query1, 3)
        tokens2 = tokenize(query2, 3)
        embed1 = self.embed(tokens1)
        embed2 = self.embed(tokens2)

        return cos(embed1, embed2)


def test():
    query1 = '我的名字是小明'
    query2 = '要买抽油烟机'
    qa = Qa('interactive')
    print(qa.similarity(query1, query2))


def main():
    qa = Qa('interactive')
    best_query, best_answer, best_score = qa.get_responses('好的知道了')
    print(best_query, best_answer, best_score)


if __name__ == '__main__':
    main()
