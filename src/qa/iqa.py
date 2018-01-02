import numpy as np
import sys
import os
import requests
import schedule, time
import pylru

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)

from utils.query_util import tokenize
from utils.solr_util import solr_qa
from utils.embedding_util import ff_embedding, mlt_ff_embedding
from qa.base import BaseKernel

from lru import LRU
from threading import Thread

from amq.sim import BenebotSim
from dmn.dmn_fasttext.vector_helper import computeSentenceSim
CACHE_SIZE = 50 #2017/12/26 设置缓存大小

class Qa:

    static_bt = None
    cache = pylru.lrucache(CACHE_SIZE)
    THRESHOLD = 0.90

    def __init__(self, core, question_key='question', answer_key='answer'):
        self.core = core
        self.question_key = question_key
        self.answer_key = answer_key
        self.base = BaseKernel()
        self.REACH = 1

        # if Qa.static_bt:
        #     self.bt = Qa.static_bt
        # else:
        #     self.bt = BenebotSim()
        #     Qa.static_bt = self.bt

        # self.schedule()
        thread = Thread(target=self.schedule)
        thread.start()
        # thread.join()
        # self.solr_addr = solr_addr

    def get_responses(self, query, user='solr', cls=''):
        if query in self.cache:
            best_query = self.cache[query]['query']
            best_answer = np.random.choice(self.cache[query]['answer'])
            best_score = self.cache[query]['score']
            best_doc = self.cache[query]['doc']
            return best_query, best_answer, best_score, best_doc
        docs = solr_qa(self.core, query, field=self.question_key + '_str', cls=cls)
        if len(docs) == 0:
            docs = solr_qa(self.core, query, field=self.question_key, cls=cls)
        else:
            doc = np.random.choice(docs)
            best_query = doc[self.question_key]
            best_answer = doc[self.answer_key]
            best_score = 1
            best_doc = doc
            if 'uid' not in best_doc:
                best_doc['uid'] = 'uid_not_defined'
            cached = {"query": best_query, "answer": best_answer, "score": best_score, "doc": best_doc}
            self.cache[query] = cached
            return best_query, np.random.choice(best_answer), best_score, best_doc

        # docs = solr_qa(self.core, query, self.question_key)
        # print(docs)
        best_query = None
        best_answer = None
        best_score = -1
        best_doc = {'uid': "third_party"}
        for index, doc in enumerate(docs):
            if index > 10:
                break
            b = doc[self.answer_key]
            g = doc[self.question_key]
            # for _g in g:
            #     score = self.similarity(query, _g)
            #     if score > best_score:
            #         best_score = score
            #         best_query = _g
            #         best_answer = b
            #         if score >= REACH:
            #             break
            # score, _g = self.m_similarity(query, g)
            score, _g = self.w2v_local_similarity(query, g)
            if score > best_score:
                best_score = score
                best_query = _g
                best_answer = b
                best_doc = doc
                if 'uid' not in best_doc:
                    best_doc['uid'] = 'uid_not_defined'
            # if score >= REACH:
            #     break
            # print(score)

        if best_score < self.THRESHOLD:
            print('redirecting to third party', best_score)
            # answer = self.base.kernel(query)
            answer = 'null'
            cached = {"query": query, "answer": [answer], "score": best_score, "doc":best_doc}
            # self.cache[query] = cached
            return query, answer, best_score, best_doc
            # return query, 'api_call_base', best_score
        else:
            cached = {"query": best_query, "answer": best_answer, "score": best_score, "doc": best_doc}
            self.cache[query] = cached
            return best_query, np.random.choice(best_answer), best_score, best_doc

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

    def m_similarity(self, query1, m_query2):
        tokens1 = ','.join(tokenize(query1, 3))
        tokens2 = '@@'.join([','.join(tokenize(t, 3)) for t in m_query2])
        score, _g = mlt_ff_embedding(tokens1, tokens2)

        return score, _g

    def w2v_local_similarity(self, query1, query2):
        tokens1 = tokenize(query1, 3)
        tokens2 = [tokenize(t, 3) for t in query2]

        max_sim = -10
        _g = query2[0]
        for words2 in tokens2:
            sim = computeSentenceSim(tokens1, words2)
            if sim > max_sim:
                max_sim = sim
                _g = words2
        return float(max_sim), _g

    def clear_cache(self):
        self.cache.clear()

    def schedule(self):
        schedule.every().day.at("01:00").do(self.clear_cache)
        while True:
            schedule.run_pending()
            time.sleep(60)

def test():
    query1 = '我的名字是小明'
    query2 = '要买抽油烟机'
    qa = Qa('interactive')
    print(qa.similarity(query1, query2))


def main():
    qa = Qa('base')
    best_query, best_answer, best_score = qa.get_responses('你叫什么名字')
    print(best_query, best_answer, best_score)


if __name__ == '__main__':
    main()
