import numpy as np
import sys
# reload(sys)
# sys.setdefaultencoding('utf8')
sys.path.insert(0, '../utils')
import os
import requests

#get parent dir path: memory_py
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)

from SolrClient import SolrClient

from utils.query_util import tokenize
from utils.solr_util import solr_qa
from utils.embedding_util import ff_embedding, mlt_ff_embedding
from qa.base import BaseKernel

THRESHOLD = 0.95
REACH = 1

class Qa:
    def __init__(self, core, question_key='question', answer_key='answers', solr_addr = 'http://localhost:11403/solr'):
        self.core = core
        self.question_key = question_key #指代solr数据库doc里的key——‘question’
        self.answer_key = answer_key #指代solr数据库doc里的key——‘answer’
        self.base = BaseKernel()
        self.solr = SolrClient(solr_addr)

    def get_responses(self, query, user='solr'):
        '''
            程序功能：传入问句query
            return  solr数据库中最大相似度的问句、最大相似度的回答以及最大相似度
        '''
        docs = solr_qa(self.core, query, solr=self.solr, field=self.question_key)
        print(docs)
        best_query = None
        best_answer = None
        best_score = -1

        #参数index：所有相似问句的数目
        #参数doc：单个相似的问句
        for index, doc in enumerate(docs):
            if index > 10:
                break
            b = doc[self.answer_key]
            g = doc[self.question_key] # solr库中相似的问句
            # for _g in g:
            #     score = self.similarity(query, _g)
            #     if score > best_score:
            #         best_score = score
            #         best_query = _g
            #         best_answer = b
            #         if score >= REACH:
            #             break
            score, _g = self.m_similarity(query, g)
            if score > best_score:
                best_score = score
                best_query = _g
                best_answer = b
            # if score >= REACH:
            #     break
            # print(score)

        if best_score < THRESHOLD:
            print('redirecting to third party', best_score)
            answer = '\n您好!您可以输入以下常见问题进行咨询：\n*科沃斯旺宝产品介绍。\n*如何购买科沃斯旺宝？\n*' \
                     '科沃斯旺宝可以在哪些行业中应用？\n*科沃斯旺宝有哪些使用实例？\n*科沃斯可以为用户和合作' \
                     '伙伴提供哪些服务？\n\n请在下方对话框中提交您的问题，小科将竭尽全力为您解答哟~'
            return query, answer, best_score
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

    def m_similarity(self, query1, m_query2):
        tokens1 = ','.join(tokenize(query1, 3))
        tokens2 = '@@'.join([','.join(tokenize(t, 3)) for t in m_query2])
        score, _g = mlt_ff_embedding(tokens1, tokens2)

        return score, _g


def ceshi():
    query1 = '我的名字是小明'
    query2 = '要买抽油烟机'
    qa = Qa('interactive')
    print(qa.similarity(query1, query2))


def main():
    qa = Qa('zx_weixin_qa')
    best_query, best_answer, best_score = qa.get_responses('科沃')
    print(best_query, best_answer, best_score)


if __name__ == '__main__':
    main()
    # ceshi()