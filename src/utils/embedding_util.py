import requests
from gensim.models.wrappers import FastText

FASTTEXT_URL = 'http://localhost:11425/fasttext/w2v?q='
FASTTEXT_URL_M = 'http://localhost:11425/fasttext/maxsim?q1={0}&q2={1}'
FASTTEXT_URL_M_POST = 'http://localhost:11425/fasttext/maxsim'
# print('load model')
# model = FastText.load_fasttext_format('/opt/fasttext/model/test.bin')

def ff_embedding(word):
    ff_url = FASTTEXT_URL + word
    r = requests.get(url=ff_url)
    # print(r)
    vector = r.json()['vector']
    return vector

def mlt_ff_embedding(q1, q2):
    # print('q1:',q1)
    # print('q2:',q2)
    # ff_url = FASTTEXT_URL_M.format(q1, q2)
    # print(requests.get(url=ff_url))
    r = requests.post(url=FASTTEXT_URL_M_POST, data={"q1":q1, "q2":q2}).json()
    # print('json:',r)
    sim = r['maxcossim']
    simq = r['simstring']
    return float(sim), simq.replace(',', '')

# def ff_embedding_local(word):
#     return model[word]

if __name__ == '__main__':
    # query = ''
    # while query != 'exit':
    #     query = input('>> ')
    #     print(ff_embedding_local(query.strip()))

    q1='我能,用,支付宝,付款,吗'
    q2="xxxxx"
    print(mlt_ff_embedding(q1,q2))