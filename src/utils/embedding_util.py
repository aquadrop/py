import requests
from gensim.models.wrappers import FastText

FASTTEXT_URL = 'http://localhost:11425/fasttext/w2v?q='
FASTTEXT_URL_M = 'http://localhost:11425/fasttext/maxsim?q1={0}&q2={1}'
print('load model')
model = FastText.load_fasttext_format('/opt/fasttext/model/test.bin')

def ff_embedding(word):
    ff_url = FASTTEXT_URL + word
    r = requests.get(url=ff_url)
    # print(r)
    vector = r.json()['vector']
    return vector

def mlt_ff_embedding(q1, q2):
    ff_url = FASTTEXT_URL_M.format(q1, q2)
    r = requests.get(url=ff_url).json()
    # print(r)
    sim = r['maxcossim']
    simq = r['simstring']
    return float(sim), simq.replace(',', '')

def ff_embedding_local(word):
    return model[word]

if __name__ == '__main__':
    query = ''
    while query != 'exit':
        query = input('>> ')
        print(ff_embedding_local(query.strip()))
