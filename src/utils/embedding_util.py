import requests

FASTTEXT_URL = 'http://localhost:11425/fasttext/w2v?q='
FASTTEXT_URL_M = 'http://localhost:11425/fasttext/maxsim?q1={0}&q2={1}'

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
