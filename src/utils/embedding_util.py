import requests

FASTTEXT_URL = 'http://localhost:11425/fasttext/w2v?q='


def ff_embedding(word):
    ff_url = FASTTEXT_URL + word
    r = requests.get(url=ff_url)
    # print(r)
    vector = r.json()['vector']
    return vector
