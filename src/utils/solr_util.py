"""
Solr Util
"""
import traceback
import pickle
import re
import json

import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
grandfatherdir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parentdir)
sys.path.append(grandfatherdir)

dir_path = os.path.dirname(os.path.realpath(__file__))

from SolrClient import SolrClient

import sys

solr = SolrClient('http://localhost:11403/solr')

def compose_fq(mapper, option_fields=['price']):
    option_mapper = dict()
    for of in option_fields:
        if of in mapper:
            option_mapper[of] = mapper[of]
    must_mapper = dict()
    for key, value in mapper.items():
        if key not in option_mapper:
            must_mapper[key] = mapper[key]
    fq = '*:*'
    if len(must_mapper) > 0:
        must = " AND ".join(["{}:{}".format(key, value) for key, value in must_mapper.items()])
        fq = must
    if len(option_mapper) > 0:
        fq = fq + ' OR ' + " OR ".join(["{}:{}".format(key, value) for key, value in option_mapper.items()])

    return fq

def query(must_mappers, option_mapper=None):
    params = {
        'q': '*:*',
        'q.op': "OR"
    }
    fill = []
    for key, value in must_mappers.items():
        fill.append(key + ":" + str(value))
    params['fq'] = " AND ".join(fill)
    options = []
    if option_mapper:
        for key, value in option_mapper.items():
            options.append(key + ":" + str(value))
        params['fq'] += ' OR ' + " OR ".join(options)
    res = solr.query('category', params)
    docs = res.docs
    return docs


def solr_qa(core, query):
    params = {'q': query, 'q.op': 'or'}
    responses = solr.query(core, params)
    docs = responses.docs
    return docs


if __name__ == "__main__":
    mapper = {"category":"a", "brand":"b"}
    options = ["price"]
    print(compose_fq(mapper, options))
