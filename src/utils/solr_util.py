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


def query(mappers):
    params = {
        'q': '*:*',
        'q.op': "OR"
    }
    fill = []
    for key, value in mappers.items():
        fill.append(key + ":" + str(value))
    params['fq'] = " AND ".join(fill)
    res = solr.query('category', params)
    docs = res.docs
    return docs


def solr_qa(core, query):
    params = {'q': query, 'q.op': 'or'}
    responses = solr.query(core, params)
    docs = responses.docs
    return docs
