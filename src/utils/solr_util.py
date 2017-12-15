
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

# solr = SolrClient('http://localhost:11403/solr')
solr = SolrClient('http://10.89.100.14:8999/solr')

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
        fq = "*:* AND " + must
    if len(option_mapper) > 0:
        fq = fq + ' OR ' + " OR ".join(["{}:{}".format(key, value) for key, value in option_mapper.items()])

    return fq

def query(must_mappers, option_mapper={}):
    params = {
        'q': '*:*',
        'q.op': "OR"
    }
    mappers = {}
    option_fields = []
    for k, v in must_mappers.items():
        mappers[k] = v
    for k, v in option_mapper.items():
        mappers[k] = v
        option_fields.append(k)

    params['fq'] = compose_fq(mappers, option_fields)
    res = solr.query('category', params)
    docs = res.docs
    return docs


def solr_qa(core, query, solr=solr, field=None):
    if not field:
        params = {'q': query, 'q.op': 'or', 'rows':20}
    else:
        params = {'q': "{}:{}".format(field, query), 'q.op': 'or', 'rows': 20}
    responses = solr.query(core, params)
    docs = responses.docs
    return docs

def solr_max_value(params, target_field):
    params['sort'] = target_field + ' desc'
    params['rows'] = 1

    res = solr.query('category', params)
    docs = res.docs
    max_val = 0
    if len(docs) > 0:
        max_val = docs[0][target_field]
    return max_val

def solr_min_value(params, target_field):
    params['sort'] = target_field + ' asc'
    params['rows'] = 1

    res = solr.query('category', params)
    docs = res.docs
    min_val = 0
    if len(docs) > 0:
        min_val = docs[0][target_field]
    return min_val

def solr_facet(mappers, facet_field, is_range, prefix='facet_', postfix='_str', core='category'):

    def render_range(a, gap):
        if len(a) == 0:
            return []
        components = []
        p = 0
        if len(a) == 1:
            components.append(str(a))
        elif len(a) == 2:
            components.append(str(a[0]))
            components.append(str(a[1]))
        else:
            for i in range(1, len(a)):
                if a[i] - a[i - 1] > gap:
                    if i - p == 1:
                        components.append(str(a[p]))
                    else:
                        components.append(str(a[p]) + "-" + str(a[i - 1]))
                    p = i
            if len(components) == 0:
                components.append(str(a[1]) + "-" + str(a[-1]))
        render = []
        if len(components) == 1:
            render.append(components[0])
        else:
            for i in range(0, len(components) - 1):
                if '-' in components[i + 1]:
                    render.append(components[i])
                    continue
                if '-' in components[i]:
                    render.append(components[i])
                    continue
                render.append(components[i] + "-" + components[i + 1])
        return render

    if not is_range:
        facet_field = prefix + facet_field + postfix
        params = {
            'q': '*:*',
            'facet': True,
            'facet.field': facet_field,
            "facet.mincount": 1
        }
        params['fq'] = compose_fq(mappers)
        res = solr.query(core, params)
        facets = res.get_facet_keys_as_list(facet_field)
        docs = res.docs
        return facets, len(facets), docs
    else:
        fq = compose_fq(mappers)
        minmax_params = {
            'q': '*:*',
            'fq': fq
        }
        max_value = solr_max_value(minmax_params, facet_field)
        min_value = solr_min_value(minmax_params, facet_field)
        start = min_value
        gap = max((max_value - min_value) / 5 * 0.5, 0.5)
        end = max_value + 10
        params = {
            'q': '*:*',
            'facet': True,
            'facet.range': facet_field,
            "facet.mincount": 1,
            'facet.range.start': start,
            'facet.range.end': end,
            'facet.range.gap': gap,
            'fq': fq
        }
        # use facet.range
        # if facet_field == "price":
        #     start = 100
        #     gap = 3000
        #     end = 30000
        # if facet_field == 'ac.power_float':
        #     start = 1
        #     gap = 0.5
        #     end = 10
        res = solr.query(core, params)
        docs =  res.docs
        ranges = res.get_facets_ranges()[facet_field].keys()
        ranges = [float("{0:.1f}".format(float(r))) for r in ranges]
        # now render the result
        facet = render_range(ranges, gap)
        return facet, len(ranges), docs


if __name__ == "__main__":
    mapper = {"category":"a", "brand":"b"}
    options = ["price"]
    print(compose_fq(mapper, options))

    mappers = {'brand':"美的","category":"空调"}
    facet_field = 'ac.power_float'
    print(solr_facet(mappers=mappers, facet_field=facet_field, is_range=True))