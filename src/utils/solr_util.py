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


def solr_facet(mappers, facet_field, is_range, prefix='facet_'):

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
        params = {
            'q': '*:*',
            'facet': True,
            'facet.field': prefix + facet_field,
            "facet.mincount": 1
        }
        params['fq'] = compose_fq(mappers)
        try:
            res = solr.query('category', params)
        except:
            return solr_facet(mappers, facet_field, is_range, prefix='')
        facets = res.get_facet_keys_as_list(prefix + facet_field)
        return facets, len(facets)
    else:
        start = 1
        gap = 1
        end = 100
        # use facet.range
        if facet_field == "price":
            start = 100
            gap = 3000
            end = 30000
        if facet_field == 'ac.power_float':
            start = 1
            gap = 0.5
            end = 10
        params = {
            'q': '*:*',
            'facet': True,
            'facet.range': facet_field,
            "facet.mincount": 1,
            'facet.range.start': start,
            'facet.range.end': end,
            'facet.range.gap': gap
        }
        params['fq'] = compose_fq(mappers)
        res = solr.query('category', params)
        ranges = res.get_facets_ranges()[facet_field].keys()
        ranges = [float("{0:.1f}".format(float(r))) for r in ranges]
        # now render the result
        facet = render_range(ranges, gap)
        return facet, len(ranges)


if __name__ == "__main__":
    mapper = {"category":"a", "brand":"b"}
    options = ["price"]
    print(compose_fq(mapper, options))

    mappers = {'brand':"苹果","category":"手机"}
    facet_field = 'location'
    print(solr_facet(mappers=mappers, facet_field=facet_field, is_range=False))
