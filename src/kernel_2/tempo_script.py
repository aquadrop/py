import sys,os
import hashlib

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
grandfatherdir = os.path.dirname(parentdir)
sys.path.append(grandfatherdir)
sys.path.append(parentdir)
print (grandfatherdir)
from utils.mongodb_client import Mongo

mongdb = Mongo(ip='10.89.100.12', db_name='bookstore')


def _load_pre_filter():
    pre_replacement = dict()
    result_list = mongdb.search(field={'given': 1, '_id': 0, 'matched': 1},
                               collection='synonym')
    for each_dict in result_list:
        pre_replacement[each_dict['given']] = each_dict['matched']

    # matched_list = mongdb.search(query={},
    #                        field = {'matched':'1'},
    #                        collection = 'synonym',
    #                        key = 'matched')
    return  pre_replacement
    # pre_replacement = dict(zip(given_list, matched_list))
print(_load_pre_filter())


def _load_media_render():
    media_render_mapper = dict()
    result_list = mongdb.search( field = {'img':1, 'instruct':1, '_id': 0},
                              collection = 'render_media')

    for each_dict in result_list:
        if not each_dict['img']:
            media_render_mapper[each_dict['instruct']] = hashlib.sha256(each_dict['instruct'].encode('utf-8')).hexdigest()
        else:
            media_render_mapper[each_dict['instruct']] = each_dict['img']
    return media_render_mapper

mapper_media = _load_media_render()
print (mapper_media)


def _load_major_render():
    major_render_mapper = dict()

    result_list = mongdb.search(field = {'replies':1, '_id': 0, 'instruct': 1},
                                collection = 'render_api')
    for each_dict in result_list:
        print(type(each_dict['replies']))
        major_render_mapper[each_dict['instruct']] = each_dict['replies']
    return major_render_mapper
mapper_major = _load_major_render()
print (mapper_major)


def _load_location_render():
    location_templates = []
    location_applicables = dict()
    location_precludes = dict()
    result_list = mongdb.search(field={'_id': 0, 'suitable': 1, 'unsuitable': 1, 'template': 1 },
                                       collection='render_location')
    print(result_list)


    index = 0
    for each_dict in result_list:
        location_templates.append(each_dict['template'])
        location_applicables[index] = each_dict['suitable'].split(',')
        location_precludes[index] = each_dict['unsuitable'].split(',')
        index += 1
    print(location_templates)
    print(location_applicables)
    print(location_precludes)
_load_location_render()









