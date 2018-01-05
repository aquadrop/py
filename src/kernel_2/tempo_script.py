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
    given_list = mongdb.search(query={},
                                 field={'given': '1'},
                                 collection='synonym',
                                 key='given')
    matched_list = mongdb.search(query={},
                           field = {'matched':'1'},
                           collection = 'synonym',
                           key = 'matched')
    pre_replacement = dict(zip(given_list, matched_list))
    print(pre_replacement)

_load_pre_filter()


def _load_media_render():
    media_render_mapper = dict()
    img_list = mongdb.search(query={},
                              field = {'img':'1'},
                              collection = 'render_media',
                              key = 'img')
    instruct_list = mongdb.search(query={},
                                  field = {'instruct':'1'},
                                  collection = 'render_media',
                                  key = 'instruct')
    for index in range(len(img_list)):

        if not img_list[index]:
            media_render_mapper[instruct_list[index]] = hashlib.sha256(instruct_list[0].encode('utf-8')).hexdigest()
        else:
            media_render_mapper[instruct_list[index]] = img_list[index]
    return media_render_mapper
mapper = _load_media_render()
print (mapper)


def _load_major_render():
    major_render_mapper = dict()

    replies_list = mongdb.search(query={},
                              field = {'replies':'1'},
                              collection = 'render_api',
                              key = 'replies')
    instruct_list = mongdb.search(query={},
                              field = {'instruct':'1'},
                              collection = 'render_api',
                              key = 'instruct')
    major_render_mapper = dict(list(zip(instruct_list, replies_list)))
    return major_render_mapper
mapper = _load_major_render()
print (mapper)













