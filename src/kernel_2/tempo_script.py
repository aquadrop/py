import sys,os
import hashlib

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
grandfatherdir = os.path.dirname(parentdir)
sys.path.append(grandfatherdir)
sys.path.append(parentdir)
print (grandfatherdir)
from utils.mongodb_client import Mongo

mongdb = Mongo(ip='10.89.100.12', db_name='bookstore')


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


            # def _load_major_render():
#     major_render_mapper = dict()
#
#     replies_list = mongdb.search(query={},
#                                   field = {'replies':'1'},
#                                   collection = 'render_api',
#                                   key = 'replies')
#     instruct_list = mongdb.search(query={},
#                                   field = {'instruct':'1'},
#                                   collection = 'render_api',
#                                   key = 'instruct')
#     major_render_mapper = dict(list(zip(instruct_list, replies_list)))
#     return major_render_mapper
# mapper = _load_major_render()
# print (mapper)


        # for line in f:
        #     line = line.strip('\n')
        #     key, replies = line.split('|')
        #     key = key.split('##')[0]
        #     replies = replies.split('/')
        #     filtered = []
        #     for r in replies:
        #         if r:
        #             filtered.append(r)
        #     self.major_render_mapper[key] = filtered


# location_templates = []
# location_applicables = dict()
# location_precludes = dict()
# suitable_list = mongdb.search(query = {},
#                               field = {'suitable':'1'},
#                               collection = 'render_location',
#                               key = 'suitable')
# unsuitable_list = mongdb.search(query = {},
#                                 field = {'unsuitable':'1'},
#                                 collection = 'render_location',
#                                 key = 'unsuitable')
# template_list = mongdb.search(query = {},
#                               field = {'template':'1'},
#                               collection = 'render_location',
#                               key = 'template')
#
# index = 0
# for index in range(len(template_list)):
#     location_applicables[index] = suitable_list[index].split(',')
#     location_precludes[index] = unsuitable_list[index].split(',')
#
# print(location_applicables)
# print(location_precludes)









