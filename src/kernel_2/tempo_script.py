import sys,os
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
grandfatherdir = os.path.dirname(parentdir)
sys.path.append(grandfatherdir)
sys.path.append(parentdir)
print (grandfatherdir)
from utils.mongodb_client import Mongo

mongdb = Mongo(ip='10.89.100.12', db_name='bookstore')
location_templates = []
location_applicables = dict()
location_precludes = dict()
suitable_list = mongdb.search(query = {},
                              field = {'suitable':'1'},
                              collection = 'render_location',
                              key = 'suitable')
unsuitable_list = mongdb.search(query = {},
                                field = {'unsuitable':'1'},
                                collection = 'render_location',
                                key = 'unsuitable')
template_list = mongdb.search(query = {},
                              field = {'template':'1'},
                              collection = 'render_location',
                              key = 'template')

index = 0
for index in range(len(template_list)):
    location_applicables[index] = suitable_list[index].split(',')
    location_precludes[index] = unsuitable_list[index].split(',')

print(location_applicables)
print(location_precludes)









