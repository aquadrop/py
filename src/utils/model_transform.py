import os,sys

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
grandfatherdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)
print(grandfatherdir)
print(parentdir)
from utils.mongodb_client import Mongo

# class mongdb_trans(Mongo):
#     def __init__(self):
#         self.db_name = 'bookstore'
#         self.ip = '10.89.100.12'
#         self.port = 27017

render_location_file = os.path.join(grandfatherdir, 'model/render_2/render_location.txt')
# print(render_location_file)

def location_transform(infile = render_location_file):
    '''语料模板数据格式转换为类json，为嵌入数据库服务'''
    with open(infile, 'r') as f_in:
        data_list = []
        for eachline in f_in.readlines():
            key_list = ['template', 'suitable', 'unsuitable']
            value_list = eachline.strip().split('#')
            key_value = tuple(zip(key_list, value_list))
            data_list.append(dict(key_value))
    return data_list



if __name__ == '__main__':
    db = Mongo(db_name = 'bookstore',ip = '10.89.100.12',port = 27017)
    data = location_transform()
    collection = 'render_location'
    db.insert(data = data, collection = collection)


