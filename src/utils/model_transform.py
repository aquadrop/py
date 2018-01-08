import os
import sys
import re

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
grandfatherdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)
print(grandfatherdir)
print(parentdir)
from utils.mongodb_client import Mongo


class Model_trans(Mongo):

    def __init__(self, fname, model_category = 'location', db_name= 'bookstore', ip= 'bookstore', port= '27017' ):

        super(Model_trans,self).__init__(db_name, ip, port)
        self.f_path = os.path.join(grandfatherdir,'model/render_2/{}'.format(fname))
        self.model_category = model_category


    def trans(self):
        '''Tansform the model files in dir(renser_2) to data format for mangdb.
            model_category: make the model category optional, such as media.  
            f_path： the abspath of given model file.
        '''
        try:
            if self.model_category == 'location':
                with open(self.f_path, 'r') as f_in:
                    data_list = []
                    value_list = []
                    key_list = ['template', 'suitable', 'unsuitable']
                    for eachline in f_in.readlines():
                        value_list = eachline.strip().split('#')
                        key_value = tuple(zip(key_list, value_list))
                        data_list.append(dict(key_value))
                # return data_list
            elif self.model_category == 'media':
                with open(self.f_path, 'r') as f_in:
                    data_list = []
                    value_list = []
                    key_list = ['instruct', 'img']
                    for eachline in f_in.readlines():
                        value_list = eachline.strip().split('#')
                        key_value = tuple(zip(key_list, value_list))
                        data_list.append(dict(key_value))
            elif self.model_category == 'api':
                with open(self.f_path, 'r') as f_in:
                    data_list = []
                    value_list = []
                    replies_list = []
                    key_list = ['instruct', 'replies']
                    for eachline in f_in.readlines():
                        sentence_list = re.split('#{2}|\|', eachline.strip())

                        # the value to the key 'replies' will be stored in array format
                        # just take the ending part
                        for eachanswer in sentence_list[2:]:
                            replies_list.append(eachanswer)
                        value_list.append(sentence_list[0])
                        value_list.append(replies_list)
                        # print(value_list)
                        key_value = tuple(zip(key_list, value_list))
                        data_list.append(dict(key_value))
                        replies_list = [] #empty the list !!!
                        value_list = []

            elif self.model_category == 'synonym':
                with open(self.f_path, 'r') as f_in:
                    data_list = []
                    value_list = []
                    key_list = ['given', 'matched']

                    for eachline in f_in.readlines():
                        value_list = eachline.strip().split('#')
                        key_value = tuple(zip(key_list, value_list))
                        data_list.append(dict(key_value))

            else:
                raise Exception('No such model documents provided.'
                                'Please reselect and try again.')
            return data_list

        except Exception as e:
            print(e)



if __name__ == '__main__':
    fname = 'synonym.txt'
    db_name = 'bookstore' # set datebase name
    ip = '10.89.100.12'
    port = 27017

    model_category = 'synonym'
    collection = 'synonym' # set collection branch name

    mt = Model_trans(fname = fname, db_name = db_name ,ip = ip ,port = port , model_category = model_category)
    mt.delete(collation={}, collection = collection)
    data = mt.trans()
    mt.insert(data = data, collection = collection)


