import os
import sys
import re

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
grandfatherdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)
print(parentdir)
# from utils.mongodb_client import Mongo

from mongodb_client import Mongo
class Model_trans(Mongo):
    '''
        Tansform the model files in dir(renser_2) to data format for mangdb.
        model_category: make the model category optional, such as media.  
        f_path： the abspath of given model file.
    '''
    def __init__(self, fname, model_category = 'location', db_name= 'bookstore', ip= 'bookstore', port= '27017' ):

        super(Model_trans,self).__init__(db_name, ip, port)
        self.f_path = os.path.join(grandfatherdir,'model/render_2/{}'.format(fname))
        self.model_category = model_category


    def trans(self):

        try:
            if self.model_category == 'render_location':
                with open(self.f_path, 'r') as f_in:
                    dataList = []
                    valueList = []
                    keyList = ['template', 'suitable', 'unsuitable']
                    for eachline in f_in.readlines():
                        valueList = eachline.strip().split('#')
                        key_value = tuple(zip(keyList, valueList))
                        dataList.append(dict(key_value))

            elif self.model_category == 'render_api':
                dataList = []
                keyList = ['media', 'equal_questions', 'answers', 'emotion_url', 'group',
                            'label', 'emotion_name', 'time_out']
                f_media = open(os.path.join(grandfatherdir, 'model/render_2/render_media.txt'), 'r')
                media = {}
                for line in f_media.readlines():
                    sentence = line.strip()
                    if sentence: media[sentence.split('#')[0]] = sentence.split('#')[-1]
                # print('media', media)

                f_api = open(self.f_path, 'r')
                dict_num = 0
                for line in f_api.readlines():
                    sentence = re.split(r'#{2}|\|', line.strip())
                    # print (sentence)
                    if line:
                        dataList.append([])
                        for key in keyList:
                            if key == 'media': dataList[dict_num].append((key, media.get(sentence[0], 'null')))
                            elif key == 'equal_questions': dataList[dict_num].append((key, [i for i in sentence[:1]]))
                            elif key == 'answers': dataList[dict_num].append((key, [i for i in sentence[-1:]]))
                            elif key == 'group': dataList[dict_num].append((key, 'render_api'))
                            elif key == 'label': dataList[dict_num].append((key, sentence[0]))
                            else: dataList[dict_num].append((key, 'null'))
                        # print(dataList)
                        dataList[dict_num] = dict(dataList[dict_num])
                        dict_num += 1
                f_media.close()
                f_api.close()

               
            elif self.model_category == 'synonym':
                with open(self.f_path, 'r') as f_in:
                    dataList = []
                    valueList = []
                    keyList = ['given', 'matched']

                    for eachline in f_in.readlines():
                        valueList = eachline.strip().split('#')
                        key_value = tuple(zip(keyList, valueList))
                        dataList.append(dict(key_value))

            else:
                raise Exception('No such model documents provided.'
                                'Please reselect and try again.')
            return dataList

        except Exception as e:
            print(e)



if __name__ == '__main__':
    fname = 'render_api.txt'
    db_name = 'bookstore' # set datebase name
    ip = '10.89.100.12'
    port = 27017

    model_category = 'render_api'
    collection = 'render_api' # set collection branch name

    mt = Model_trans(fname = fname, db_name = db_name ,ip = ip ,port = port , model_category = model_category)
    mt.delete(collation={}, collection = collection)
    data = mt.trans()
    mt.insert(data = data, collection = collection)


