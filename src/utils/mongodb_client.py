#!/usr/bin/env python3
import os,sys
import traceback
from pymongo import MongoClient

class Mongo():
    def __init__(self, db_name, ip='127.0.0.1', port=27017):
        self.db_name = db_name
        self.db = MongoClient(ip, port)[db_name]

    def insert(self, data, collection='template'):
        try:
            self.db[collection].insert(data)
            return 1
        except:
            traceback.print_exc()
            return 0

    def delete(self, collation={}, collection='template'):
        try:
            self.db[collection].delete_many(collation)
            return 1
        except:
            traceback.print_exc()
            return 0



    def search(self, query={}, field={}, collection='template'):
        try:

            data = [x for x in self.db[collection].find(query, field)]
            return data
        except:
            traceback.print_exc()
            return None


    def update(self, data, filter={},  collection = 'template'):
        try:
            self.db[collection].update_many(filter, {'$set': data})
            return 1
        except:
            traceback.print_exc()
            return 0

if __name__ == '__main__':

    mongo = Mongo(ip='10.89.100.12', db_name='bookstore')


    # data = [{'type':'location', 'context':'XXXX[X]XX'},
    #         {'type':'qa', 'context':'YYY[Y]YYY'}]
    # if not mongo.delete():
    #     print('delete error!')
    # if not mongo.insert(data):
    #     print('insert data error!!!')

    data = mongo.search(query={}, field={'instruct':1, 'replies': 1, '_id' : 0}, collection = 'training')

    for x in data:
        print(x)
    
