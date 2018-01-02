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

    def search(self, query={}, filter={}, collection='template'):
        try:
            data = [x for x in self.db[collection].find(query)]
            return data
        except:
            traceback.print_exc()
            return None


if __name__ == '__main__':
    mongo = Mongo(ip='127.0.0.1', db_name='template')
    data = [{'type':'location', 'context':'XXXX[X]XX'},
            {'type':'qa', 'context':'YYY[Y]YYY'}]
    if not mongo.delete():
        print('delete error!')
    if not mongo.insert(data):
        print('insert data error!!!')
    data = mongo.search(query={'type':'location'})
    for x in data:
        print(x)
    
