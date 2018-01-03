#!/usr/bin/env python3
import os, sys
import traceback
from pymongo import MongoClient

_defalut_attrs = {
        'scene':'_default',
        'name':'旺宝',
        'gender':'女生',
        'age':'3',
        'birthday':'2月14日',
        'birthplace':'苏州科沃斯',
        'constellation':'水瓶座',
        'height':'1.1米',
        'weight':'20公斤',
        'father':'工程师',
        'mother':'工程师',
        'boyfriend':'null',
        'girlfriend':'null',
        'brother':['地宝'],
        'sister':['窗宝', '地宝'],
        'where':'苏州',
        'hobby':['聊天'],
        'skill':['聊天'],
        }

class Machine_portrait():
    def __init__(self, ip='127.0.0.1', port=27017, db_name='machine_portrait'):
        self.db = MongoClient(ip, port)[db_name]
        self.collection = self.db.data

    def create_default(self, attrs):
        try:
            self.collection.delete_many(attrs)
            self.collection.insert(attrs)
            return 1
        except:
            traceback.print_exc()
            return 0
    def create(self, scene):
        if scene == '_default':
            return 0
        try:
            if self.collection.find_one({'scene':scene}):
                return 1
            attrs = self.collection.find_one({'scene':'_default'}, {'_id':0})
            attrs['scene'] = scene
            self.collection.insert(attrs)
            return 1
        except:
            traceback.print_exc()
            return 0

    def delete(self, scene):
        if scene == '_default':
            return 0
        try:
            self.collection.delete_one({'scene':scene})
            return 1
        except:
            traceback.print_exc()
            return 0

    def update(self, scene, attrs):
        try:
            self.collection.update_one({'scene':scene}, {'$set':attrs})
            return 1
        except:
            traceback.print_exc()
            return 0

    def search(self, scene, field={'_id':0}):
        try:
            data = self.collection.find_one({'scene':scene}, field)
            return data
        except:
            traceback.print_exc()
            return None


if __name__ == '__main__':
    mp = Machine_portrait()
    #mp.create_default(_defalut_attrs)
    #mp.create('test')
    #mp.update('test', {'age':1})
    #print(mp.search('test'))
    print(dir(mp.collection))

