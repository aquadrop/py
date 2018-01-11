"""
-------------------------------------------------
   File Name：     nlu
   Description :
   Author :       deep
   date：          18-1-11
-------------------------------------------------
   Change Activity:
                   18-1-11:
                   
   __author__ = 'deep'
-------------------------------------------------
"""

class NLU:
    def __init__(self, mode=['keyword']):
        self.mode = mode

    def process(self, q, state=None):
        pass

    def keyword(self, q):
        pass

    def regex(self, q):
        pass

    def ml(self, q):
        pass