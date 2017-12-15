import sys
sys.path.append("..")
from amq.IMQ import IMessageQueue
import json
import time
import amq.random_helper as random_helper

class BenebotSim():

    mq = None #messagequeue

    def __init__(self):
        publish_key = 'nlp.sim.normal.request.'+random_helper.random_string()
        receive_key = publish_key.replace('request', 'reply')
        self.mq = IMessageQueue(receive_key, publish_key, receive_key, receive_key, '')

    def getSim(self, sequence, sentences, unk=False):
        if not sequence or not sentences:
            return {}
        #st = time.time()
        result = self.mq.request_synchronize(json.dumps({'sequence': sequence, 'sentences': sentences, 'unk': unk}))
        #print('time: ', time.time()-st)
        if result:
            return json.loads(result)
        return {}

if __name__ == '__main__':
    bt = BenebotSim()
    while 1:
        s1 = input('input: ')
        s2 = input('input: ')
        result = bt.getSim(s1, s2, True)
        print(result)
