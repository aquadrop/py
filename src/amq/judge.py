import sys
sys.path.append("..")
import json
from amq.IMQ import IMessageQueue
import amq.random_helper as random_helper

class JUDGE_SENT():

    mq = None
    def __init__(self):
        publish_key = 'nlp.judgesent.request.'+random_helper.random_string()
        receive_key = publish_key.replace('request', 'reply')
        self.mq = IMessageQueue(receive_key, publish_key, receive_key, receive_key, '')

    def get_sentinfo(self,sentence):
        if not sentence:
            return -1
        result = self.mq.request_synchronize(json.dumps(sentence))
        result = json.loads(result)
        return result


if __name__=='__main__':
    js = JUDGE_SENT()
    #bc.get_web_content()
    while 1:
        sentence = input('Enter a sentence:')
        result = js.get_sentinfo(sentence)
        print(result)