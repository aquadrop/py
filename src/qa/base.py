import requests
import json
import numpy as np
import schedule, time

from lru import LRU
from threading import Thread

'''
import schedule
import time

def job():
    print("I'm working...")

schedule.every(10).minutes.do(job)
schedule.every().hour.do(job)
schedule.every().day.at("10:30").do(job)
schedule.every(5).to(10).minutes.do(job)
schedule.every().monday.do(job)
schedule.every().wednesday.at("13:15").do(job)

while True:
    schedule.run_pending()
    time.sleep(1)
'''

class BaseKernel:
    appid = "841e6cd456e05713213f413e8765648e"
    user_ids = np.array(['0112DBCD5299791D5A53287D27F4E18A5',
                         '0480704B8A3471FF360DD22AB5C3D9F8E',
                         '09B78AFBFCF3F97F34F12F945769FBD8B'])
    def __init__(self):
        # if not self.user_ids or self.user_ids.size == 0:
        #     self.uid = self.register()
        # else:
        self.uid = self.user_ids[1]#np.random.choice(self.user_ids, 1)[0]
        self.cache = LRU(300)
        # thread = Thread(target=self.schedule)
        # thread.start()

    def kernel(self, q):
        # if q in self.cache:
        #     return self.cache[q]
        answer = self.chat(q)
        # self.cache[q] = answer
        return answer

    def register(self):
        register_data = {"cmd": "register", "appid": self.appid}
        url = "http://idc.emotibot.com/api/ApiKey/openapi.php"
        r = requests.post(url, params=register_data)
        response = json.dumps(r.json(), ensure_ascii=False)
        jsondata = json.loads(response)
        datas = jsondata.get('data')
        for data in datas:
            return data.get('value')

    def chat(self, q):
        try:
            register_data = {"cmd": "chat", "appid": self.appid, "userid": self.uid, "text": q,
                             "location": "南京"}
            url = "http://idc.emotibot.com/api/ApiKey/openapi.php"
            r = requests.post(url, params=register_data)
            response = json.dumps(r.json(), ensure_ascii=False)
            jsondata = json.loads(response)
            datas = jsondata.get("data")
            for data in datas:
                response = data.get('value')
                if response:
                    break
            return response
        except Exception:
            return 'base kernel is detached'

    def clear_cache(self):
        print('clear')
        self.cache.clear()

    def schedule(self):
        schedule.every().hour.do(self.clear_cache)
        while True:
            schedule.run_pending()
            time.sleep(60)

if __name__ == '__main__':
    bk = BaseKernel()
    # bk.register()
    print(bk.kernel(u'吴中万达有6楼吗'))