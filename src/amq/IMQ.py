import pika
import sys
import threading
import time
import inspect
import ctypes
import amq.nlp_property

class IMessageQueue():

    is_synchronize = False
    reply_value = None
    t = None

    def __init__(self, tag, publish_key='', queue_name='', receive_key='', callback_func='', exchange_type='topic'):
        self.cond = threading.Condition()

        self.tag = tag
        self.user_name = 'rabbitmq'
        self.user_pwd = 'rabbitmq@0'
        self.ip = amq.nlp_property.NLP_FRAMEWORK_IP
        self.port = 5672

        self.queue_name = queue_name
        self.receive_key = receive_key
        self.publish_key = publish_key
        self.callback_func = callback_func
        self.exchange = 'nlp_'+exchange_type
        self.exchange_type = exchange_type

        crt = pika.PlainCredentials(self.user_name, self.user_pwd)

        self.connection = pika.BlockingConnection(pika.ConnectionParameters(self.ip, self.port, '/', credentials=crt))
        self.chan = self.connection.channel()

        self.chan.exchange_declare(exchange=self.exchange, exchange_type=self.exchange_type)

        if self.receive_key and self.queue_name:
            def target():
                self.chan.basic_qos(prefetch_count=1)
                #result = self.chan.queue_declare(exclusive=True)   
                #self.queue_name = result.method.queue
                self.chan.queue_declare(queue=self.queue_name, auto_delete=True)
                self.chan.queue_bind(exchange=self.exchange,
                           queue=self.queue_name,
                           routing_key=self.receive_key)
                self.chan.basic_consume(self.callback, queue=self.queue_name)
                self.chan.start_consuming()
            self.t = threading.Thread(target=target)
            self.t.start()

        print("[%s] start" %(self.tag))

    def close(self):
        if self.receive_key and self.queue_name:
            self.chan.stop_consuming()
            self.chan.close()
            #print(dir(self.chan))
            self._async_raise(self.t.ident, SystemExit)
        self.connection.close()
        #print(dir(self.connection))

    def __exit__(self):
        #self.connection.close()
        self.close()

    def publish(self, value, routing_key=''):
        self.reply_value = None
        key = self.publish_key
        if routing_key:
            key = routing_key
        #print("[%s] pubs key %s, body %s" %(self.tag, key, value))
        if key:
            self.chan.basic_publish(exchange=self.exchange, 
                    routing_key=key,
                    body=value,
                    properties=pika.BasicProperties(
                        content_type = 'text/plain', message_id = self.tag,
                    ))

    def callback(self, ch, method, properties, body):
        key = method.routing_key
        body = body.decode('utf-8')
        #print("[%s] recv exchange %s, key %s, body %s" %(self.tag, method.exchange, key, body))
        if self.callback_func:
            #print('call ', str(self.callback_func))
            self.callback_func(key, body, self.publish)
        #else:
        #    print('no callback found')
        self.chan.basic_ack(delivery_tag = method.delivery_tag)
        self.reply_value = body
        if self.is_synchronize:
            self.cond.acquire()
            self.cond.notify()
            self.cond.release()

    def request_synchronize(self, value, timeout=1):
        self.is_synchronize = True
        self.reply_value = '' 
        self.cond.acquire()
        self.publish(value)
        self.cond.wait(timeout=timeout)
        self.cond.release()
        self.is_synchronize = False
        return self.reply_value

    def _async_raise(self, tid, exctype):  
        """raises the exception, performs cleanup if needed"""  
        tid = ctypes.c_long(tid)  
        if not inspect.isclass(exctype):  
            exctype = type(exctype)  
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))  
        if res == 0:  
            raise ValueError("invalid thread id")  
        elif res != 1:  
            # """if it returns a number greater than one, you're in trouble,  
            # and you should call it again with exc=NULL to revert the effect"""  
            ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)  
            raise SystemError("PyThreadState_SetAsyncExc failed")
