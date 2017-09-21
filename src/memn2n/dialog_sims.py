"""
Created on september 16, 2017

a rule-based user simulator

UserIntent : [<greet>, <chat>, <buy>, <ask>, <deny>, <provide>, <accept>, <other>]
BotAction : [<api_call_greet_search>, <api_call_emotibot>, <api_call_list_search>, <api_call_info_request>, <api_call_user_accept>]

template:
user : <greet>      bot : <api_call_greet_search>
user : <chat>       bot : <api_call_emotibot>
user : <buy>        bot : <api_call_info_request>
user : <provide>    bot : <api_call_info_request>
user : <provide>    bot : <api_call_list_search>
user : <deny>       bot : <api_call_list_search>
user : <accept>     bot : <api_call_user_accept>
user : <chat>       bot : <api_call_emotibot>
"""


import random
import copy

categories = ['phone', 'airconditioner', 'computer']
greet_set = ['哇哦', '嗨', '你好']
chat_set = ['你叫什么名字', '你是谁', '你是傻逼么', '啥都不懂，还聊啥',
            '安利听过吗', '凭你的智慧，我很难跟你解释', '准确的说，我是一个演员', '还踢球？！', '少林功夫好耶，实在好']
deny_set = ['不要这个', '太贵', '太便宜，凸显不了我的气质', '不好看', '还有别的吗', '不喜欢']
accept_set = ['这个还不错', '就要这个了', '不必再多说，请把钱拿走', '可以的老哥', '这个很强，我喜欢']
buy_set = {'phone': ['买个手机', '手机卖吗', '都有啥手机'], 'airconditioner': [
    '买个空调', '空调卖吗', '都有啥空调'], 'computer': ['买个电脑', '电脑卖吗', '都有啥电脑']}
ask_set = {'phone': ['手机都有哪些啊', '手机牌子有哪些'], 'airconditioner': [
    '空调都有哪些啊', '空调牌子有哪些'], 'computer': ['电脑都有哪些啊', '电脑牌子有哪些']}
provide_set = {'phone': ['苹果的吧', '贵一点的，装逼用的', '便宜点的'], 'airconditioner': [
    '格力的', '四十平米左右', '稍微贵一点的，不差钱'], 'computer': ['联想', '捡贵的来', '显卡牛逼的', '内存最大的']}

candidates_set = {'<api_call_greet_search>': '你好，有什么可以帮您的', '<api_call_emotibot>': '调用竹间机器人',
                  '<api_call_info_request>': '询问想要购买物品的价格，品牌等',
                  '<api_call_list_search>': '提供顾客询问物品列表信息',
                  '<api_call_user_accept>': '很高兴您能买到满意的商品'}


def DialogSims(f):
    dialog = list()

    greet = random.choice(greet_set)
    greet_answer = candidates_set['<api_call_greet_search>']
    dialog.append(greet + '\t' + greet_answer)

    chats = copy.deepcopy(chat_set)
    for _ in range(2):
        chat = random.choice(chats)
        chat_answer = candidates_set['<api_call_emotibot>']
        dialog.append(chat + '\t' + chat_answer)
        chats.remove(chat)

    category = random.choice(categories)
    buy = random.choice(buy_set[category])
    buy_answer = candidates_set['<api_call_info_request>']
    dialog.append(buy + '\t' + buy_answer)

    provides = copy.deepcopy(provide_set[category])
    # print(provides)
    provide = random.choice(provides)
    provide_answer = candidates_set['<api_call_info_request>']
    dialog.append(provide + '\t' + provide_answer)

    provides.remove(provide)
    provide2 = random.choice(provides)
    provide2_answer = candidates_set['<api_call_list_search>']
    dialog.append(provide2 + '\t' + provide2_answer)

    deny = random.choice(deny_set)
    deny_answer = candidates_set['<api_call_list_search>']
    dialog.append(deny + '\t' + deny_answer)

    accept = random.choice(accept_set)
    accept_answer = candidates_set['<api_call_user_accept>']
    dialog.append(accept + '\t' + accept_answer)

    chat = random.choice(chats)
    chat_answer = candidates_set['<api_call_emotibot>']
    dialog.append(chat + '\t' + chat_answer)

    f.write('\n'.join(dialog) + '\n')
    f.write('\n')


if __name__ == '__main__':
    with open('data/dialogs_train.txt', 'w') as f:
        for _ in range(100):
            DialogSims(f)
