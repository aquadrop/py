"""
Created on september 21, 2017

a rule-based dialog simulator

UserIntent: [greet, chat, qa, {buy: [ask, deny, provide, accept]}, bye]
BotAction: [api_call_greet, api_call_chat, api_call_qa, api_call_buy, api_call_bye]

template:
user : <greet>      bot : <api_call_greet>
user : <chat>       bot : <api_call_chat>
user : <qa>         bot : <api_call_qa>
user : <buy>        bot : <api_call_buy>
user : <bye>        bot : <api_call_bye>

"""

import os
import numpy as np
from collections import OrderedDict


class DialogSimulator:
    """
    dialog simulator
    """

    mapper = {'greet': 'api_call_greet', 'chat': 'api_call_chat',
              'qa': 'api_call_qa', 'buy': 'api_call_buy', 'bye': 'api_call_bye'}

    apicallMapper = {'api_call_greet': '您好，请问有什么可以帮助您的？', 'api_call_chat': 'apicall embotibot',
                     'api_call_qa': 'api_call qa', 'api_call_bye': '再见，谢谢光临！'}

    def __init__(self, userIntentFiles, dialogsFiles, category):
        self.data = self.loadData(userIntentFiles)

    def loadData(self, userIntentFiles):
        greetFile = userIntentFiles['greet']
        chatFile = userIntentFiles['chat']
        qaFile = userIntentFiles['qa']
        byeFile = userIntentFiles['bye']
        buyFile = userIntentFiles['buy']

        files = [greetFile, chatFile, qaFile, byeFile, buyFile]
        data = OrderedDict()
        for file in files:
            name, _ = os.path.splitext(os.path.basename(file))
            with open(file, 'r') as f:
                lines = f.readlines()
                data[name] = [l.strip() for l in lines]

        return data

    def genDialog(self, dialogsFiles, category):
        keys = self.data.keys()
        commons = key[:-1]
        buy = keys[-1]
        commonData = dict()
        for key in commons:
            commonData[key] = self.data[key]
        buyData = self.data[buy]

        commonDialogs = self.genCommonDialog(commonData)
        buyDialogs = self.genBuyDialog(buyData, category)
        dialogs = self.mergeDialog(commonDialogs, buyDialogs)
        self.writeDialog(dialogsFiles, dialogs)

    def genCommonDialog(self, data):
        dialogs = OrderedDict()
        for k, v in data.items():
            response = apicallMapper[mapper[k]]  # response can be responses
            pairs = [[query] + [response] for query in v]
            dialogs[k] = pairs

        return dialogs

    def genBuyDialog(self, data, category):

    def writeDialog(self, paths, data):
        pass


def main():
    pass


if __name__ == '__main__':
    main()
