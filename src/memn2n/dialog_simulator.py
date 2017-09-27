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
from enum import Enum
from collections import OrderedDict

from simulator_utils import mapper, buyQueryMapper


class DialogSimulator:
    """
    dialog simulator
    """

    def __init__(self, userIntentFiles, dialogsFiles, merchandisesFile):
        self.data = self.loadData(userIntentFiles)
        self.merchandises = self.loadMerchandises(merchandisesFile)
        self.dialogsFiles = dialogsFiles
        self.keys = ['greet', 'chat', 'qa', 'bye', 'buy']
        self.candidatesSet = set()

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

    def loadMerchandises(self, merchandisesFile):
        merchandises = list()
        with open(merchandisesFile, 'r') as f:
            lines = f.readlines()
        for line in lines:
            dic = dict()
            [category, brand, price] = line.strip().split('/')
            dic['category'] = category
            dic['brand'] = brand
            dic['price'] = price
            merchandises.append(dic)
        return merchandises

    def genDialog(self):
        commons = self.keys[:-1]
        buy = self.keys[-1]
        commonData = dict()
        for key in commons:
            commonData[key] = self.data[key]
        buyData = self.data[buy]

        commonDialogs = self.genCommonDialog(commonData)
        buyDialogs = self.genBuyDialog(buyData)
        self.mergeDialog(commonDialogs, buyDialogs)

    def genCommonDialog(self, data):
        dialogs = OrderedDict()
        for k, v in data.items():
            # response can be responseS
            response = mapper[k]
            self.candidatesSet.add(response)
            pairs = [[query] + [response] for query in v]
            dialogs[k] = pairs

        return dialogs

    # fix labellllllllllllllllllllll!!!!!!!!!!!!!
    def genBuyDialog(self, data):
        def process(querys):
            dialogs = list()
            answer = 'api_call buy'
            for merchandise in self.merchandises:
                brand = merchandise['brand']
                category = merchandise['category']
                price = merchandise['price']
                mapper = {'<brand>': brand,
                          '<category>': category, '<price>': price}
                dialog = list()
                for query in querys:
                    for key, value in mapper.items():
                        query = query.replace(key, value)
                    dialog.append([query] + [answer])

                dialogs.append(dialog)
                # print(dialog)
            return dialogs

        def fill(dialog, slot):
            dialogs = list()
            for merchandise in self.merchandises:
                brand = '<' + merchandise['brand'] + '>'
                category = '<' + merchandise['category'] + '>'
                price = '<' + merchandise['price'] + '>'
                querySlotMap = {'<brand>': brand,
                                '<category>': category, '<price>': price}

                filledDialog = list()
                fillIntent = ['rhetorical_ask', 'provide', 'deny']
                # print('dialog:', dialog)
                # print('slot:', slot)
                for index, pair in enumerate(dialog):
                    query = pair[0]
                    answer = pair[1]
                    # print('query:', query)
                    # print('answer:', answer)
                    # print('sslot:', slot[index][0])

                    if slot[index][0] in fillIntent:
                        beforeSlots = [slot[i][1] for i in range(
                            index) if slot[i][0] != 'rhetorical_ask']
                        # print('before:', beforeSlots)
                        for sslot in beforeSlots:
                            sslot = ' <' + sslot + '>'
                            answer += sslot
                            # print('answer:', answer)

                    for key, value in querySlotMap.items():
                        query = query.replace(key, value)
                        answer = answer.replace(key, value)
                    query = query.replace(
                        '<', '').replace('>', '')
                    self.candidatesSet.add(answer)
                    filledDialog.append([query] + [answer])
                dialogs.append(filledDialog)
            return dialogs

        buyDialogs = list()
        for template in data:
            lines = template.strip().split('/')
            dialog = list()
            slot = list()
            for line in lines:
                ll = line.split()
                intent = ll[0]
                merchandise = ll[1]
                slot.append((intent, merchandise))

                queryList = buyQueryMapper[intent]['user'][merchandise]
                query = np.random.choice(queryList)
                answer = buyQueryMapper[intent]['bot'][merchandise]
                dialog.append([query] + [answer])

            buyDialogs.extend(fill(dialog, slot))
        # print(buyDialogs)
        return buyDialogs

    def mergeDialog(self, commonDialogs, buyDialogs):
        keys = self.keys
        keys.remove('buy')
        with open(self.dialogsFiles, 'w') as f:
            for buyDialog in buyDialogs:
                dialog = list()
                head = keys[:-1]
                tail = keys[-1]

                for key in head:
                    dialog.append(
                        commonDialogs[key][np.random.choice(len(commonDialogs[key]))])
                dialog.extend(buyDialog)
                dialog.append(
                    commonDialogs[tail][np.random.choice(len(commonDialogs[tail]))])

                for line in dialog:
                    line = '\t'.join(line)
                    f.write(line + '\n')
                f.write('\n')
        with open('data/memn2n/candidates.txt', 'w') as f:
            for candidate in list(self.candidatesSet):
                f.write(candidate + '\n')

    def writeDialog(self, paths, data):
        pass


def main():

    userIntentFiles = {
        'greet': 'data/memn2n/dialog_simulator/greet.txt',
        'chat': 'data/memn2n/dialog_simulator/chat.txt',
        'qa': 'data/memn2n/dialog_simulator/qa.txt',
        'bye': 'data/memn2n/dialog_simulator/bye.txt',
        'buy': 'data/memn2n/dialog_simulator/buy.txt'
    }

    # dialogsFiles suppose to be a list : ['train','valid','test']
    dialogsFiles = 'data/memn2n/train.txt'
    merchandisesFile = 'data/memn2n/dialog_simulator/merchandises.txt'
    ds = DialogSimulator(userIntentFiles, dialogsFiles, merchandisesFile)
    ds.genDialog()


if __name__ == '__main__':
    main()
