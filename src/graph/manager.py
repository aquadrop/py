"""
-------------------------------------------------
   File Name：     manager
   Description :
   Author :       deep
   date：          18-1-15
-------------------------------------------------
   Change Activity:
                   18-1-15:
                   
   __author__ = 'deep'

   http://10.90.26.222:8181/NLP/nlp_proposals/blob/master/Automata%E5%AF%B9%E8%AF%9D%E6%A1%86%E6%9E%B6.md
-------------------------------------------------
"""
import json
import traceback

from graph.automata import Automata
from graph.nlu import NLU
from qa.iqa import Qa as QA

class Drive():

    BASE_MACHINES = [] # some base machines including 'story, weather, set clock alerts...'
    WAIT_ENTRY = 'wait_entry' # waitin' for common_nlu to provide intent associated with store_id
    IN_QUEST = 'in_quest' # in the progress of automata quest

    def __init__(self):
        self.registered_entry_intents = {} # map store_id to store_ids
        self.registered_store_ids = set()
        self.registered_store_ids = set(self.BASE_MACHINES)
        self.machines = {}
        self.common_intents = set()
        self.common_nlu = None
        self.qa = QA('base')
        self.activated_machine = None

    def register_common_nlu(self, config):
        self.common_nlu = NLU(config)
        self.common_intents = self.common_nlu.get_intents()

    def drive(self, q, store_ids=[]):
        if not self.activated_machine:
            return self.entry_drive(q, store_ids)
        else:
            return self.automata_drive(q, store_ids)

    def automata_drive(self, q, store_ids):
        """

        :param q:
        :param store_ids: for traceback to entry_drive
        :return:
        """
        reply = self.activated_machine.drive(q)
        if not reply['matched']:
            self.activated_machine = None
            return self.entry_drive(q, store_ids)
        return reply['instruction']

    def entry_drive(self, q, store_ids):
        intent = self.common_nlu.process(q, state_interpreter='regex')
        if len(store_ids) > 0:
            for store_id in store_ids:
                if store_id in self.registered_store_ids \
                        and intent in self.registered_entry_intents[store_id]:
                    self.activated_machine = self.machines[store_id]
                    self.activated_machine.reset()
                    return self.machines[store_id].drive(q)

            for store_id in self.BASE_MACHINES:
                if store_id in self.registered_store_ids \
                        and intent in self.registered_entry_intents[store_id]:
                    self.activated_machine = self.machines[store_id]
                    self.activated_machine.reset()
                    return self.machines[store_id].drive(q)

        else:
            for store_id in self.BASE_MACHINES:
                if store_id in self.registered_store_ids \
                        and intent in self.registered_entry_intents[store_id]:
                    self.activated_machine = self.machines[store_id]
                    self.activated_machine.reset()
                    return self.machines[store_id].drive(q)
        response = self.qa.get_responses(q)[1]
        return response

    def register_machine(self, machine):
        self.machines[machine.store_id] = machine
        self.registered_entry_intents[machine.store_id] = machine.get_open_intents()
        self.registered_store_ids.add(machine.store_id)

        for input_ in machine.get_open_intents():
            regex = machine.nlu.regex_interpreter_helper[input_]
            self.common_nlu.regex_interpreter_helper[input_] = regex

    def destroy_machine(self, store_id):
        try:
            store_id_destroy = self.machines[store_id]
            del self.machines[store_id]
            del self.registered_entry_intents[store_id_destroy]
        except:
            traceback.print_exc()

def load_json(file_path):
    with open(file_path, 'r') as f:
        config = json.load(f)
    return config

if __name__ == '__main__':
    manager = Drive()
    common_nlu_config = load_json('common_nlu.json')
    manager.register_common_nlu(common_nlu_config)

    bookstore_machine_config = load_json('bookstore_config.json')
    bookstore_machine = Automata(bookstore_machine_config)
    wrd_machine_config = load_json('wrd_config.json')
    wrd_machine = Automata(wrd_machine_config)

    manager.register_machine(bookstore_machine)
    manager.register_machine(wrd_machine)

    store_ids = ['bookstore']

    while True:
        input_ = input('input:')
        try:
            print(manager.drive(input_, store_ids=store_ids))
        except:
            traceback.print_exc()