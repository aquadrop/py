"""
-------------------------------------------------
   File Name：     automata_driver
   Description :
   Author :       deep
   date：          18-1-11
-------------------------------------------------
   Change Activity:
                   18-1-11:

   __author__ = 'deep'
-------------------------------------------------
"""

import random
import uuid
import re

import json

from transitions.extensions import GraphMachine as Machine
from transitions import State

from graph.policy import Policy
from graph.state_card import StateCard
from graph.nlu import NLU


class Automata(Machine):

    RESET_CODE = 'CLEAR_AUTOMATA'

    def __init__(self, config):
        self.id = str(uuid.uuid4())
        self.store_id = config['store_id']
        self.name = config['name']
        self._load_states(config)
        self.transitions = config['transitions']
        self.policy = Policy(config)
        Machine.__init__(self, model=self.policy, states=self.states,
                         transitions=self.transitions,
                         initial=self.init_state,
                         after_state_change=['instruct'])
        self.nlu = NLU(config)
        self._load_state_inputs()

    def _load_states(self, config):
        self.states = []
        self.state_mapper = {}
        for state_json in config['states']:
            state = StateCard(state_json)
            self.states.append(state)
            self.state_mapper[state.name] = state
        self.init_state = self.state_mapper[config['init_state']]

    def _load_state_inputs(self):
        for transition in self.transitions:
            trigger_input = transition['trigger']
            state = self.state_mapper[transition['source']]
            state.append_input(trigger_input)

    def kernel(self, q):
        if q == self.RESET_CODE:
            self.reset()
            return self.RESET_CODE
        input_ = self.nlu.process(q, self.current_state())
        return self.drive(input_)

    def reset(self):
        self.set_init_state()
        self.policy.reset()

    def current_state(self):
        return self.state_mapper[self.policy.state]

    def false_instruct(self):
        return self.policy.false_instruct()

    def drive(self, input_):
        try:
            self.policy.trigger(input_)
            return self.policy.current_instruction
        except Exception:
            return self.false_instruct()

    def set_init_state(self):
        self.set_state(state=self.init_state.name)

    def show_graph(self):
        self.get_graph().draw('my_state_diagram.png', prog='dot')


def main():
    with open("wrd_config.json", 'r') as f:
        config = json.load(f)
    machine = Automata(config)
    # machine.set_init_state()

    machine.show_graph()

    while True:
        input_ = input('input:')
        try:
            print(machine.kernel(input_))
        except:
            print(machine.current_state())
    


if __name__ == '__main__':
    main()
    # test()
