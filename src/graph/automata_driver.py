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

from transitions.extensions import GraphMachine as Machine
from transitions import State

from graph.policy import Policy
from graph.state_card import StateCard as MyState
from graph.nlu import NLU

class Automata(object):
    def __init__(self, config):
        self.id = str(uuid.uuid4())
        self.store_id = config['store_id']
        self.name = config['name']
        self.init_state = config['states'][0]
        self.states = config['states']
        self.transitions = config['transitions']
        self.policy = config['policy']
        self.instruct = config['instruct']
        self.inputs_regex_interpret_helper =  config['inputs_regex_interpret_helper']
        self.graph = Machine(model=self.policy, states=self.states,
                               transitions=self.transitions,
                               initial=self.init_state,
                               after_state_change=['instruct'])
        self.nlu = NLU(config, self.graph)
        self._load_state_mapper()
        self._load_state_inputs()

    def _load_state_mapper(self):
        self.state_mapper = {}
        for state in self.states:
            self.state_mapper[state.name] = state

    def _load_state_inputs(self):
        for transition in self.transitions:
            trigger_input = transition['trigger']
            state = self.state_mapper[transition['source']]
            state.append_input(trigger_input)

    def kernel(self, q):
        input_ = self.nlu.process(q, self.current_state())
        return self.drive(input_)

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

    def set_state(self, state_name):
        self.graph.set_state(state=state_name)

    def set_init_state(self):
        self.graph.set_state(state=self.init_state.name)


def main():
    config = {"store_id":"bookstore", "name":"bookstore",
              "instruct":"instruct",
              "states":[MyState(name='root', interpreter='regex'),
                        MyState(name='register', interpreter='regex'),
                        MyState(name='auth', interpreter='regex'),
                        MyState(name='register_complete', interpreter='regex'),
                        MyState(name='scan_a_0', interpreter='regex'),
                        MyState(name='auth_a_0', interpreter='regex'),
                        MyState(name='scan_fail', interpreter='regex'),
                        MyState(name='auth_fail', interpreter='regex')],
              "transitions":[
            { 'trigger': 'register', 'source': 'root', 'dest': 'register' },
                    { 'trigger': 'ok', 'source': 'register', 'dest': 'auth' },
                  {'trigger': 'auth', 'source': 'register', 'dest': 'auth'},
        { 'trigger': 'ok', 'source': 'auth', 'dest': 'register_complete' },
        { 'trigger': 'oops', 'source': 'register', 'dest': 'scan_a_0' },
        {'trigger': 'oops', 'source': 'scan_a_0', 'dest': 'scan_fail'},
                  {'trigger': 'oops', 'source': 'auth', 'dest': 'auth_a_0'},
                  {'trigger': 'oops', 'source': 'auth_a_0', 'dest': 'auth_fail'},
                  {'trigger': 'oops', 'source': 'auth', 'dest': 'auth_a_0'},
                  {'trigger': 'ok', 'source': 'scan_a_0', 'dest': 'auth'},
                  {'trigger': 'ok', 'source': 'auth_a_0', 'dest': 'register_complete'}]
              }

    inputs_interpreter_regex_helper = {'register': r'register|注册',
                                       'ok': r'ok',
                                       'oops': r'不行',
                                       'auth': r'auth'}

    instructions = {"root":"hello", "register":"please scan", "auth":"please auth", "register_complete":"congrats",
                    "scan_a_0":"scan again", "scan_fail":"noob", "auth_a_0":"auth again", "auth_fail":"noob"}
    false_instructions = {"root": "hello", "register": "say ok", "auth": "say ok", "register_complete": "congrats",
                    "scan_a_0": "say ok", "scan_fail": "noob", "auth_a_0": "say ok", "auth_fail": "noob"}
    config['instructions'] = instructions
    config['false_instructions'] = false_instructions
    policy = Policy(config)
    config['policy'] = policy
    config['inputs_regex_interpret_helper'] = inputs_interpreter_regex_helper
    machine = Automata(config)
    # machine.set_init_state()

    # machine.machine.get_graph().draw('my_state_diagram.png', prog='dot')

    while True:
        input_ = input('input:')
        try:
            print(machine.kernel(input_))
        except:
            print(machine.current_state())
    


if __name__ == '__main__':
    main()
    # test()
