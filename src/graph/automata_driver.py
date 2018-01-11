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

from transitions.extensions import GraphMachine as Machine
from transitions import State

from graph.policy import Policy
from graph.state_card import StateCard as MyState

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
        self.machine = Machine(model=self.policy, states=self.states,
                               transitions=self.transitions,
                               initial=self.init_state,
                               after_state_change=['instruct'])
        self.last_state = None

    def current_state(self):
        return self.policy.state

    def go(self, input_):
        self.policy.trigger(input_)

    def set_state(self, state_name):
        self.machine.set_state(state=state_name)

    def set_init_state(self):
        self.machine.set_state(state=self.init_state.name)


def main():
    config = {"store_id":"bookstore", "name":"bookstore",
              "instruct":"instruct",
              "states":[MyState(name='root'),
                        MyState(name='register'),
                        MyState(name='auth'),
                        MyState(name='register_complete'),
                        MyState(name='scan_a_0'),
                        MyState(name='auth_a_0'),
                        MyState(name='scan_fail'),
                        MyState(name='auth_fail')],
              "transitions":[
    { 'trigger': 'register', 'source': 'root', 'dest': 'register' },
    { 'trigger': 'ok', 'source': 'register', 'dest': 'auth' },
    { 'trigger': 'ok', 'source': 'auth', 'dest': 'register_complete' },
    { 'trigger': 'oops', 'source': 'register', 'dest': 'scan_a_0' },
    {'trigger': 'oops', 'source': 'scan_a_0', 'dest': 'scan_fail'},
                  {'trigger': 'oops', 'source': 'auth', 'dest': 'auth_a_0'},
                  {'trigger': 'oops', 'source': 'auth_a_0', 'dest': 'auth_fail'},
                  {'trigger': 'oops', 'source': 'auth', 'dest': 'auth_a_0'},
                  {'trigger': 'ok', 'source': 'scan_a_0', 'dest': 'auth'},
                  {'trigger': 'ok', 'source': 'auth_a_0', 'dest': 'register_complete'}]
              }

    instructions = {"root":"hello", "register":"please scan", "auth":"please auth", "register_complete":"congrats",
                    "scan_a_0":"scan again", "scan_fail":"noob", "auth_a_0":"auth again", "auth_fail":"noob"}
    false_instructions = {"root": "hello", "register": "say ok", "auth": "say ok", "register_complete": "congrats",
                    "scan_a_0": "say ok", "scan_fail": "noob", "auth_a_0": "say ok", "auth_fail": "noob"}
    config['instructions'] = instructions
    config['false_instructions'] = false_instructions
    policy = Policy(config)
    config['policy'] = policy

    machine = Automata(config)
    # machine.set_init_state()

    # machine.machine.get_graph().draw('my_state_diagram.png', prog='dot')

    while True:
        input_ = input('input:')
        try:
            machine.go(input_)
        except:
            print(machine.current_state())
    


if __name__ == '__main__':
    main()
    # test()
