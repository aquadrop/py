from enum import Enum, unique
from transitions import Machine
import random


class FSM(object):

    def __init__(self, name, states, transitions, triggers_map_user, triggers_map_bot):
        self.name = name
        self.init_state = states[0]
        self.terminate_state = states[-1]
        self._init_states(states)
        self._init_transitions(transitions)
        self.triggers_map_user = triggers_map_user
        self.triggers_map_bot = triggers_map_bot
        # self._register_triggers(triggers_map)
        self.machine = Machine(model=self, states=self.states,
                               transitions=self.transitions, initial=self.init_state)
        # self._register_triggers(triggers_map)

    def _init_states(self, states):
        self.states = states

    def _init_transitions(self, transitions):
        self.transitions = transitions

    def is_init_state(self):
        return self.state == self.init_state

    def is_terminate_state(self):
        return self.state == self.terminate_state

    def goto_init_state(self):
        self.to_root()

    def goto_next_state(self):
        available_triggers = self.machine.get_triggers(self.state)
        available_triggers = [
            trigger for trigger in available_triggers
            if not trigger.startswith('to_')]
        trigger = random.choice(available_triggers)
        self.__dict__[trigger]()

        return self.triggers_map_user[trigger], self.triggers_map_bot[trigger],

    def travel_cycle(self):
        while True:
            if self.is_terminate_state():
                print('**************')
                self.goto_init_state()
            # print(self.state)
            u, b = self.goto_next_state()
            print(u + ' ' + b)


def main():
    name = 'xinhua'
    states = ['root', 'scan', 'auth', 'complete']
    transitions = [
        {'trigger': 'register', 'source': 'root', 'dest': 'scan'},
        {'trigger': 'scan_success', 'source': 'scan', 'dest': 'auth'},
        {'trigger': 'scan_fail', 'source': 'scan', 'dest': 'scan'},
        {'trigger': 'auth_success', 'source': 'auth', 'dest': 'complete'},
        {'trigger': 'auth_fail', 'source': 'auth', 'dest': 'auth'}
    ]
    triggers_map_user = {
        'register': '注册',
        'scan_success': '扫码成功',
        'scan_fail': '扫码失败',
        'auth_success': '验证成功',
        'auth_fail': '验证失败'
    }
    triggers_map_bot = {
        'register': '请扫码',
        'scan_success': '请验证',
        'scan_fail': '再扫一次吧',
        'auth_success': '祝贺你,注册成功',
        'auth_fail': '再验证一次吧'
    }
    bs = FSM(name, states, transitions, triggers_map_user, triggers_map_bot)
    bs.travel_cycle()


if __name__ == '__main__':
    main()
    # test()
