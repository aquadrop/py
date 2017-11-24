from enum import Enum, unique
from transitions import Machine
import random


class FSM(object):

    def __init__(self, name, max_loop_num, states, triggers_map_user, triggers_map_bot, transitions=None):
        self.name = name
        self.max_loop_num = max_loop_num
        self.current_loop_num = 0
        self.init_state = states[0]
        self.terminate_state = states[-1]
        self._init_states(states)
        self._init_transitions(transitions)
        self.triggers_map_user = triggers_map_user
        self.triggers_map_bot = triggers_map_bot
        self.machine = Machine(model=self, states=self.states,
                               transitions=self.transitions, initial=self.init_state)
        self.last_state = self.state

    def _init_states(self, states):
        self.states = states

    def _init_transitions(self, transitions):
        self.transitions = transitions

    def is_init_state(self):
        return self.state == self.init_state

    def is_terminate_state(self):
        return self.state == self.terminate_state

    def get_current_state_num(self):
        if self.state == self.last_state:
            self.current_loop_num += 1
        else:
            self.current_loop_num = 0
        return self.current_loop_num

    def goto_init_state(self):
        self.to_root()

    def goto_next_state(self):
        available_triggers = self.machine.get_triggers(self.state)
        available_triggers = [
            trigger for trigger in available_triggers
            if not trigger.startswith('to_')]

        current_loop_num = self.get_current_state_num()
        if current_loop_num <= self.max_loop_num:
            trigger = random.choice(available_triggers)
        else:
            pass
        self.__dict__[trigger]()

        return trigger

    def travel_cycle(self):
        while True:
            if self.is_terminate_state():
                print('**************')
                self.goto_init_state()
            # print(self.state)
            trigger = self.goto_next_state()


def is_register():
    return False


def is_scan_success():
    return True


def is_auth_success():
    return True


def is_times_scan_fail(current_loop_num, max_loop_num):
    if current_loop_num > max_loop_num:
        return True
    else:
        return False


def is_times_auth_fail(current_loop_num, max_loop_num):
    if current_loop_num > max_loop_num:
        return True
    else:
        return False


def main():
    name = 'xinhua'
    max_loop_num = 2
    states = ['root', 'scan', 'auth', 'success', 'fail']

    triggers_map_user = {
        'register': '注册',
        'scan_success': '扫码成功',
        'scan_fail': '扫码失败',
        'times_scan_fail': '扫码还是失败'
        'auth_success': '验证成功',
        'auth_fail': '验证失败',
        'times_auth_fail': '验证还是失败'
    }
    triggers_map_bot = {
        'register': '请扫码',
        'scan_success': '请验证',
        'scan_fail': '再扫一次吧',
        'times_scan_fail': '别试了,行不行',
        'auth_success': '祝贺你,注册成功',
        'auth_fail': '再验证一次吧',
        'times_auth_fail': '别搞了,你失败了'
    }

    bs = FSM(name, max_loop_num, states,
             triggers_map_user, triggers_map_bot)
    bs.__dict__['is_register'] = is_register
    bs.__dict__['is_scan_success'] = is_scan_success
    bs.__dict__['is_auth_success'] = is_auth_success
    bs.__dict__['is_times_scan_fail'] = is_times_scan_fail
    bs.__dict__['is_times_auth_fail'] = is_times_auth_fail

    bs.machine.add_transition(
        'register', 'root', 'scan', conditions='is_register')
    # bs.machine.add_transition(
    #     'register', 'root', 'root', conditions='is_register')
    bs.machine.add_transition(
        'scan_success', 'scan', 'auth', conditions='is_scan_success')
    bs.machine.add_transition(
        'scan_fail', 'scan', 'scan', conditions='is_scan_success')
    bs.machine.add_transition(
        'auth_success', 'auth', 'success', conditions='is_auth_success')
    bs.machine.add_transition(
        'auth_fail', 'auth', 'auth', conditions='is_auth_success')
    bs.machine.add_transition(
        'times_scan_fail', 'scan', 'fail', conditions='is_times_scan_fail')
    bs.machine.add_transition(
        'times_auth_fail', 'auth', 'fail', conditions='is_times_auth_fail')

    print(bs.state)
    bs.register()
    print(bs.state)


if __name__ == '__main__':
    main()
    # test()
