from enum import Enum, unique
from transitions import Machine
import random


class FSM(object):

    def __init__(self, name, max_loop_num, states, triggers_map_user, triggers_map_bot, conditions_map, transitions=None):
        self.name = name
        self.max_loop_num = max_loop_num
        self.current_loop_num = 0
        self.init_state = states[0]
        self.terminate_state = states[-2:]
        self._init_states(states)
        self._init_transitions(transitions)
        self.triggers_map_user = triggers_map_user
        self.triggers_map_bot = triggers_map_bot
        self._register_conditions(conditions_map)
        self.machine = Machine(model=self, states=self.states,
                               transitions=self.transitions, initial=self.init_state)
        self.last_state = None

    def _init_states(self, states):
        self.states = states

    def _init_transitions(self, transitions):
        self.transitions = transitions

    def is_init_state(self):
        return self.state == self.init_state

    def is_terminate_state(self):
        return self.state in self.terminate_state

    def _register_conditions(self, conditions_map):
        self.conditions_map = dict()
        for k, v in conditions_map.items():
            self.__dict__[k] = v
            self.conditions_map[k] = v

    def get_current_state_num(self):
        if self.state == self.last_state:
            self.current_loop_num += 1
            self.last_state = self.state
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
        # print('current_loop_num: ', current_loop_num)

        trigger = random.choice(available_triggers)
        self.__dict__[trigger](current_loop_num, self.max_loop_num)

        return trigger

    def travel_cycle(self):
        while True:
            if self.is_terminate_state():
                print('**************')
                self.goto_init_state()
            print(self.state)
            trigger = self.goto_next_state()
            # print(self.triggers_map_user[trigger] +
            #       ' ' + self.triggers_map_bot[trigger])


def is_register(current_loop_num, max_loop_num):
    return True


def is_scan_success(current_loop_num, max_loop_num):
    return True


def is_auth_success(current_loop_num, max_loop_num):
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
        'times_scan_fail': '扫码还是失败',
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

    conditions_map = {
        'is_register': is_register,
        'is_scan_success': is_scan_success,
        'is_auth_success': is_auth_success,
        'is_times_scan_fail': is_times_scan_fail,
        'is_times_auth_fail': is_times_auth_fail
    }

    bs = FSM(name, max_loop_num, states,
             triggers_map_user, triggers_map_bot, conditions_map)

    bs.machine.add_transition(
        'register', 'root', 'scan', conditions='is_register')
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

    # print(bs.state)
    # bs.register()
    # print(bs.state)
    bs.travel_cycle()


if __name__ == '__main__':
    main()
    # test()
