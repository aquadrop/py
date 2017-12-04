import os
import sys
import pickle

from collections import OrderedDict
from fsm import FSM


class Bookstore(object):
    def __init__(self):
        self.config = dict()
        self.config['name'] = 'xinhua'
        self.config['max_loop_num'] = 2
        self.config['terminate_num'] = 2
        self.config['states'] = ['root', 'scan', 'auth', 'success', 'fail']
        self.config['transitions'] = None
        self.config['conditions_map'] = {
            'is_register': self.is_register,
            'is_scan_success': self.is_scan_success,
            'is_auth_success': self.is_auth_success,
            'is_scan_fail': self.is_scan_fail,
            'is_auth_fail': self.is_auth_fail,
            'is_times_scan_fail': self.is_times_scan_fail,
            'is_times_auth_fail': self.is_times_auth_fail
        }
        self.config['triggers_conditions_map'] = {
            'register': 'is_register',
            'scan_success': 'is_scan_success',
            'scan_fail': 'is_scan_fail',
            'auth_success': 'is_auth_success',
            'auth_fail': 'is_auth_fail',
            'times_scan_fail': 'is_times_scan_fail',
            'times_auth_fail': 'is_times_auth_fail'

        }
        self.init_fsm()
        self.terminate_states = ['success', 'fail']
        self.success_triggers = ['scan_success', 'auth_success']
        self.fail_triggers = ['scan_fail', 'auth_fail']
        self.times_fail_triggers = ['times_scan_fail', 'times_auth_fail']
        self.fail_triggers_map = ['scan_fail':'times_scan_fail', 'auth_fail':'times_auth_fail']

    def init_fsm(self):
        self.fsm = FSM(self.config)
        self.fsm.machine.add_transition(
            'register', 'root', 'scan', conditions='is_register')
        self.fsm.machine.add_transition(
            'scan_success', 'scan', 'auth', conditions='is_scan_success')
        self.fsm.machine.add_transition(
            'scan_fail', 'scan', 'scan', conditions='is_scan_fail')
        self.fsm.machine.add_transition(
            'auth_success', 'auth', 'success', conditions='is_auth_success')
        self.fsm.machine.add_transition(
            'auth_fail', 'auth', 'auth', conditions='is_auth_fail')
        self.fsm.machine.add_transition(
            'times_scan_fail', 'scan', 'fail', conditions='is_times_scan_fail')
        self.fsm.machine.add_transition(
            'times_auth_fail', 'auth', 'fail', conditions='is_times_auth_fail')

    def is_register(current_loop_num, max_loop_num):
        return True

    def is_scan_success(current_loop_num, max_loop_num):
        return True

    def is_auth_fail(current_loop_num, max_loop_num):
        if current_loop_num < max_loop_num:
            return True
        else:
            return False

    def is_scan_fail(current_loop_num, max_loop_num):
        if current_loop_num < max_loop_num:
            return True
        else:
            return False

    def is_auth_success(current_loop_num, max_loop_num):
        return True

    def is_times_scan_fail(current_loop_num, max_loop_num):
        if current_loop_num >= max_loop_num:
            return True
        else:
            return False

    def is_times_auth_fail(current_loop_num, max_loop_num):
        if current_loop_num >= max_loop_num:
            return True
        else:
            return False

    def issue_trigger(trigger):
        if trigger in self.fail_triggers:
            current_loop_num = self.get_current_loop_num()
            if current_loop_num >= self.config['max_loop_num']:
                self.fsm.goto_init_state()
                return self.fail_triggers_map[trigger]
            else:
                return trigger
        if trigger == 'register':
            self.fsm.goto_init_state()
            self.fsm.goto_next_state()
            return trigger
        if trigger in self.success_triggers:
            self.fsm.goto_next_state()
            current_state = self.fsm.state
            if current_state in self.terminate_states:
                self.fsm.goto_init_state()


def main():
    fsm = Bookstore()
    prefix = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(prefix, 'model/graph/fsm_graph.pkl')

    with open(path, 'wb') as f:
        pickle.dump(fsm, f)


if __name__ == '__main__':
    main()
