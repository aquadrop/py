import json

from collections import OrderedDict
from automata import Automata


class AutomataGraph(object):
    def __init__(self,path):
        self.config=self.load_config(path)
        self.init_automata()
        self.terminate_states = self.config['terminate_states']
        self.success_inputs = self.config['success_inputs']
        self.fail_inputs = self.config['fail_inputs']
        self.times_fail_inputs = self.config['times_fail_inputs']
        self.fail_inputs_map = self.config['fail_inputs_map']

    def load_config(self,path):
        with open(path,'r') as f:
            config=json.load(f)
        config['conditions_map'] = {
            'is_register': self.is_register,
            'is_scan_success': self.is_scan_success,
            'is_auth_success': self.is_auth_success,
            'is_scan_fail': self.is_scan_fail,
            'is_auth_fail': self.is_auth_fail,
            'is_times_scan_fail': self.is_times_scan_fail,
            'is_times_auth_fail': self.is_times_auth_fail
        }
        print(config)
        return config

    def init_automata(self):
        self.automata = Automata(self.config)
        self.automata.machine.add_transition(
            'register', 'root', 'scan', conditions='is_register')
        self.automata.machine.add_transition(
            'scan_success', 'scan', 'auth', conditions='is_scan_success')
        self.automata.machine.add_transition(
            'scan_fail', 'scan', 'scan', conditions='is_scan_fail')
        self.automata.machine.add_transition(
            'auth_success', 'auth', 'success', conditions='is_auth_success')
        self.automata.machine.add_transition(
            'auth_fail', 'auth', 'auth', conditions='is_auth_fail')
        self.automata.machine.add_transition(
            'times_scan_fail', 'scan', 'fail', conditions='is_times_scan_fail')
        self.automata.machine.add_transition(
            'times_auth_fail', 'auth', 'fail', conditions='is_times_auth_fail')

    def is_register(self, current_loop_num, max_loop_num):
        return True

    def is_scan_success(self, current_loop_num, max_loop_num):
        return True

    def is_auth_fail(self, current_loop_num, max_loop_num):
        if current_loop_num < max_loop_num:
            return True
        else:
            return False

    def is_scan_fail(self, current_loop_num, max_loop_num):
        if current_loop_num < max_loop_num:
            return True
        else:
            return False

    def is_auth_success(self, current_loop_num, max_loop_num):
        return True

    def is_times_scan_fail(self, current_loop_num, max_loop_num):
        if current_loop_num >= max_loop_num:
            return True
        else:
            return False

    def is_times_auth_fail(self, current_loop_num, max_loop_num):
        if current_loop_num >= max_loop_num:
            return True
        else:
            return False

    def clear_memory(self):
        self.automata.goto_init_state()
        self.automata.current_loop_num = 0

    def issue_input(self, input):
        if input == 'register':
            self.automata.goto_init_state()
            self.automata.goto_next_state(input)
            return input, True

        jump = self.automata.check_jump(input)
        if not jump:
            return input, False

        if input in self.fail_inputs:
            current_loop_num = self.automata.get_current_loop_num()
            if current_loop_num >= self.config['max_loop_num']:
                self.automata.goto_init_state()
                return self.fail_inputs_map[input], True
            else:
                return input, True

        if input in self.success_inputs:
            self.automata.goto_next_state(input)
            current_state = self.automata.state
            if current_state in self.terminate_states:
                self.automata.goto_init_state()
            return input, True


def main():
    am=AutomataGraph('config.json')


if __name__ == '__main__':
    main()
