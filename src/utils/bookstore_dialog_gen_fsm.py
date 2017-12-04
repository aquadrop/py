import os
import sys

prefix = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, prefix)

from graph.fsm import FSM


class Bookstore(object):
    def __init__(self, config):
        self.config = config
        self.states = self.config['states']
        self.max_loop_num = self.config['max_loop_num']
        self.terminate_num = self.config['terminate_num']
        self.conditions_map = self.config['conditions_map']
        self.triggers_conditions_map = self.config['triggers_conditions_map']
        self.init_fsm()

    def init_fsm(self):
        self.bs = FSM(self.config)
        self.bs.machine.add_transition(
                'register', 'root', 'scan', conditions='is_register')
        self.bs.machine.add_transition(
                'scan_success', 'scan', 'auth', conditions='is_scan_success')
        self.bs.machine.add_transition(
                'scan_fail', 'scan', 'scan', conditions='is_scan_fail')
        self.bs.machine.add_transition(
                'auth_success', 'auth', 'success', conditions='is_auth_success')
        self.bs.machine.add_transition(
                'auth_fail', 'auth', 'auth', conditions='is_auth_fail')
        self.bs.machine.add_transition(
                'times_scan_fail', 'scan', 'fail', conditions='is_times_scan_fail')
        self.bs.machine.add_transition(
                'times_auth_fail', 'auth', 'fail', conditions='is_times_auth_fail')


    def gen_dialog(self):
        dialog = list()
        res = self.bs.travel_cycle()
        triggers_list = res['triggers']
        print(len(triggers_list), triggers_list)
        triggers_map_user = self.config['triggers_map_user']
        triggers_map_bot = self.config['triggers_map_bot']
        for triggers in triggers_list:
            dialog.append([triggers_map_user[trigger] + '##' + triggers_map_bot[trigger]] for trigger in triggers)

        return dialog

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

def main():

    config = dict()
    config['name'] = 'xinhua'
    config['max_loop_num'] = 2
    config['terminate_num'] = 2
    config['states'] = ['root', 'scan', 'auth', 'success', 'fail']
    config['transitions'] = None

    triggers_map_user = {
        'register'       : '注册',
        'scan_success'   : '扫码成功',
        'scan_fail'      : '扫码失败',
        'times_scan_fail': '扫码还是失败',
        'auth_success'   : '验证成功',
        'auth_fail'      : '验证失败',
        'times_auth_fail': '验证还是失败'
    }
    triggers_map_bot = {
        'register'       : '请扫码',
        'scan_success'   : '请验证',
        'scan_fail'      : '再扫一次吧',
        'times_scan_fail': '别试了,行不行',
        'auth_success'   : '祝贺你,注册成功',
        'auth_fail'      : '再验证一次吧',
        'times_auth_fail': '别搞了,猪头'
    }

    conditions_map = {
        'is_register'       : is_register,
        'is_scan_success'   : is_scan_success,
        'is_auth_success'   : is_auth_success,
        'is_scan_fail'      : is_scan_fail,
        'is_auth_fail'      : is_auth_fail,
        'is_times_scan_fail': is_times_scan_fail,
        'is_times_auth_fail': is_times_auth_fail
    }

    triggers_conditions_map = {
        'register'       : 'is_register',
        'scan_success'   : 'is_scan_success',
        'scan_fail'      : 'is_scan_fail',
        'auth_success'   : 'is_auth_success',
        'auth_fail'      : 'is_auth_fail',
        'times_scan_fail': 'is_times_scan_fail',
        'times_auth_fail': 'is_times_auth_fail'

    }

    config['triggers_map_user'] = triggers_map_user
    config['triggers_map_bot'] = triggers_map_bot
    config['conditions_map'] = conditions_map
    config['triggers_conditions_map'] = triggers_conditions_map

    bs = Bookstore(config)
    dialogs = bs.gen_dialog()
    for d in dialogs:
        for line in d:
            print(line)
        print('********')

if __name__ == '__main__':
    main()
    # test()
