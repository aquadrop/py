from enum import Enum, unique
from transitions import Machine
import random


class FSM(object):
    def __init__(self, config):
        self.name = config['name']
        self.max_loop_num = config['max_loop_num']
        self.current_loop_num = 0
        self.init_state = config['states'][0]
        self.terminate_state = config['states'][-config['terminate_num']:]
        self._init_states(config['states'])
        self._init_transitions(config['transitions'])
        self._register_conditions(config['conditions_map'])
        self.triggers_conditions_map = config['triggers_conditions_map']
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

    def get_current_loop_num(self):
        if self.state == self.last_state:
            self.current_loop_num += 1
        else:
            self.current_loop_num = 0
        self.last_state = self.state
        return self.current_loop_num

    def goto_init_state(self):
        self.to_root()

    def goto_next_state(self, trigger):
        available_triggers = self.machine.get_triggers(self.state)
        current_loop_num = self.get_current_loop_num()
        if trigger in available_triggers:
            self.__dict__[trigger](current_loop_num, self.max_loop_num)
        else:
            print('Can not goto next state with trigger {}'.format(trigger))

    def goto_next_state_random(self):
        available_triggers = self.machine.get_triggers(self.state)
        available_triggers = [
            trigger for trigger in available_triggers
            if not trigger.startswith('to_')]
        # print(available_triggers)

        current_loop_num = self.get_current_loop_num()
        available_triggers = [trigger for trigger in available_triggers if
                              self.__dict__[self.triggers_conditions_map[trigger]](current_loop_num, self.max_loop_num)]
        trigger = random.choice(available_triggers)
        self.__dict__[trigger](current_loop_num, self.max_loop_num)

        return trigger

    def travel_cycle(self, threshold=100):
        count = 0
        previous_len = 0

        res_states = set()
        res_triggers = set()
        one_cycle_states = list()
        one_cycle_triggers = list()
        while True:
            if self.is_terminate_state():
                print(self.state)
                one_cycle_states.append(self.state)
                print('**************')

                one_cycle_states = ','.join(one_cycle_states)
                one_cycle_triggers = ','.join(one_cycle_triggers)
                res_states.add(one_cycle_states)
                res_triggers.add(one_cycle_triggers)

                if len(res_states) == previous_len:
                    count += 1
                else:
                    previous_len = len(res_states)
                    count = 0
                if count > threshold:
                    res_states = list(res_states)
                    res_states = [s.split(',') for s in res_states]
                    res_triggers = list(res_triggers)
                    res_triggers = [t.split(',') for t in res_triggers]
                    break
                one_cycle_states = list()
                one_cycle_triggers = list()
                self.goto_init_state()
            print(self.state)
            one_cycle_states.append(self.state)
            trigger = self.goto_next_state_random()
            print(trigger)
            one_cycle_triggers.append(trigger)

        return {'states': res_states, 'triggers': res_triggers}

    def debug(self):
        triggers = ['register', 'scan_fail', 'scan_fail', 'times_scan_fail']
        for tr in triggers:
            last_state = self.last_state
            state = self.state
            print(self.state)
            current_loop_num = self.current_loop_num
            current_loop_num = self.get_current_loop_num()
            self.__dict__[tr](current_loop_num, self.max_loop_num)
        print(self.state)


def main():
    pass


if __name__ == '__main__':
    main()
    # test()
