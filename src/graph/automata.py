import random

from transitions import Machine

class Automata(object):
    def __init__(self, config):
        self.name = config['name']
        self.max_loop_num = config['max_loop_num']
        self.current_loop_num = 0
        self.init_state = config['states'][0]
        self.terminate_state = config['states'][-config['terminate_num']:]
        self._init_states(config['states'])
        self._init_transitions(config['transitions'])
        self._register_conditions(config['conditions_map'])
        self.inputs_conditions_map = config['inputs_conditions_map']
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

    def check_jump(self,input):
        available_inputs = self.machine.get_triggers(self.state)
        if input in available_inputs:
            return True
        else:
            return False

    def goto_next_state(self, input):
        # print(available_inputs)
        current_loop_num = self.get_current_loop_num()
        self.__dict__[input](current_loop_num, self.max_loop_num)

    def goto_next_state_random(self):
        available_inputs = self.machine.get_triggers(self.state)
        available_inputs = [
            input for input in available_inputs
            if not input.startswith('to_')]
        # print(available_inputs)

        current_loop_num = self.get_current_loop_num()
        available_inputs = [input for input in available_inputs if
                              self.__dict__[self.inputs_conditions_map[input]](current_loop_num, self.max_loop_num)]
        input = random.choice(available_inputs)
        self.__dict__[input](current_loop_num, self.max_loop_num)

        return input

    def travel_cycle(self, threshold=100):
        count = 0
        previous_len = 0

        res_states = set()
        res_inputs = set()
        one_cycle_states = list()
        one_cycle_inputs = list()
        while True:
            if self.is_terminate_state():
                print(self.state)
                one_cycle_states.append(self.state)
                print('**************')

                one_cycle_states = ','.join(one_cycle_states)
                one_cycle_inputs = ','.join(one_cycle_inputs)
                res_states.add(one_cycle_states)
                res_inputs.add(one_cycle_inputs)

                if len(res_states) == previous_len:
                    count += 1
                else:
                    previous_len = len(res_states)
                    count = 0
                if count > threshold:
                    res_states = list(res_states)
                    res_states = [s.split(',') for s in res_states]
                    res_inputs = list(res_inputs)
                    res_inputs = [t.split(',') for t in res_inputs]
                    break
                one_cycle_states = list()
                one_cycle_inputs = list()
                self.goto_init_state()
            print(self.state)
            one_cycle_states.append(self.state)
            input = self.goto_next_state_random()
            print(input)
            one_cycle_inputs.append(input)

        return {'states': res_states, 'inputs': res_inputs}

    def debug(self):
        inputs = ['register', 'scan_fail', 'scan_fail', 'times_scan_fail']
        for tr in inputs:
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
