import json

from transitions.extensions import LockedHierarchicalGraphMachine as Machine


class Automata(Machine):
    statics_nlu = None

    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        self.policy = self.config['policy']
        self._load_nlu()
        self._init_machine()

    def _load_config(self, config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config

    def _load_nlu(self):
        if not Automata.statics_nlu:
            self.nlu = NLU()
            Automata.statics_nlu = self.nlu
        else:
            self.nlu = Automata.statics_nlu

    def _init_machine(self):
        states = self.config['states']
        transitions = self.config['transitions']
        init_state = self.config['init_state']
        ignore_invalid_triggers = self.config['ignore_invalid_triggers']
        Machine.__init__(self, states=states,
                         transitions=transitions, initial=init_state,
                         ignore_invalid_triggers=ignore_invalid_triggers)

    def control(self, utterence):
        self.trigger = self.nlu.nlu(utterence)
        available_trigger = self.get_triggers(self.state)
        available_trigger = [
            trigger for trigger in available_trigger
            if not trigger.startswith('to_')]
        if self.trigger in available_trigger:
            self.__dict__[self.trigger]()
        else:
            print('Please finish previous steps.')
            self.to_root()

    def response(self):
        key = self.state + ':' + self.trigger
        print(self.policy[key])

    def show_graph(self):
        self.get_graph().draw('my_state_diagram.png', prog='dot')


class NLU(object):
    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        self._load_interpreter()

    def _load_config(self, config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config

    def _load_interpreter(self):
        interpreter_id = self.config['interpreter_id']
        interpreter_path = self.config['interpreter_path'][interpreter_id]
        if interpreter_id == 'regex':
            self.interpreter = self.load_regex_interpreter(interpreter_path)
        elif interpreter_id =='keyword':
            self.interpreter = self.load_keyword_interpreter(interpreter_path)
        elif interpreter_id == 'ml':
            self.interpreter = self.load_ml_interpreter(interpreter_path)

    def load_regex_interpreter(self, interpreter_path):
        with open(interpreter_path, 'r') as f:
            self.regex_mapper = json.load(f)
        return self.regex_mapper

    def load_keyword_interpreter(self, interpreter_path):
        pass

    def load_ml_interpreter(self, interpreter_path):
        pass

    def nlu(self, utterence):
        pass


def main():
    automata = Automata('config.json')
    automata.show_graph()
    # while True:
    #     ipt = input("input:")
    #     automata.control(ipt)


if __name__ == '__main__':
    main()
