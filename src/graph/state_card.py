import random
import uuid

from transitions.extensions import GraphMachine as Machine
import transitions

from graph.policy import Policy


class StateCard(transitions.State):
    def __init__(self, config):
        name = config['name']
        on_enter = config.get('on_enter')
        on_exit = config.get('on_exit')
        ignore_invalid_triggers = config.get('ignore_invalid_triggers', False)
        interpreter = config.get('interpreter', 'keyword')
        super().__init__(name, on_enter, on_exit, ignore_invalid_triggers)
        self.id = str(uuid.uuid4())
        self.max_out_num = 2
        self.interpreter = interpreter
        self._inputs = set()

    def append_input(self, input_):
        self._inputs.add(input_)

    def get_inputs(self):
        return self._inputs.copy()


