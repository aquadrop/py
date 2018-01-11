import random
import uuid

from transitions.extensions import GraphMachine as Machine
import transitions

from graph.policy import Policy


class StateCard(transitions.State):
    def __init__(self, name, on_enter=None, on_exit=None,
                 ignore_invalid_triggers=False, regex=None):
        super().__init__(name, on_enter, on_exit, ignore_invalid_triggers)
        self.id = str(uuid.uuid4())
        self.max_out_num = 2
        self.regex = regex


