import random
import uuid

from transitions import Machine


class Policy(object):
    def __init__(self, config):
        self.id = str(uuid.uuid4())
        self.slots = []
        self.instructions = config['instructions']
        self.false_instructions = config['false_instructions']

        self.current_instruction = None

    def instruct(self):
        self.current_instruction = self.instructions[self.state]

    def false_instruct(self):
        return self.false_instructions[self.state]

    def reset(self):
        self.current_instruction = None


