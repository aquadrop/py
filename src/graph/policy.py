import random
import uuid

from transitions import Machine


class Policy(object):
    def __init__(self, config):
        self.id = str(uuid.uuid4())
        self.slots = []
        self.instructions = config['instructions']
        self.false_instructions = config['false_instructions']

    def instruct(self):
        print(self.instructions[self.state])

    def false_instruct(self):
        print(self.false_instructions[self.state])


