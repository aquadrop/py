from enum import Enum, unique


@unique()
class State(Enum):
    root = 1
    scan = 2
    auth = 3
    complete = 4
