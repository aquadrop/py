"""
-------------------------------------------------
   File Name：     nlu
   Description :
   Author :       deep
   date：          18-1-11
-------------------------------------------------
   Change Activity:
                   18-1-11:
                   
   __author__ = 'deep'
-------------------------------------------------
"""

from transitions.extensions import GraphMachine as Machine
import utils.regex_rank as regex_rank

class NLU:

    def __init__(self, config):

        self.interpreter_function_mapper = {"keyword": self.keyword_interpret,
                                       "regex": self.regex_interpret,
                                       "ml": self.ml_interpret}

        self.store_id = config['store_id']
        self.regex_interpreter_helper = config['regex_interpreter_helper']
        self.default_interpreter = self.keyword_interpret

    def process(self, q, state=None):
        """

        :param q:
        :param state: state is state object
        :return:
        """
        if not state:
            return self.default_interpreter(q)
        return self.interpreter_function_mapper[state.interpreter](q, state)

    def keyword_interpret(self, q, state):
        return q

    def regex_interpret(self, q, state):
        avail_inputs = state.get_inputs()
        best_trigger = q
        best_reg_rank = 0
        for input_ in avail_inputs:
            if input_ not in self.regex_interpreter_helper:
                return self.keyword_interpret(q)
            reg = self.regex_interpreter_helper[input_]
            rank = regex_rank.rank_regs(s=q, reg=reg)
            if rank > best_reg_rank:
                best_trigger = input_
        return best_trigger

    def ml_interpret(self, q, state):
        pass