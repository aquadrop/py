import os
import sys
import traceback
import pickle

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
grandfatherdir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parentdir)
sys.path.append(grandfatherdir)

from graph.fsm import FSM
from graph.fsm_graph import FSMGraph
# from dmn.dmn_fasttext.dmn_session import DmnInfer
from fsm_render import triggers_map_bot


class FSMKernel(object):
    static_dmn = None
    static_fsm_graph = None

    def __init__(self):
        self._init_fsm_graph()
        # self._init_dmn()
        # self.sess = self.dmn.get_session()

        self.fake_dmn = {
            '注册': 'register',
            '扫码成功': 'scan_success',
            '扫码失败': 'scan_fail',
            '验证成功': 'auth_success',
            '验证失败': 'auth_fail'
        }

    def _init_fsm_graph(self):
        if not FSMKernel.static_fsm_graph:
            try:
                print('attaching fsm graph...100%')
                # with open(self.fsm_graph_path, 'rb') as f:
                # self.fsm_graph = pickle.load(f)
                self.fsm_graph = FSMGraph()
                FSMKernel.static_fsm_graph = self.fsm_graph
            except:
                traceback.print_exc()
        else:
            self.fsm_graph = FSMKernel.static_fsm_graph

    # def _init_dmn(self):
    #     if not FSMKernel.static_dmn:
    #         try:
    #             print('attaching dmn...100%')
    #             self.dmn = DmnInfer()
    #             FSMKernel.static_dmn = self.dmn
    #         except:
    #             traceback.print_exc()
    #     else:
    #         self.dmn = FSMKernel.static_dmn

    def render(self, trigger):
        return triggers_map_bot[trigger]

    def kernel(self, q, user='solr'):
        if not q:
            return 'api_call_error'
        # api, prob = self.sess.reply(q)
        api = self.fake_dmn[q]

        if api.startswith('reserved_'):
            return 'api_call_reserved'
        if api.startswith('api_call_base'):
            return 'api_call_base'
        if api.startswith('api_call_qa'):
            return 'api_call_qa'
        trigger = api
        trigger, jump = self.fsm_graph.issue_trigger(trigger)
        if jump:
            response = self.render(trigger)
        else:
            response = 'Please finish previous steps.'
        return trigger, response


def main():
    fsmkernel = FSMKernel()
    while True:
        ipt = input("input:")
        trigger, response = fsmkernel.kernel(ipt)
        print('trigger:{},response:{}'.format(trigger, response))


if __name__ == '__main__':
    main()
