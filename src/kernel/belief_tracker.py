"""
Belief Tracker
"""
import traceback
import pickle

import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
grandfatherdir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parentdir)
sys.path.append(grandfatherdir)

import sys
from graph.node import Node
from graph.belief_graph import Graph
from utils.query_util import QueryUtils


class BeliefTracker:
    # static
    static_gbdt = None
    static_belief_graph = None
    static_qa_clf = None

    API_REQUEST_STATE = "api_request_state"
    API_CALL_STATE = "api_call_state"
    TRAVEL_STATE = "travel_state"
    AMBIGUITY_STATE = "ambiguity_state"
    REQUEST_PROPERTY_STATE = "request_property_state"
    RESET_STATE = "reset_state"

    def __init__(self, graph_path):
        self.gbdt = None
        self.state_cleared = True
        self._load_graph(graph_path)
        # self._load_clf(clf_path)
        self.search_node = self.belief_graph.get_root_node()

        # keep tracker of user profile, for instance: name, location, gender
        self.user_slots = {}
        # keep track of pushed product ids
        self.product_push_list = []

        self.score_stairs = [1, 4, 16, 64, 256]
        self.machine_state = None  # API_NODE, NORMAL_NODE
        self.filling_slots = dict()  # current slot
        self.required_slots = list()  # required slot obtained from api_node, ordered

        self.ambiguity_slots = dict()
        # self.negative = False
        # self.negative_clf = Negative_Clf()
        # self.simple = SimpleQAKernel()

    last_slots = None

    guide_url = "http://localhost:11403/solr/sc_sale_gen/select?defType=edismax&indent=on&wt=json"
    # tokenizer_url = "http://localhost:5000/pos?q="

    def kernel(self, query):
        query = QueryUtils.static_simple_remove_punct(query)
        response = self.r_walk_with_pointer_with_clf(
            query=query)
        return response

    def clear_state(self):
        self.search_graph = BeliefGraph()
        self.remaining_slots = {}
        self.negative_slots = {}
        self.negative = False

    def _load_clf(self, path):
        if not BeliefTracker.static_gbdt:
            try:
                print('attaching classifier...100%')
                with open(path, "rb") as input_file:
                    self.gbdt = pickle.load(input_file)
                    BeliefTracker.static_gbdt = self.gbdt
                    # self.gbdt = Multilabel_Clf.load(path)
            except Exception:
                traceback.print_exc()
        else:
            print('skipping attaching gbdt classifier as already attached...')
            self.gbdt = BeliefTracker.static_gbdt

    def _load_graph(self, path):
        if not BeliefTracker.static_belief_graph:
            try:
                print('attaching logic graph...100%')
                with open(path, "rb") as input_file:
                    self.belief_graph = pickle.load(input_file)
                    BeliefTracker.static_belief_graph = self.belief_graph
            except:
                traceback.print_exc()
        else:
            self.belief_graph = BeliefTracker.static_belief_graph

    def travel_with_clf(self, query):
        """
        the memory network predictor has mainly two types of classes: api_call_... and slots_...
        """
        filtered_slots_list = []
        try:
            # flipped, self.negative = self.negative_clf.predict(input_=query)
            intention, prob = self.rnn_clf(query)
            filtered_slot_values_list = self.unfold_slots_list(intention)
        except:
            traceback.print_exc()
            return False

        # build belief graph
        # self.update_remaining_slots(expire=True)
        # filtered_slots_list = self.inter_fix(filtered_slots_list)
        # self.should_expire_all_slots(filtered_slots_list)
        self.update_belief_graph(
            slot_values_list=filtered_slot_values_list)
        return self.issune_api()

    def unfold_slots_list(self, intention):
        """
        api_call, slots_ We unfold slots from slots_
        """
        # slots = "_".join(intention.split("_")[1:-1]).split(",")
        # return slots
        return intention

    def move_to_node(self, node):
        self.search_node = node
        self.required_slots = self.search_node.gen_required_slot_fields()
        self.filling_slots.clear()

    def fill_slot(self, slot, value):
        self.filling_slots[slot] = value
        if slot in self.required_slots:
            self.required_slots.remove(slot)

    def update_belief_graph(self, slot_values_list, slot_values_marker=None):
        """
        1. if node is single, go directly
        2. if there are ambiguity nodes, try remove ambiguity resorting to slot_values_list AND search_parent_node
            2.1 if fail, enter ambiguity state
        """
        if not slot_values_marker:
            slot_values_marker = [0] * len(slot_values_list)
        # slot_values_list = list(set(slot_values_list))

        if self.machine_state == self.API_CALL_STATE:
            self.machine_state = self.TRAVEL_STATE
            self.move_to_node(self.belief_graph.get_root_node())
            return self.update_belief_graph(slot_values_list, slot_values_marker)

        if self.machine_state == self.API_CALL_STATE:
            self.machine_state = self.TRAVEL_STATE
            self.move_to_node(self.belief_graph.get_root_node())
            return self.update_belief_graph(slot_values_list, slot_values_marker)

        if self.machine_state == self.AMBIGUITY_STATE:
            for i, value in enumerate(slot_values_list):
                if slot_values_marker[i] == 1:
                    continue
                if value in self.ambiguity_slots:
                    slot_values_marker[i] = 1
                    self.machine_state = self.TRAVEL_STATE
                    self.move_to_node(self.ambiguity_slots[value])
                    return self.update_belief_graph(slot_values_list=slot_values_list, slot_values_marker=slot_values_marker)
            # ambiguity removal failed, abandon
            self.machine_state = self.TRAVEL_STATE
            self.move_to_node(self.belief_graph.get_root_node())
            return self.update_belief_graph(slot_values_list=slot_values_list, slot_values_marker=slot_values_marker)

        # session terminal end api_call_node
        # issune api_call_to_solr
        # node_水果 是 api_node, 没有必要再继续往下了
        if self.search_node.is_api_node():
            self.machine_state = self.API_REQUEST_STATE
            if len(self.required_slots) == 0:
                self.machine_state = self.API_CALL_STATE
                return
            for i, value in enumerate(slot_values_list):
                if slot_values_marker[i] == 1:
                    continue
                if self.search_node.has_child(value):
                    # slot_values_marker[i] = 1
                    slot = self.search_node.get_slot_by_value(value)
                    self.fill_slot(slot, value)

                    if len(self.required_slots) == 0:
                        self.machine_state = self.API_CALL_STATE
                else:
                    if self.search_node.has_ancestor_by_value(value):
                        # just ignore this stupid input value
                        continue
                    self.move_to_node(self.belief_graph.get_root_node())
                    self.machine_state = self.API_REQUEST_STATE

                    return self.update_belief_graph(
                        slot_values_list=slot_values_list, slot_values_marker=slot_values_marker)
            return

        # if property node, go up
        if self.search_node.is_property_node():
            parent_node = self.search_node.parent_node
            slot = self.search_node.slot
            value = self.search_node.value
            self.move_to_node(parent_node)
            # mark parent slot_value mark 1
            if parent_node.value in slot_values_list:
                slot_values_marker[slot_values_list.index(
                    parent_node.value)] = 1
            self.fill_slot(slot, value)
            return self.update_belief_graph(slot_values_list=slot_values_list, slot_values_marker=slot_values_marker)

        for i, value in enumerate(slot_values_list):
            if slot_values_marker[i] == 1:
                continue
            candidate_nodes = self.search_node.get_posterity_nodes_by_value(
                value, self.belief_graph)
            # consider single node first
            if len(candidate_nodes) == 1:
                next_parent_search_node = candidate_nodes[0]
                slot_values_marker[i] = 1
                self.move_to_node(next_parent_search_node)
                self.machine_state = self.API_REQUEST_STATE
                return self.update_belief_graph(
                    slot_values_list=slot_values_list, slot_values_marker=slot_values_marker)

            # enter ambiguity state
            if len(candidate_nodes) > 1:
                slot_values_marker[i] = 1
                self.machine_state = self.AMBIGUITY_STATE
                # search_node stays
                #
                filtered = [1] * len(candidate_nodes)
                removal_slot_value_index = set()
                for k, node in enumerate(candidate_nodes):
                    parent_values = node.get_ancestry_values()

                    # self.ambiguity_slots[parent_values[0]] = node
                    # filter
                    for j, slot_value in enumerate(slot_values_list):
                        if slot_values_marker[j] == 1:
                            continue
                        if slot_value not in parent_values:
                            filtered[k] = 0
                        else:
                            removal_slot_value_index.add(j)
                filtered_nodes = []
                for m in removal_slot_value_index:
                    slot_values_marker[m] = 1
                for idx, flag in enumerate(filtered):
                    if flag == 1:
                        filtered_nodes.append(candidate_nodes[idx])
                if len(filtered_nodes) == 1:
                    # found
                    # remove AMBIGUITY_STATE
                    node = filtered_nodes[0]
                    self.machine_state = self.TRAVEL_STATE
                    self.move_to_node(node)
                    return self.update_belief_graph(
                        slot_values_list=slot_values_list,
                        slot_values_marker=slot_values_marker)
                elif len(filtered_nodes) > 1:
                    for node in filtered_nodes:
                        self.ambiguity_slots[node.parent_node.value] = node
                    return
                else:
                    self.machine_state = self.RESET_STATE
                    self.move_to_node(self.belief_graph.get_root_node())
                    return

            # go directly to ROOT
            self.move_to_node(self.belief_graph.get_root_node())
            self.machine_state = self.API_REQUEST_STATE
            return self.update_belief_graph(
                slot_values_list=slot_values_list, slot_values_marker=slot_values_marker)

    def issune_api(self):
        if self.machine_state == self.API_REQUEST_STATE:
            slot = self.required_slots[0]
            return "api_request_value_of_" + slot
        if self.machine_state == self.AMBIGUITY_STATE:
            param = ','.join(self.ambiguity_slots.keys())
            return "api_request_ambiguity_removal_" + param
        if self.machine_state == self.API_CALL_STATE:
            # first filling slots
            param = "api_call_"
            fill = []
            for key, value in self.filling_slots.items():
                fill.append(key + ":" + value)

            node = self.search_node
            while node.value != self.belief_graph.ROOT:
                fill.append(node.slot + ":" + node.value)
                node = node.parent_node
            return param + ",".join(fill)
        if self.machine_state == self.RESET_STATE:
            return "reset_requested"

    def r_walk_with_pointer_with_clf(self, query):
        if not query or len(query) == 1:
            return 'invalid query'
        api = self.travel_with_clf(query)
        if not api:
            return 'invalid query'
        # return self.search()
        return api

    def single_last_slot(self, split=' OR '):
        return self.single_slot(self.last_slots, split=split)

    def remove_slots(self, key):
        new_remaining_slots = {}
        for remaining_slot, index in self.remaining_slots.items():
            if remaining_slot == key:
                continue
            node = self.belief_graph.get_global_node(remaining_slot)
            if node.has_ancester(key):
                continue
            new_remaining_slots[remaining_slot] = self.remaining_slots[remaining_slot]
        self.remaining_slots = new_remaining_slots

    def single_slot(self, slots, split=' OR '):
        return split.join(slots)

    def flag_state(self):
        self.state_cleared = False

    def compose(self):
        intentions = []
        size = len(self.remaining_slots)
        for slot, i in self.remaining_slots.items():
            node = self.belief_graph.get_global_node(slot)
            score = self.score_stairs[i]
            importance = self.belief_graph.slot_importances[slot]
            if size > 2:
                if self.negative_slots[slot] and node.is_leaf(Node.KEY):
                    slot = '-' + slot
            elif size == 2:
                if self.negative_slots[slot] and self.belief_graph.slot_identities[slot] != 'intention':
                    slot = self.sibling(slot=slot, maximum_num=1)[0]
            elif size == 1:
                if self.negative_slots[slot]:
                    slot = self.sibling(slot=slot, maximum_num=1)[0]
            intention = slot + '^' + str(float(score) * float(importance))
            intentions.append(intention)
        return intentions, ' OR '.join(intentions)

    def contain_negative(self, intentions):
        for intention in intentions:
            if "-" in intention:
                return True

        return False

    def sibling(self, slot, maximum_num):
        black_list = ['facility', 'entertainment']
        node = self.belief_graph.get_global_node(slot)
        sibling = node.sibling_names(value_type=Node.KEY)
        # must be of same identities
        identity = self.belief_graph.slot_identities[slot.decode('utf-8')]
        cls_sibling = []
        for s in sibling:
            try:
                if s in black_list:
                    continue
                if self.belief_graph.slot_identities[s.decode('utf-8')] == identity:
                    cls_sibling.append(s)
            except:
                pass
        maximum_num = np.minimum(maximum_num, len(cls_sibling))
        return np.random.choice(a=cls_sibling, replace=False, size=maximum_num)

    def rnn_clf(self, q):
        # try:
        #     rnn_url = "http://localhost:10001/sc/rnn/classify?q={0}".format(q)
        #     r = requests.get(rnn_url)
        #     text = r.text
        #     if text:
        #         slots_list = text.split(",")
        #         probs = [1.0 for slot in slots_list]
        #         return slots_list, probs
        #     else:
        #         return None, None
        # except:
        #     return None, None
        slot_values_list = q.split(",")
        probs = [1.0] * len(slot_values_list)
        return slot_values_list, probs

    def should_clear_state(self, multi_slots):
        try:
            single_slot = self.single_slot(multi_slots)
            node = self.graph.get_global_node(single_slot)
            if node.is_leaf(Node.REGEX):
                self.clear_state()
        except:
            self.clear_state()


def test():
    with open(os.path.join(grandfatherdir, "model/graph/belief_graph.pkl"), "rb") as input_file:
    bt = BeliefTracker(os.path.join(grandfatherdir,
                                    "model/graph/belief_graph.pkl"))

    with open(os.path.join(grandfatherdir, "log/test2.log"), 'a') as logfile:
        while(True):
            try:
                ipt = input("input:")
                print(ipt, file=logfile)
                resp = bt.kernel(ipt)
                print(resp)
                print(resp, file=logfile)
            except Exception as e:
                print(e)
                print('error:', e, end='\n\n', file=logfile)
                break


if __name__ == "__main__":
    with open(os.path.join(grandfatherdir, "model/graph/belief_graph.pkl"), "rb") as input_file:
        bt = BeliefTracker(os.path.join(grandfatherdir,
                                        "model/graph/belief_graph.pkl"))

        with open(os.path.join(grandfatherdir, "log/test2.log"), 'a') as logfile:
            while(True):
                try:
                    ipt = input("input:")
                    print(ipt, file=logfile)
                    resp = bt.kernel(ipt)
                    print(resp)
                    print(resp, file=logfile)
                except Exception as e:
                    print(e)
                    print('error:', e, end='\n\n', file=logfile)
                    break
