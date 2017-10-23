"""
Belief Tracker
"""
import traceback
import pickle
import re
import json

import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
grandfatherdir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parentdir)
sys.path.append(grandfatherdir)

dir_path = os.path.dirname(os.path.realpath(__file__))

from SolrClient import SolrClient

import sys
from graph.node import Node
from graph.belief_graph import Graph
from utils.cn2arab import *
import utils.query_util as query_util
import utils.solr_util as solr_util

class BeliefTracker:
    # static
    static_gbdt = None
    static_belief_graph = None
    static_qa_clf = None

    API_REQUEST_STATE = "api_request_state"
    API_REQUEST_RULE_STATE = "api_request_rule_state"
    API_CALL_STATE = "api_call_state"
    TRAVEL_STATE = "travel_state"
    AMBIGUITY_STATE = "ambiguity_state"
    REQUEST_PROPERTY_STATE = "request_property_state"
    NO_CHILD_STATE = 'no_child_state'
    RESET_STATE = "reset_state"

    VIRTUAL = 'virtual_category'
    API = 'category'
    PROPERTY = 'property'
    AMBIGUITY_PICK = 'ambiguity_removal'

    def __init__(self, config):
        self.config = config
        self.gbdt = None
        self.state_cleared = True
        self._load_graph(config['belief_graph'])
        # self._load_clf(clf_path)
        self.search_node = self.belief_graph.get_root_node()

        # keep tracker of user profile, for instance: name, location, gender
        self.user_slots = {}
        # keep track of pushe   d product ids
        self.product_push_list = []

        self.score_stairs = [1, 4, 16, 64, 256]
        self.machine_state = None  # API_NODE, NORMAL_NODE
        self.filling_slots = dict()  # current slot
        self.requested_slots = list()  # required slot obtained from api_node, ordered
        self.numerical_slots = dict()  # common like distance, size, price
        self.ambiguity_slots = dict()
        self.wild_card = dict()
        self.place_holder = dict()
        self.avails = dict()

        # self.negative = False
        # self.negative_clf = Negative_Clf()
        # self.simple = SimpleQAKernel()

    last_slots = None

    guide_url = "http://localhost:11403/solr/sc_sale_gen/select?defType=edismax&indent=on&wt=json"
    # tokenizer_url = "http://localhost:5000/pos?q="
    solr = SolrClient('http://localhost:11403/solr')

    def kernel(self, query):
        response = self.r_walk_with_pointer_with_clf(
            query=query)
        return response

    def memory_kernel(self, query, query_mapper, wild_card=None):
        if isinstance(query_mapper, str):
            query_mapper = json.loads(query_mapper, encoding='utf-8')

        self.color_graph(query=query, slot_values_mapper=query_mapper, range_render=True)
        # self.use_wild_card(wild_card)
        if wild_card:
            self.exploit_wild_card(wild_card=wild_card)
        print(self.requested_slots)
        api, avails = self.issue_api()
        return api, avails

    def user_wild_card(self, wild_card):
        if 'price' in wild_card:
            pass
        if '_inch_' in wild_card:
            pass

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
        # clear wide card
        self.wild_card.clear()
        if self.search_node == self.belief_graph.get_root_node():
            self.rule_base_num_retreive(query)
        self.update_belief_graph(query=query,
                                 slot_values_list=filtered_slot_values_list)
        return self.issue_api() + '@@' + self.issue_class()

    def unfold_slots_list(self, intention):
        """
        api_call, slots_ We unfold slots from slots_
        """
        # slots = "_".join(intention.split("_")[1:-1]).split(",")
        # return slots
        return intention

    def move_to_node(self, node):
        self.search_node = node
        # self.clear_memory()
        self.filling_slots.clear()
        self.requested_slots = self.search_node.gen_required_slot_fields()
        self.fill_slot(node.slot, node.value)

    def fill_slot(self, slot, value):
        self.filling_slots[slot] = value
        if slot in self.requested_slots:
            self.requested_slots.remove(slot)

    def clear_memory(self):
        self.filling_slots.clear()
        self.requested_slots.clear()
        self.machine_state = self.TRAVEL_STATE
        self.search_node = self.belief_graph.get_root_node()
        # self.requested_slots.append(self.API)

    def graph_render(self, value_list, required_field):
        """
        Rool Based will be introduced
        :param value_list:
        :param required_field:
        :return:
        """
        field_type = self.belief_graph.get_field_type(required_field)
        pass

    def get_requested_field(self):
        if len(self.requested_slots) == 0:
            return None
        return self.requested_slots[0]

    # a tree rendering process...
    def color_graph(self, slot_values_mapper, query=None, values_marker=None, range_render=True):
        """
        gen api_call_ambiguity_...
            api_call_request_brand...
        see wiki: https://github.com/aquadrop/memory_py/wiki
        :param slot_values_mapper: {"entity":[entities], "slot1":"value1", "slot2":"value2"}
        :param values_marker:
        :param query
        :param range_render
        :return:
        """

        if not values_marker:
            values_marker = dict()
            for key, value in slot_values_mapper.items():
                values_marker[key] = 0

        if self.machine_state == self.AMBIGUITY_STATE:
            flag = True
            if self.AMBIGUITY_PICK in slot_values_mapper:
                self.machine_state = self.API_REQUEST_STATE
                values_marker[self.AMBIGUITY_STATE] = 1
                value = slot_values_mapper[self.AMBIGUITY_PICK]
                if value in self.ambiguity_slots:
                    nodes = self.ambiguity_slots[value]
                    if len(nodes) == 1:
                        self.move_to_node(nodes[0].parent_node)
                        self.fill_slot(nodes[0].slot, nodes[0].value)
                        self.ambiguity_slots.clear()
                    else:
                        flag = False
                        self.ambiguity_slots.clear()
                        for node in nodes:
                            # self.machine_state = self.AMBIGUITY_STATE
                            self.ambiguity_slots[node.slot] = [node]
            else:
                # check category before leaving
                api_slot_value = ''
                fetch_key = [self.API, 'entity']
                if fetch_key[0] in slot_values_mapper:
                    api_slot_value = slot_values_mapper[fetch_key[0]]
                    values_marker[fetch_key[0]] = 1
                if fetch_key[1] in slot_values_mapper:
                    api_slot_value = slot_values_mapper[fetch_key[1]]
                    values_marker[fetch_key[1]] = 1
                if api_slot_value in self.ambiguity_slots:
                    nodes = self.ambiguity_slots[api_slot_value]
                    if len(nodes) == 1:
                        self.move_to_node(nodes[0].parent_node)
                        self.fill_slot(nodes[0].slot, nodes[0].value)
                        self.ambiguity_slots.clear()
                    else:
                        flag = False
                        self.ambiguity_slots.clear()
                        for node in nodes:
                            # self.machine_state = self.AMBIGUITY_STATE
                            self.ambiguity_slots[node.slot] = [node]
                else:
                    # fail to remove ambiguity
                    self.machine_state = self.API_REQUEST_STATE
                    self.ambiguity_slots.clear()
                    flag = True
            if self.AMBIGUITY_PICK in self.requested_slots and flag:
                self.requested_slots.remove(self.AMBIGUITY_PICK)
            # del slot_values_mapper[self.AMBIGUITY_PICK]
        # look for virtual memory
        if self.VIRTUAL in slot_values_mapper:
            self.machine_state = self.API_REQUEST_STATE
            value = slot_values_mapper[self.VIRTUAL]
            values_marker[self.VIRTUAL] = 1
            node = self.belief_graph.get_nodes_by_value(value)[0]
            self.move_to_node(node)

        # look for api memory
        if self.API in slot_values_mapper:
            if values_marker[self.API] == 0:
                if len(slot_values_mapper) == 1 or self.API not in self.filling_slots\
                        or slot_values_mapper[self.API] != self.filling_slots[self.API]:
                    self.machine_state = self.API_REQUEST_STATE
                    value = slot_values_mapper[self.API]
                    node = self.belief_graph.get_nodes_by_value(value)[0]
                    self.move_to_node(node)

        for key, value in slot_values_mapper.items():
            if key == self.VIRTUAL or key == self.API or key == self.AMBIGUITY_PICK:
                # checked
                continue
            if key != 'entity' and self.belief_graph.get_field_type(key) == Node.RANGE:
                # value is range
                if range_render:
                    self.rule_base_fill(query, key)
                else:
                    self.fill_slot(key, value)
                # self.fill_slot(key, value)
                continue
            if key == 'entity':
                nodes = self.belief_graph.get_nodes_by_value(value)
            else:
                nodes = self.belief_graph.get_nodes_by_value_and_field(value, key)
            if len(nodes) == 1:
                node = nodes[0]
                #
                if node.has_ancestor_by_value(self.search_node.value):
                    if node.parent_node != self.search_node:
                        # move to parent node if relation is grand
                        self.move_to_node(node.parent_node)
                    self.fill_slot(node.slot, node.value)
            else:
                filtered = []
                # ambiguity state, 删除非self.search_node节点
                for i, node in enumerate(nodes):
                    if node.has_ancestor_by_value(self.search_node.value):
                        filtered.append(node)
                if len(filtered) == 1:
                    self.fill_slot(filtered[0].slot, filtered[0].value)
                elif len(filtered) > 1:
                    # enter ambiguity state
                    # self.machine_state = self.AMBIGUITY_STATE
                    self.requested_slots.insert(0, self.AMBIGUITY_PICK)
                    parent_values = set()
                    for node in filtered:
                        parent_values.add(node.parent_node.value)
                    if len(parent_values) > 1:
                        for node in filtered:
                            if node.parent_node.value not in self.ambiguity_slots:
                                self.ambiguity_slots[node.parent_node.value] = []
                            self.ambiguity_slots[node.parent_node.value].append(node)
                    else:
                        for node in filtered:
                            self.ambiguity_slots[node.slot] = [node]
                else:
                    # swith branch
                    # current_requested = self.get_requested_field()
                    # if current_requested == key:
                    #     # remain in the current node
                    #     self.machine_state = self.NO_CHILD_STATE
                    # else:
                    self.move_to_node(self.belief_graph.get_root_node())
                    # return self.color_graph(query=query, slot_values_mapper=slot_values_mapper)
                    self.clear_memory()

        if len(self.requested_slots) == 0:
            self.machine_state = self.API_CALL_STATE
            # placeholder
            for key, holder in self.place_holder.items():
                self.filling_slots[key] = holder
        else:
            self.machine_state = self.API_REQUEST_STATE
            if self.requested_slots[0] == self.AMBIGUITY_PICK:
                self.machine_state = self.AMBIGUITY_STATE

    def exploit_wild_card(self, wild_card, given_slot=None):
        """
        shall_exploit_range is true
        :param wild_card:
        :return:
        """
        flag = False

        if given_slot:
            adapter = self.belief_graph.range_adapter(given_slot)
            if adapter in wild_card:
                self.fill_slot(given_slot, wild_card[adapter])
                flag = True
            if not flag:
                if self.shall_exploit_range():
                    if 'number' in wild_card:
                        self.fill_slot(self.get_requested_field(), wild_card['number'])
                        flag = True
            return flag

        for slot in self.requested_slots:
            if slot in self.belief_graph.range_adapter_mapper:
                adapter = self.belief_graph.range_adapter(slot)
                if adapter in wild_card and adapter != 'number':
                    self.fill_slot(slot, wild_card[adapter])
                    flag = True
        if not flag:
            if self.shall_exploit_range():
                if 'number' in wild_card:
                    self.fill_slot(self.get_requested_field(), wild_card['number'])
                    flag = True
                    del wild_card['number']
        # fill and change state
        if len(self.requested_slots) == 0:
            self.machine_state = self.API_CALL_STATE
        return flag

    def shall_exploit_range(self):
        requested = self.get_requested_field()
        if not requested:
            return False
        if Node.RANGE == self.belief_graph.get_field_type(requested):
            return True
        return False

    def update_belief_graph(self, slot_values_list, query, slot_values_marker=None):
        """
        1. if node is single, go directly
        2. if there are ambiguity nodes, try remove ambiguity resorting to slot_values_list AND search_parent_node
            2.1 if fail, enter ambiguity state
        """

        def has_child(values):
            for v in values:
                if self.search_node.has_child(v):
                    self.rule_base_num_retreive(query)
                    return True
            return False

        if not slot_values_marker:
            slot_values_marker = [0] * len(slot_values_list)
        # slot_values_list = list(set(slot_values_list))

        if self.machine_state == self.API_CALL_STATE:
            if has_child(slot_values_list):
                # restart search node
                self.move_to_node(self.search_node)
            else:
                self.machine_state = self.TRAVEL_STATE
                self.move_to_node(self.belief_graph.get_root_node())
                return self.update_belief_graph(slot_values_list=slot_values_list,
                                                slot_values_marker=slot_values_marker, query=query)

        if self.machine_state == self.AMBIGUITY_STATE:
            for i, value in enumerate(slot_values_list):
                if slot_values_marker[i] == 1:
                    continue
                if value in self.ambiguity_slots:
                    slot_values_marker[i] = 1
                    nodes = self.ambiguity_slots[value]
                    if len(nodes) == 1:
                        node = nodes[0]
                        self.machine_state = self.TRAVEL_STATE
                        self.move_to_node(node)
                        return self.update_belief_graph(slot_values_list=slot_values_list,
                                                        slot_values_marker=slot_values_marker, query=query)
                    else:
                        self.ambiguity_slots.clear()
                        for node in nodes:
                            self.ambiguity_slots[node.slot] = [node]
                        # return self.update_belief_graph(slot_values_list=slot_values_list,
                        #                                 slot_values_marker=slot_values_marker, query=query)
                        return

            # ambiguity removal failed, abandon
            self.machine_state = self.TRAVEL_STATE
            self.ambiguity_slots.clear()
            self.move_to_node(self.belief_graph.get_root_node())
            return self.update_belief_graph(slot_values_list=slot_values_list,
                                            slot_values_marker=slot_values_marker, query=query)

        # session terminal end api_call_node
        # issune api_call_to_solr
        # node_水果 是 api_node, 没有必要再继续往下了

        if self.search_node.is_api_node():
            self.fill_with_wild_card()
            # if self.machine_state == self.API_REQUEST_RULE_STATE:
            #     if self.search_node.get_field_type(self.requested_slots[0]) == Node.RANGE:
            #         state = self.rule_base_fill(query, self.requested_slots[0])
            self.machine_state = self.API_REQUEST_STATE
            if len(self.requested_slots) == 0:
                self.machine_state = self.API_CALL_STATE
                return
            if self.search_node.get_field_type(self.requested_slots[0]) == Node.RANGE:
                # self.machine_state = self.API_REQUEST_RULE_STATE
                self.rule_base_fill(query, self.requested_slots[0])
                if len(self.requested_slots) == 0:
                    self.machine_state = self.API_CALL_STATE
            for i, value in enumerate(slot_values_list):
                if not self.belief_graph.has_node_by_value(value):
                    continue
                if slot_values_marker[i] == 1:
                    continue
                if self.search_node.value == value:
                    slot_values_marker[i] = 1
                    continue
                if self.search_node.has_child(value):
                    # slot_values_marker[i] = 1
                    slot = self.search_node.get_slot_by_value(value)
                    self.fill_slot(slot, value)

                    if len(self.requested_slots) == 0:
                        self.machine_state = self.API_CALL_STATE
                else:
                    if self.search_node.has_ancestor_by_value(value):
                        # just ignore this stupid input value
                        continue
                    self.move_to_node(self.belief_graph.get_root_node())
                    self.machine_state = self.API_REQUEST_STATE

                    return self.update_belief_graph(slot_values_list=slot_values_list,
                                                    slot_values_marker=slot_values_marker, query=query)
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
            return self.update_belief_graph(slot_values_list=slot_values_list,
                                            slot_values_marker=slot_values_marker, query=query)

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
                return self.update_belief_graph(slot_values_list=slot_values_list,
                                                slot_values_marker=slot_values_marker, query=query)

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
                    return self.update_belief_graph(slot_values_list=slot_values_list,
                                                    slot_values_marker=slot_values_marker, query=query)
                elif len(filtered_nodes) > 1:
                    self.machine_state = self.AMBIGUITY_STATE
                    parent_node_value = set()
                    for node in filtered_nodes:
                        parent_node_value.add(node.parent_node.value)
                    if len(parent_node_value) > 1:
                        for node in filtered_nodes:
                            if node.parent_node.value not in self.ambiguity_slots:
                                self.ambiguity_slots[node.parent_node.value] = []
                            self.ambiguity_slots[node.parent_node.value].append(node)
                    # nice, differentiating at slots, not parent_nodes
                    else:
                        for node in filtered_nodes:
                            self.ambiguity_slots[node.slot] = [node]
                    return
                else:
                    self.machine_state = self.RESET_STATE
                    self.move_to_node(self.belief_graph.get_root_node())
                    return

            # go directly to ROOT
            self.move_to_node(self.belief_graph.get_root_node())
            self.machine_state = self.API_REQUEST_STATE
            return self.update_belief_graph(slot_values_list=slot_values_list,
                                            slot_values_marker=slot_values_marker, query=query)

    def fill_with_wild_card(self):
        for key, value in self.wild_card.items():
            # forcefully, 可以设置隐藏节点
            if self.search_node.has_field(key):
                self.fill_slot(key, value)
        self.wild_card.clear()

    def rule_base_num_retreive(self, query):
        tv_size_dual = r".*(([-+]?\d*\.\d+|\d+)[到|至]([-+]?\d*\.\d+|\d+)寸).*"
        tv_distance_dual = r".*([-+]?\d*\.\d+|\d+)[到|至]([-+]?\d*\.\d+|\d+)米.*"
        ac_power_dual = r".*([-+]?\d*\.\d+|\d+)[到|至]([-+]?\d*\.\d+|\d+)[P|匹].*"
        price_dual = r".*([-+]?\d*\.\d+|\d+)[到|至]([-+]?\d*\.\d+|\d+)[块|元].*"

        tv_size_single = r".*([-+]?\d*\.\d+|\d+)寸.*"
        tv_distance_single = r".*([-+]?\d*\.\d+|\d+)米.*"
        ac_power_single = r".*([-+]?\d*\.\d+|\d+)[P|匹].*"
        price_single = r".*([-+]?\d*\.\d+|\d+)[块|元].*"

        dual = {"tv.size": tv_size_dual, "tv.distance": tv_distance_dual,
                "ac.power": ac_power_dual, "price": price_dual}
        single = {"tv.size": tv_size_single, "tv.distance": tv_distance_single, "ac.power": ac_power_single,
                  "price": price_single}

        query = str(new_cn2arab(query))
        flag = False
        for key, value in dual.items():
            numbers, _ = self.range_extract(value, query, False)
            if numbers:
                flag = True
                self.wild_card[key] = numbers

        for key, value in single.items():
            numbers, _ = self.range_extract(value, query, True)
            if numbers:
                flag = True
                self.wild_card[key] = numbers

        if flag:
            return
        price_dual_default = r"([-+]?\d*\.\d+|\d+)[到|至]([-+]?\d*\.\d+|\d+)"
        price_single_default = r"([-+]?\d*\.\d+|\d+)"
        remove_regex = r"\d+[个|只|条|部|本|台]"
        query = re.sub(remove_regex, '', query)
        numbers, _ = self.range_extract(price_dual_default, query, False)
        if numbers:
            self.wild_card['price'] = numbers
        numbers, _ = self.range_extract(price_single_default, query, True)
        if numbers:
            self.wild_card['price'] = numbers

    def range_extract(self, pattern, query, single):
        numbers = []
        array_numbers = numbers
        match = re.search(pattern, query)
        if single:
            if match:
                numbers = match.group(0)
                numbers = float(re.findall(r"[-+]?\d*\.\d+|\d+", numbers)[0])
                numbers = [numbers * 0.9, numbers * 1.1]
                array_numbers = numbers
                numbers = '[' + str(numbers[0]) + " TO " + \
                    str(numbers[1]) + "]"
        else:
            if match:
                numbers = match.group(0)
                numbers = [float(r) for r in re.findall(
                    r"[-+]?\d*\.\d+|\d+", numbers)[0:2]]
                array_numbers = numbers
                numbers = '[' + str(numbers[0]) + " TO " + \
                    str(numbers[1]) + "]"
        return numbers, array_numbers

    def rule_base_fill(self, query, slot):
        _, wild_card = query_util.rule_base_num_retreive(query)
        self.exploit_wild_card(wild_card, given_slot=slot)

    def is_key_type(self, slot):
        return self.search_node.get_field_type(slot) == Node.KEY

    def solr_facet(self):
        if self.config['solr.facet'] != 'on':
            return ['facet is off'], 0
        node = self.search_node
        fill = []
        facet_field = self.requested_slots[0]

        def render_range(a, gap):
            if len(a) == 0:
                return []
            components = []
            p = 0
            if len(a) == 1:
                components.append(str(a))
            elif len(a) == 2:
                components.append(str(a[0]))
                components.append(str(a[1]))
            else:
                for i in range(1, len(a)):
                    if a[i] - a[i - 1] > gap:
                        if i - p == 1:
                            components.append(str(a[p]))
                        else:
                            components.append(str(a[p]) + "-" + str(a[i - 1]))
                        p = i
                if len(components) == 0:
                    components.append(str(a[1]) + "-" + str(a[-1]))
            render = []
            if len(components) == 1:
                render.append(components[0])
            else:
                for i in range(0, len(components) - 1):
                    if '-'  in components[i + 1]:
                        render.append(components[i])
                        continue
                    if '-' in components[i]:
                        render.append(components[i])
                        continue
                    render.append(components[i] + "-" + components[i + 1])
            return render

        if self.is_key_type(facet_field):
            params = {
                'q': '*:*',
                'facet': True,
                'facet.field': facet_field,
                "facet.mincount": 1
            }
            mapper = dict()
            for key, value in self.filling_slots.items():
                # fill.append(key + ":" + str(value))
                mapper[key] = value
            while node.value != self.belief_graph.ROOT:
                if node.slot.startswith('virtual'):
                    node = node.parent_node
                    continue
                # fill.append(node.slot + ":" + node.value)
                mapper[node.slot] = node.value
                node = node.parent_node
            params['fq'] = solr_util.compose_fq(mapper)
            res = self.solr.query('category', params)
            facets = res.get_facet_keys_as_list(facet_field)
            return res.get_facet_keys_as_list(facet_field), len(facets)
        else:
            start = 1
            gap = 1
            end = 100
            # use facet.range
            if facet_field == "price":
                start = 100
                gap = 3000
                end = 30000
            if facet_field == 'ac.power_float':
                start = 1
                gap = 0.5
                end = 10
            params = {
                'q': '*:*',
                'facet': True,
                'facet.range': facet_field,
                "facet.mincount": 1,
                'facet.range.start': start,
                'facet.range.end': end,
                'facet.range.gap': gap
            }
            mapper = dict()
            for key, value in self.filling_slots.items():
                # fill.append(key + ":" + str(value))
                mapper[key] = value
            while node.value != self.belief_graph.ROOT:
                if node.slot.startswith('virtual'):
                    node = node.parent_node
                    continue
                # fill.append(node.slot + ":" + node.value)
                mapper[node.slot] = node.value
                node = node.parent_node
            params['fq'] = solr_util.compose_fq(mapper)
            res = self.solr.query('category', params)
            ranges = res.get_facets_ranges()[facet_field].keys()
            ranges = [float("{0:.1f}".format(float(r))) for r in ranges]
            # now render the result
            facet = render_range(ranges, gap)
            return facet, len(ranges)

    def issue_class(self):
        if self.machine_state == self.TRAVEL_STATE:
            return "api_greeting_search_normal"
        if self.machine_state in [self.API_REQUEST_STATE, self.API_REQUEST_RULE_STATE]:
            fill = []
            for key, value in self.filling_slots.items():
                fill.append(key + ":" + str(value))
            fill.append(self.search_node.slot + ":" + self.search_node.value)
            return "api_call_slot_" + ','.join(fill)
        if self.machine_state == self.AMBIGUITY_STATE:
            param = ','.join(self.ambiguity_slots.keys())
            return "api_request_ambiguity_removal_" + param
        if self.machine_state == self.API_CALL_STATE:
            # first filling slots
            param = "api_call_search_"
            fill = []
            for key, value in self.filling_slots.items():
                fill.append(key + ":" + str(value))

            node = self.search_node
            while node.value != self.belief_graph.ROOT:
                fill.append(node.slot + ":" + node.value)
                node = node.parent_node
            return param + ",".join(fill)
        if self.machine_state == self.RESET_STATE:
            return "reset_requested"

    def issue_api(self, attend_facet=True):
        if self.machine_state == self.TRAVEL_STATE:
            return "api_greeting_search_normal", []
        if self.machine_state in [self.API_REQUEST_STATE, self.API_REQUEST_RULE_STATE]:
            slot = self.requested_slots[0]
            avails, num_avails = self.solr_facet()
            self.avails.clear()
            self.avails[slot] = avails
            if attend_facet:
                if num_avails == 1:
                    self.fill_slot(slot, avails[0])
                    if len(self.requested_slots) == 0:
                        self.machine_state = self.API_CALL_STATE
                    return self.issue_api()
                # if len(avails) == 0 and Node.KEY == self.belief_graph.get_field_type(slot):
                #     return 'api_call_nonexist_' + slot, []
            return "api_call_request_" + slot, avails
        if self.machine_state == self.AMBIGUITY_STATE:
            param = ','.join(self.ambiguity_slots.keys())
            return "api_call_request_ambiguity_removal_" + param, []
        if self.machine_state == self.API_CALL_STATE:
            # first filling slots
            param = "api_call_search_"
            fill = []
            for key, value in self.filling_slots.items():
                fill.append(key + ":" + str(value))
            # node = self.search_node
            # while node.value != self.belief_graph.ROOT:
            #     if node.slot != 'virtual_category':
            #         fill.append(node.slot + ":" + node.value)
            #     node = node.parent_node
            return param + ",".join(fill), []
        if self.machine_state == self.RESET_STATE:
            return "reset_requested", []

    def r_walk_with_pointer_with_clf(self, query):
        query = str(query)
        if not query:
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
        tokens = jieba_cut(q)
        # slot_values_list = q.split(",")

        slot_values_list = []
        for t in tokens:
            if self.belief_graph.has_node_by_value(t):
                slot_values_list.append(t)
            if self.machine_state == self.AMBIGUITY_STATE:
                if self.belief_graph.has_slot(t):
                    slot_values_list.append(t)
        slot_values_list = list(set(slot_values_list))
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
    # query = '空调'
    #
    # metadata_dir = os.path.join(
    #     grandfatherdir, 'data/memn2n/processed/metadata.pkl')
    # data_dir = os.path.join(
    #     grandfatherdir, 'data/memn2n/processed/data.pkl')
    # ckpt_dir = os.path.join(grandfatherdir, 'model/memn2n/ckpt')
    #
    # memInfer = MemInfer(metadata_dir, data_dir, ckpt_dir)
    # sess = memInfer.getSession()
    #
    # reply = sess.reply(query)
    # print(reply)

    graph_dir = os.path.join(grandfatherdir, "model/graph/belief_graph.pkl")
    config = dict()
    config['belief_graph'] = graph_dir
    config['solr.facet'] = 'off'
    # memory_dir = os.path.join(grandfatherdir, "model/memn2n/ckpt")
    log_dir = os.path.join(grandfatherdir, "log/test2.log")
    bt = BeliefTracker(config)

    with open(log_dir, 'a', encoding='utf-8') as logfile:
        while(True):
            try:
                ipt = input("input:")
                print(ipt, file=logfile)
                resp = bt.memory_kernel(ipt)
                print(resp)
                print(resp, file=logfile)
            except Exception as e:
                traceback.print_exc()
                print('error:', e, end='\n\n', file=logfile)
                break

def test_facet():
    graph_dir = os.path.join(grandfatherdir, "model/graph/belief_graph.pkl")
    config = dict()
    config['belief_graph'] = graph_dir
    config['solr.facet'] = 'on'
    # memory_dir = os.path.join(grandfatherdir, "model/memn2n/ckpt")
    bt = BeliefTracker(config)
    bt.search_node = bt.belief_graph.get_nodes_by_value('空调')[0]
    bt.requested_slots = ['ac.power_float']
    bt.filling_slots = {"brand":"美的","price":"[2700 TO 3300]"}
    print(bt.solr_facet())

if __name__ == "__main__":
    test_facet()
