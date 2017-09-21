""" A neural chatbot using sequence to sequence model with
attentional decoder.

This is based on Google Translate Tensorflow model
https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/

Sequence to sequence model by Cho et al.(2014)

Created by Chip Huyen as the starter code for assignment 3,
class CS 20SI: "TensorFlow for Deep Learning Research"
cs20si.stanford.edu

This file contains the code to run the model.

See readme.md for instruction on how to run the starter code.

This implementation learns NUMBER SORTING via seq2seq. Number range: 0,1,2,3,4,5,EOS

https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf

See README.md to learn what this code has done!

Also SEE https://stackoverflow.com/questions/38241410/tensorflow-remember-lstm-state-for-next-batch-stateful-lstm
for special treatment for this code

Belief Tracker
"""
import sys
sys.path.insert(0, '..')

import cPickle as pickle

import kernel.memory_network.MemoryNetwork
from graph.graph import Graph
from graph.node import Node
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

    def __init__(self, graph_path, clf_path):
        self.gbdt = None
        self.state_cleared = True
        self._load_graph(graph_path)
        self._load_clf(clf_path)
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
        next_scene, inside_intentions, response = self.r_walk_with_pointer_with_clf(
            query=query)
        return next_scene, inside_intentions, response

    def clear_state(self):
        self.search_graph = BeliefGraph()
        self.remaining_slots = {}
        self.negative_slots = {}
        self.negative = False

    def _load_clf(self, path):
        if not BeliefTracker.static_gbdt:
            try:
                print('attaching gbdt classifier...100%')
                with open(path, "rb") as input_file:
                    self.gbdt = pickle.load(input_file)
                    BeliefTracker.static_gbdt = self.gbdt
                    # self.gbdt = Multilabel_Clf.load(path)
            except Exception, e:
                print('failed to attach main kernel...detaching...', e.message)
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
                    # print(self.graph.go('购物', Node.REGEX))
            except:
                print('failed to attach logic graph...detaching...')
        else:
            print('skipping attaching logic graph...')
            self.belief_graph = BeliefTracker.static_belief_graph

    def travel_with_clf(self, query):
        """
        the memory network predictor has mainly two types of classes: api_call_... and slots_...
        """
        filtered_slots_list = []
        try:
            # flipped, self.negative = self.negative_clf.predict(input_=query)
            intention, is_api_call, prob = self.rnn_clf(query)
            if is_api_call:
                self.api_call = intention
                return True
            filtered_slots_list = self.unfold_slots_list(intention)
        except:
            traceback.print_exc()
            return False

        # build belief graph
        # self.update_remaining_slots(expire=True)
        # filtered_slots_list = self.inter_fix(filtered_slots_list)
        # self.should_expire_all_slots(filtered_slots_list)
        self.update_belief_graph(
            search_parent_node=self.search_graph, slots_list=filtered_slots_list)
        return True

    def sort_slots_list_by_level(self, slot_value_list):
        """
        unfold nodes sharing the same name/slot_value, bottom to top
        """
        if Graph.ROOT not in slots_list:
            slot_value_list.append(Graph.ROOT)
        node_list = []
        for slot_value in slot_value_list:
            header_list = self.belief_graph.get_nodes_by_value(slot_value)
            node_list.extend(header_list)
        node_list.sort(key=lambda x: x.level, reverse=True)
        return node_list

    def unfold_slots_list(self, intention):
    """
    api_call, slots_ We unfold slots from slots_
    """
        slots = "_".join(intention.split("_")[1:-1]).split(",")
        return slots

    def retrieve_intention_from_solr(self, q):
        tokens = QueryUtils.static_jieba_cut(
            query=q, smart=False, remove_single=True)
        url = "http://localhost:11403/solr/sc_sale_gen/select?q.op=OR&defType=edismax&wt=json&q=intention:({0})"
        append = "%20".join(tokens)
        request_url = url.format(append)
        r = requests.get(request_url)
        if SolrUtils.num_answer(r) > 0:
            intentions = self._get_response(
                r=r, key='intention', random_hit=False, random_field=False, keep_array=True)
            return intentions
        return []

    def should_expire_all_slots(self, slots_list):
        slots_list = list(slots_list)
        if len(slots_list) == 1:
            slot = slots_list[0]
            if self.belief_graph.has_child(slot, Node.KEY) \
                    and self.belief_graph.slot_identities[slot] == 'intention':
                self.remaining_slots.clear()
                self.negative_slots.clear()
                self.search_graph = BeliefGraph()

    def move_to_node(self, node):
        self.search_node = node
        self.required_slots = search_node.generate_required_slots()

    def update_belief_graph(self, ambiguity_nodes=[], slot_values_list, slot_values_marker=None):
        """
        1. if node is single, go directly
        2. if there are ambiguity nodes, try remove ambiguity resorting to slot_values_list AND search_parent_node
            2.1 if fail, enter ambiguity state
        """
        if not slot_values_marker:
            slot_values_marker = [0] * len(slots_list)
        slot_values_list = list(set(slot_values_list))

        if len(ambiguity_nodes) > 1:
            # now there are ambiguity nodes
            # build profiles of ambiguity, first use information from slot_values_list to remove ambiguity
            for node in search_parent_nodes:
                parent_values = node.get_parent_values()
                self.ambiguity_slots.clear()
                self.ambiguity_slots[parent_values[0]] = node
                for slot_value in slot_values_list:
                    if slot_values_marker[i] == 1:
                        continue
                    if slot_value in parent_values:
                        # found
                        self.move_to_node(node)
                        return self.update_belief_graph(
                            next_parent_search_node, slot_values_list, slot_values_marker)

            # fail to remove ambiguity, enter ambiguity state
            self.machine_state = self.AMBIGUITY_STATE
            return

        # session terminal end api_call_node
        # issune api_call_to_solr
        # node_水果 是 api_node, 没有必要再继续往下了
        if self.search_node.is_api_node():
            self.machine_state = self.API_REQUEST_STATE
            for i, value in enumerate(slot_values_list):
                if slot_values_marker[i] == 1:
                    continue
                if self.search_node.has_child(value):
                    slot = search_parent_node.get_slot_by_value(value)
                    self.filling_slots[slot] = value
                    self.required_slots.remove(slot)

                    if len(self.required_slots) == 0:
                        self.machine_state = self.API_CALL_STATE
            return

        for i, value in enumerate(slot_values_list):
            if slot_values_marker[i] == 1:
                continue
            candidate_nodes = self.search_node.get_posterity_nodes_by_value(
                value)
            # consider single node first
            if len(candidate_nodes) == 1:
                self.slot_values_marker[i] == 1
                next_parent_search_node = candidate_nodes[0]
                self.move_to_node(next_parent_search_node)
                self.machine_state = self.API_REQUEST_STATE
                return self.update_belief_graph(
                    slot_values_list=slot_values_list, slot_values_marker=slot_values_marker)

            # enter ambiguity state
            if len(candidate_nodes) > 1:
                self.slot_values_marker[i] == 1
                # search_node stays
                return self.update_belief_graph(
                    ambiguity_nodes=candidate_nodes,
                    slot_values_list=slot_values_list, slot_values_marker=slot_values_marker)

            # go directly to ROOT
            self.move_to_node(self.belief_graph.get_root_node())
            self.machine_state = self.API_REQUEST_STATE
            return self.update_belief_graph(
                slot_values_list=slot_values_list, slot_values_marker=slot_values_marker)

    def issune_api():
        if self.machine_state == self.API_REQUEST_STATE:
            slot = self.required_slots[0]
            return "api_request_value_of_" + slot
        if self.machine_state == self.AMBIGUITY_STATE:
            param = ','.join(self.ambugity_slots.keys())
            return "api_request_ambiguity_removal_" + param
        if self.machine_state == self.API_CALL_STATE:
            # first filling slots
            param = "api_call_"
            for key, value in self.filling_slots:
                param += key + ":" + value

            node = self.search_node
            while node.value != self.belief_graph.ROOT:
                param += node.slot + ":" + node.value
            return param

    def update_remaining_slots(self, slot=None, expire=False):
        if expire:
            for remaining_slot, index in self.remaining_slots.iteritems():
                self.remaining_slots[remaining_slot] = index - 1
                # special clear this slot once
                if remaining_slot in ['随便']:
                    self.remaining_slots[remaining_slot] = -1
            self.remaining_slots = {
                k: v for k, v in self.remaining_slots.iteritems() if v >= 0}
        if slot:
            if self.negative:
                self.negative_slots[slot] = True
            else:
                self.negative_slots[slot] = False
            self.remaining_slots[slot] = len(self.score_stairs) - 1

    def r_walk_with_pointer_with_clf(self, query):
        if not query or len(query) == 1:
            return None, 'invalid query', ''
        sucess = self.travel_with_clf(query)
        if not sucess:
            return 'sale', 'invalid_query', ''
        return self.search()

    def single_last_slot(self, split=' OR '):
        return self.single_slot(self.last_slots, split=split)

    def remove_slots(self, key):
        new_remaining_slots = {}
        for remaining_slot, index in self.remaining_slots.iteritems():
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
        for slot, i in self.remaining_slots.iteritems():
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
        try:
            rnn_url = "http://localhost:10001/sc/rnn/classify?q={0}".format(q)
            r = requests.get(rnn_url)
            text = r.text
            if text:
                slots_list = text.split(",")
                probs = [1.0 for slot in slots_list]
                return slots_list, probs
            else:
                return None, None
        except:
            return None, None

    def search(self):
        try:
            intentions, fq = self.compose()
            if len(self.remaining_slots) == 0:
                return 'base', 'empty', None
            if 'facility' in self.remaining_slots or 'entertainment' in self.remaining_slots:
                qa_intentions = ','.join(self.remaining_slots)
                self.remove_slots('facility')
                self.remove_slots('entertainment')
                _, response = self.simple.kernel(qa_intentions)
                return None, ','.join(intentions), response
            url = self.guide_url + "&q=intention:(%s)" % fq
            cn_util.print_cn("gbdt_result_url", url)
            r = requests.get(url)
            if SolrUtils.num_answer(r) > 0:
                response = self._get_response(r, 'answer', random_hit=self.contain_negative(
                    intentions), random_field=True, keep_array=False)
                labels = self._get_response(
                    r, 'intention', random_field=True, keep_array=True)
                remove_labels = [u"购物", u"吃饭", u"随便"]
                for rl in remove_labels:
                    if rl in labels:
                        labels.remove(rl)
                response = self.generate_response(response, labels)
                return None, ','.join(intentions), response
        except:
            traceback.print_exc()
            return 'base', 'error', '我好像不知道哦, 问问咨询台呢'

    def generate_response(self, response, labels):
        graph_url = 'http://localhost:11403/solr/graph_v2/select?wt=json&q=%s'
        if '<s>' in response:
            condition = []
            for label in labels:
                try:
                    string = 'label:%s' % (
                        label + '^' + str(self.belief_graph.slot_importances[label.decode('utf-8')]))
                except:
                    string = 'label:%s' % label
                condition.append(string)
            condition = '%20OR%20'.join(condition)
            url = graph_url % condition
            cn_util.print_cn(url)
            r = requests.get(url)
            if SolrUtils.num_answer(r) > 0:
                name = self._get_response(
                    r=r, key='name', random_hit=True, random_field=True)
                location = self._get_response(
                    r=r, key='rich_location', random_hit=True, random_field=True)
                new_response = response.replace(
                    '<s>', name).replace('<l>', location)
                return new_response
            else:
                return '没有找到相关商家哦.您的需求有点特别哦.或者不在知识库范围内...'
        else:
            return response

    def _get_response(self, r, key, random_hit=False, random_field=True, keep_array=False):
        try:
            num = np.minimum(SolrUtils.num_answer(r), 3)
            if random_hit:
                hit = np.random.randint(0, num)
            else:
                hit = 0
            a = r.json()["response"]["docs"][hit][key]
            if keep_array:
                return a
            else:
                if random_field:
                    rr = np.random.choice(a, 1)[0]
                else:
                    rr = ','.join(a)
            return rr.decode('utf8')
        except:
            return None

    def should_clear_state(self, multi_slots):
        try:
            single_slot = self.single_slot(multi_slots)
            node = self.graph.get_global_node(single_slot)
            if node.is_leaf(Node.REGEX):
                self.clear_state()
        except:
            self.clear_state()


if __name__ == "__main__":
    bt = BeliefTracker("../model/sc/belief_graph.pkl",
                       '../model/sc/belief_clf.pkl')
    ipts = [u"不购物"]
    for ipt in ipts:
        # ipt = raw_input()
        # chinese comma
        # bt.travel_with_clf(ipt)
        cn_util.print_cn(",".join(bt.kernel(ipt)[1:-1]))
        # cn_util.print_cn(bt.compose()[0])
