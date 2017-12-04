import sys
import os

import numpy as np

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)

from kernel_2.belief_tracker import BeliefTracker
from graph.belief_graph import Graph
import memory.config as m_config
from utils.translator import Translator

grandfatherdir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parentdir)
sys.path.append(grandfatherdir)

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, parentdir)

translator = Translator()


class TreeSimilator:
    def __init__(self, config):
        self.belief_tracker = BeliefTracker(config)
        self._load_template(config)

    def _load_template(self, config):
        template_file = config['template']
        self.field_trans = dict()
        self.quantum = dict()
        self.unit = dict()
        self.thesaurus = dict()
        with open(template_file, 'r') as tf:
            for line in tf:
                line = line.strip('\n')
                parsers = line.split('|')
                if parsers[0] == 'prefix_buy':
                    self.prefix_buy = parsers[1].split('/')
                if parsers[0] == 'prefix_buy_root':
                    self.prefix_buy_root = parsers[1].split('/')
                if parsers[0] == 'prefix_deny':
                    self.prefix_deny = parsers[1].split('/')
                if parsers[0] == 'modifier_query_cateory_location':
                    self.modifier_query_cateory_location = parsers[1].split('/')
                if parsers[0] == 'prefix_price':
                    self.prefix_price = parsers[1].split('/')
                if parsers[0] == 'prefix_brand':
                    self.prefix_brand = parsers[1].split('/')
                if parsers[0] == 'postfix_price':
                    self.postfix_price = parsers[1].split('/')
                if parsers[0] == 'postfix_brand':
                    self.postfix_brand = parsers[1].split('/')
                if parsers[0] == 'postfix':
                    self.postfix = parsers[1].split('/')
                if parsers[0] == 'field_trans':
                    self.field_trans[parsers[1]] = parsers[2].split('/')
                if parsers[0] == 'quantum':
                    self.quantum[parsers[1]] = parsers[2].split('/')
                if parsers[0] == 'unit':
                    self.unit[parsers[1]] = parsers[2].split('/')
                if parsers[0] == 'thesaurus':
                    self.thesaurus[parsers[1]] = parsers[2].split('/')[:] + [parsers[1]]


    def gen_sessions(self, output_files):
        """
        :param belief_tracker:
        :param output_files: candidate_files, train, val, test
        :return:
        """
        belief_graph = self.belief_tracker.belief_graph
        """
        主要逻辑
        :return:
        """
        def gen_random_slot_values(required_field):
            slot_values_mapper = dict()
            num_rnd_external_max = 0
            if required_field == 'virtual_category':
                nodes = belief_graph.get_nodes_by_slot(required_field)
                # choose one
                node = np.random.choice(nodes)
                slot_values_mapper[node.slot] = node.value
                return slot_values_mapper
            if required_field == 'category':
                if self.belief_tracker.search_node == belief_graph:
                    nodes = belief_graph.get_nodes_by_slot(required_field)
                    node = np.random.choice(nodes)
                else:
                    children_names = self.belief_tracker.search_node.get_children_names_by_slot(
                        required_field)
                    # choose one
                    name = np.random.choice(children_names)
                    slot = self.belief_tracker.search_node.get_slot_by_value(
                        name)
                    slot_values_mapper[slot] = name
                    node = self.belief_tracker.search_node.get_node_by_value(
                        name)
                slot_values_mapper[node.slot] = node.value
                fields = list(node.fields.keys())

                # print(fields)

                if 'ac.power' in fields:
                    fields.remove('ac.power')
                    fields.append('ac.power_float')
                n = np.random.randint(
                    0, np.min([len(fields), num_rnd_external_max]) + 1)
                # fields = list(belief_tracker.requested_slots)
                if fields:
                    picked_fields = np.random.choice(fields, n).tolist()
                else:
                    picked_fields = []
                # picked_fields = set()
                # for i in range(n):
                #     if np.random.uniform() < 0.25 and 'brand' in fields:
                #         picked_fields.add('brand')
                #         continue
                #     if 'brand' in picked_fields and np.random.uniform() < 0.1:
                #         picked_fields.add(np.random.choice(fields))
                for f in picked_fields:
                    value = random_property_value(f, node)
                    if not value:
                        continue
                    # weird
                    if value != 'range' and belief_graph.is_entity_value(value):
                        slot_values_mapper['entity'] = value
                    else:
                        slot_values_mapper[f] = value
                return slot_values_mapper

            my_search_node = self.belief_tracker.search_node

            fields = list(self.belief_tracker.requested_slots)
            n = np.random.randint(
                0, np.min([len(fields), num_rnd_external_max]) + 1)

            picked_fields = np.random.choice(fields, n).tolist()
            # append required fields
            picked_fields.append(required_field)
            picked_fields = set(picked_fields)
            for f in picked_fields:
                value = random_property_value(f, my_search_node)
                # weird
                if value != 'range' and belief_graph.is_entity_value(value):
                    slot_values_mapper['entity'] = value
                else:
                    slot_values_mapper[f] = value
            return slot_values_mapper

        def gen_ambiguity_response(availables):
            availables = availables[0].replace(
                'api_call_request_ambiguity_removal_', '').split(",")
            pick = np.random.choice(availables)
            slot_values_mapper = dict()
            num_rnd_external_max = 1
            if len(belief_graph.get_nodes_by_value(pick)) > 0:
                slot_values_mapper[belief_graph.get_nodes_by_value(pick)[
                    0].slot] = pick
            else:
                slot_values_mapper['ambiguity_removal'] = belief_graph.slots_trans[pick]
            # my_search_node = belief_tracker.ambiguity_slots[pick].parent_node
            #
            # fields = list(my_search_node.fields.keys())
            # n = np.random.randint(0, np.min([len(fields), num_rnd_external_max]) + 1)
            #
            # picked_fields = np.random.choice(fields, n).tolist()
            # # append required fields
            # # picked_fields.append(required_field)
            # for f in picked_fields:
            #     value = random_property_value(f, my_search_node)
            #     slot_values_mapper[f] = value
            return slot_values_mapper

        def render_rhetorical(slot):
            """
            'user': {
                'brand': ['你们这都有什么牌子？', '品牌都有哪些？'],
                'category': ['你们这儿都卖什么种类？', '种类都有哪些？'],
                'price': ['都有哪些价格？', '都有什么价啊？']
            },
            :param slot:
            :return:
            """
            template = ["你们这都有什么<fill>", "<fill>都有哪些",
                        "你们这儿都卖什么<fill>", '你有什么<fill>', '你有哪些<fill>']
            trans = belief_graph.slots_trans[slot]
            t = np.random.choice(template)
            if np.random.uniform() < 0.5:
                t = t.replace("<fill>", trans)
            else:
                t = t.replace("<fill>", "")
            cls = "api_call_rhetorical_" + slot
            return t, cls

        def gen_ambiguity_initial():
            slot_values_mapper = dict()
            nodes_value_list = list(
                self.belief_tracker.belief_graph.node_header.keys())
            property_flag = False
            while True:
                key = np.random.choice(nodes_value_list)
                nodes = self.belief_tracker.belief_graph.get_nodes_by_value(
                    key)
                if len(nodes) == 1 or not belief_graph.is_entity_value(key):
                    node = nodes[0]
                    if node.slot == 'category' or node.slot == 'virtual_category':
                        continue
                    property_flag = True  # other property must identity category
                    slot_values_mapper[node.slot] = node.value
                    break
                slot_values_mapper['entity'] = key
                break
            if np.random.uniform() < 0.25 or property_flag:
                node = np.random.choice(nodes)
                slot_values_mapper[node.parent_node.slot] = node.parent_node.value
            # slot_values_mapper.clear()
            # slot_values_mapper['entity'] = '4G'
            return slot_values_mapper

        def random_property_value(field, search_node):
            if field == 'price':
                value = 'range'
                # if np.random.uniform() < 0.3:
                #     if np.random.uniform() < 0.5:
                #         value += '元'
                #     else:
                #         value += '块'

                return value

            # if field == 'ac.power':
            #     value = 'range'
            #
            #     if np.random.uniform() < 0.5:
            #         value += 'P'
            #     else:
            #         value += '匹'
            #
            #     return value
            if field == 'ambiguity_removal':
                value = np.random.choice(
                    list(self.belief_tracker.ambiguity_slots.keys()))
                # nodes = belief_graph.get_nodes_by_value(value)
                # if len(nodes) > 0:
                #     slot = nodes[0].slot
                #     del slot_values_mapper[field]
                #     slot_values_mapper[slot] = value
                return value

            if belief_graph.get_field_type(field) == 'range':
                # return search_node.get_node_slot_trans(field) + 'range'
                return 'range'
            else:
                children_names = search_node.get_children_names()
                randoms = np.random.choice(children_names)
                return randoms

        def get_avail_brands(category):
            category_node = belief_graph.get_nodes_by_value(category)[0]
            brands = category_node.get_children_names_by_slot('brand')
            return brands

        def get_requested_field():
            requested = np.random.choice(
                ['virtual_category', 'category', 'property', 'ambiguity_removal'], p=[0, 1, 0, 0])
            return requested

        def render_thesaurus(v):
            if v in self.thesaurus:
                v = np.random.choice(self.thesaurus[v])
            return v

        def render_lang(slot_values_mapper, fresh):

            def render_template(key, replace):
                append = np.random.choice(self.field_trans[key])\
                    .replace('<{}>'.format(key), replace)
                return append

            search_node = self.belief_tracker.search_node
            prefix = self.prefix_buy
            postfix = self.postfix
            lang = np.random.choice(prefix)
            if 'brand' in slot_values_mapper:
                lang += render_template('brand', slot_values_mapper['brand'])
            if 'price' in slot_values_mapper:
                unit = np.random.choice(self.unit['price'])
                with_unit = slot_values_mapper['price'] + unit
                lang += render_template('price', with_unit)
            if 'category' in slot_values_mapper:
                v = slot_values_mapper['category']
                quantum = ''
                syn = v
                if v in self.quantum:
                    quantum = np.random.choice(self.quantum[v])
                if v in self.thesaurus:
                    syn = np.random.choice(self.thesaurus[v])
                v = quantum + syn
                lang += v + ","

            for k, v in slot_values_mapper.items():
                if k in ['brand', 'price', 'category']:
                    continue
                if k not in ['entity', 'ambiguity_removal'] and belief_graph.get_field_type(k) == 'range':
                    trans = search_node.get_node_slot_trans(k)
                    if fresh or 'range' in lang:
                        lang += trans + 'range'
                        unit = np.random.choice(self.unit[k])
                        lang += unit
                    else:
                        lang += trans + 'range'
                        unit = np.random.choice(self.unit[k])
                        lang += unit
                else:
                    quantum = ''
                    syn = v
                    if v in self.quantum:
                        quantum = np.random.choice(self.quantum['v'])
                    if v in self.thesaurus:
                        syn = np.random.choice(self.thesaurus[v])
                    v = quantum + syn
                    lang += v + ","

            if lang[-1] == ',':
                lang = lang[0:-1]
            lang = lang + np.random.choice(postfix)
            lang = lang.lower()
            if 'root' in lang:
                lang = np.random.choice(self.prefix_buy_root)
            return lang

        def render_cls(slot_values_mapper):
            params = []
            for key in sorted(slot_values_mapper.keys()):
                params.append(key + ":" + slot_values_mapper[key])
            line = 'api_call_slot_' + ','.join(params)
            if line == 'api_call_slot_virtual_category:root':
                line = 'api_call_query_general'
            return line

        def render_api(api):
            return api[0]

        def render_deny():
            prefix = np.random.choice(self.prefix_deny)
            lang = prefix
            cls = 'api_call_deny_all'
            if 'brand' in self.belief_tracker.filling_slots:
                if np.random.uniform() < 0.7:
                    lang += self.belief_tracker.filling_slots['brand']
                    cls = 'api_call_deny_brand'
            return lang, cls, 'placeholder'

        requested = get_requested_field()
        i = 0

        # thesaurus = dict()
        # with open('../../data/gen_product/thesaurus.txt', 'r') as f:
        #     for line in f:
        #         line = line.strip('\n')
        #         key, value = line.split('#')
        #         thesaurus[key] = value.split(',')

        candidates = set()
        api_set = set()
        train_set = []
        test_set = []
        val_set = []
        train_gbdt = set()

        container = []
        flow_container = []
        single_container = []
        duplicate_removal = set()
        flow_removal = set()
        flows = list()
        mapper = {'train': train_set, 'val': val_set, 'test': test_set}
        which = np.random.choice(['train', 'val', 'test'], p=[0.8, 0.1, 0.1])
        fresh = True

        mlt_container = []
        mlt_candidates = []

        with_multiple = False
        with_qa_location = False
        with_map_location = True
        with_qa_price = False
        with_deny = False
        with_whatever = False
        with_flow = True
        with_base = True
        with_gbdt = False
        with_main = True
        with_faq = False
        with_single = True
        qa_prob = 0.25
        with_ad = 0.75
        ad_cls = 'api_call_request_category:注册'
        while 1:
            if requested == 'property':
                slot_values_mapper = gen_ambiguity_initial()
            elif requested == 'ambiguity_removal':
                slot_values_mapper = gen_ambiguity_response(
                    self.belief_tracker.issue_api(attend_facet=False))
            else:
                slot_values_mapper = gen_random_slot_values(
                    required_field=requested)
                self.belief_tracker.color_graph(slot_values_mapper=slot_values_mapper, range_render=False)
            user_reply = render_lang(slot_values_mapper, fresh)

            if not fresh:
                gbdt = 'plugin:' + 'api_call_slot' + '|'\
                    + '|'.join([key + ":" + value for key, value in slot_values_mapper.items()])\
                    + '#' + requested + '$' + user_reply
            else:
                gbdt = 'plugin:' + 'api_call_slot' + '|'\
                    + '|'.join([key + ":" + value for key, value in slot_values_mapper.items()]) \
                       + '#' + user_reply

            gbdt = gbdt.lower()
            train_gbdt.add(gbdt)

            fresh = False
            cls = render_cls(slot_values_mapper)
            candidates.add(cls.lower())
            api = render_api(self.belief_tracker.issue_api(attend_facet=False))
            if not user_reply:
                print()
            line = user_reply + '\t' + cls + '\t' + api
            # print(line)
            flow = cls + '\t' + api
            # if cls == 'api_call_slot_reg.repeat_scan_follow:扫码关注失败':
            #     print()
            if requested == 'category':
                single_container.append(line)
                # single_container.append('')
            requested = self.belief_tracker.get_requested_field()
            trans_api = translator.en2cn(api)
            if not api.startswith('api_call_search'):
                api_set.add(api + '##' + trans_api)
            container.append(line.lower())
            flow_container.append(flow.lower())
            if api.startswith('api_call_search'):
                if np.random.uniform() < 0.4 and with_deny:
                    a, b, c = render_deny()
                    candidates.add(b)
                    container.append('\t'.join([a, b, c]))
            mlt_line = user_reply + '\t' + 'plugin:apl_call_slot,' +\
                '|'.join([key + ":" + value for key, value in slot_values_mapper.items()])\
                + '\t' + api
            mlt_container.append(mlt_line)

            if with_qa_location and requested:
                filling_slots = self.belief_tracker.filling_slots
                if 'category' in filling_slots:
                    if np.random.uniform() < qa_prob:
                        category = np.random.choice([render_thesaurus(
                            self.belief_tracker.belief_graph.slots_trans[requested]), ''])
                        if np.random.uniform() < 0.5:
                            qa = category\
                                + np.random.choice(self.modifier_query_cateory_location)
                        else:
                            qa = np.random.choice(self.modifier_query_cateory_location) \
                                 + category
                        line = qa + '\t' + 'api_call_query_location_' + '{}:'.format(requested)\
                            + self.belief_tracker.belief_graph.slots_trans[requested] + '\t' + 'placeholder'
                        container.append(line)
                        # flow_container.append(flow.lower())
                        candidates.add('api_call_query_location_' + '{}:'.format(requested)\
                                       + self.belief_tracker.belief_graph.slots_trans[requested])
                        if category:
                            single_container.append(line)
                            # single_container.append('')

            if requested and requested != 'ambiguity_removal':
                if np.random.uniform() < 0:
                    reh, cls = render_rhetorical(requested)
                    rhetorical = "plugin:" + cls + '#' + requested + "$" + reh
                    memory_line = reh + '\t' + cls + '\t' + 'placeholder'
                    flow = cls + '\t' + 'placeholder'
                    cls = cls.lower()
                    candidates.add(cls)
                    memory_line = memory_line.lower()
                    container.append(memory_line)
                    # flow_container.append(flow.lower())
                    train_gbdt.add(rhetorical.lower())
            # print(line)
            if not requested:
                fresh = True
                requested = get_requested_field()
                self.belief_tracker.clear_memory()
                line = ''
                container.append(line.lower())
                flow_container.append(line.lower())
                # check duplicate
                bulk = '#'.join(container).lower()
                single_bulk = '#'.join(single_container).lower()
                if bulk not in duplicate_removal:
                    duplicate_removal.add(bulk)
                    if with_main:
                        mapper[which].extend(container)
                    # for a in container:
                    #     print(a)
                else:
                    print('# duplicate #')
                if single_bulk not in duplicate_removal:
                    duplicate_removal.add(single_bulk)
                    if with_single:
                        mapper[which].extend(single_container * 4)

                flow_bulk = '#'.join(flow_container).lower()
                if flow_bulk not in flow_removal:
                    flow_removal.add(flow_bulk)
                    flows.extend(flow_container)
                which = np.random.choice(
                    ['train', 'val', 'test'], p=[0.8, 0.1, 0.1])
                container = []
                if np.random.uniform() < 0.5:
                    single_container = [""]
                flow_container = []
                # print(line)
                i += 1
                print(i)


                if i >= 20000:

                    break

        # lower everything

        # print('writing', len(train_set), len(
        #     val_set), len(test_set), len(candidates))
        #

        if with_flow:
            with open(grandfatherdir + '/data/memn2n/train/tree/origin/flow.txt', 'w', encoding='utf-8') as f:
                for line in flows:
                    f.writelines(line + '\n')

        base_count = 0
        base = []
        if with_base:
            with open(grandfatherdir + '/data/memn2n/train/base/interactive_memory.txt', encoding='utf-8') as cf:
                for line in cf:
                    line = line.strip('\n')
                    base.append(line)
                    if not line:
                        base_count += 1
            with open(grandfatherdir + '/data/memn2n/train/base/base_bookstore.txt',
                      encoding='utf-8') as cf:
                for line in cf:
                    line = line.strip('\n')
                    base.append(line)
                    if not line:
                        base_count += 1

        # whatever = []
        if with_whatever:
            with open(grandfatherdir + '/data/memn2n/train/base/whatever.txt', encoding='utf-8') as cf:
                for line in cf:
                    line = line.strip('\n')
                    base.append(line)
                    if not line:
                        base_count += 1

        # whatever = []
        map_count = 0
        if with_map_location:
            with open(grandfatherdir + '/data/memn2n/train/map/map.txt', encoding='utf-8') as cf:
                for line in cf:
                    line = line.strip('\n')
                    if line:
                        candidate = line.split('\t')[1]
                    candidates.add(candidate)
                    base.append(line)
                    if not line:
                        map_count += 1

        train_count = 0
        with open(output_files[1], 'w', encoding='utf-8') as f:
            for line in mapper['train']:
                f.writelines(line + '\n')
            for b in base:
                f.writelines(b + '\n')

        with open(output_files[2], 'w', encoding='utf-8') as f:
            for line in mapper['val']:
                f.writelines(line + '\n')
            for b in base:
                f.writelines(b + '\n')

        with open(output_files[3], 'w', encoding='utf-8') as f:
            for line in mapper['test']:
                f.writelines(line + '\n')
            for b in base:
                f.writelines(b + '\n')

        if with_faq:
            faq = set()
            with open(grandfatherdir + '/data/memn2n/train/faq/facility.txt', encoding='utf-8') as cf:
                for line in cf:
                    a, b, c = line.strip('\n').split('\t')
                    candidates.add(b)
                    line = '\t'.join([a, b, 'placeholder'])
                    faq.add(line)
                for i in range(1, len(output_files)):
                    with open(output_files[i], 'a', encoding='utf-8') as f:
                        for line in faq:
                            f.writelines('\n')
                            f.writelines(line + '\n')

            with open(grandfatherdir + '/data/memn2n/train/faq/discount.txt', encoding='utf-8') as cf:
                for line in cf:
                    a, b, c = line.strip('\n').split('\t')
                    candidates.add(b)
                    line = '\t'.join([a, b, 'placeholder'])
                    faq.add(line)
                for i in range(1, len(output_files)):
                    with open(output_files[i], 'a', encoding='utf-8') as f:
                        for line in faq:
                            f.writelines('\n')
                            f.writelines(line + '\n')

        # candidate
        if with_base:
            with open(grandfatherdir + '/data/memn2n/train/base/candidates.txt', encoding='utf-8') as cf:
                for line in cf:
                    line = line.strip('\n')
                    candidates.add(line)
        candidates = list(candidates)
        candidates.sort()
        len_origin = len(candidates)
        if len_origin < m_config.CANDIDATE_POOL:
            for i in range(m_config.CANDIDATE_POOL - len_origin):
                candidates.append('reserved_' + str(i + len_origin))
        with open(output_files[0], 'w', encoding='utf-8') as f:
            for line in candidates:
                f.writelines(line + '\n')

        with open(grandfatherdir + '/data/memn2n/train/tree/api.txt', 'w', encoding='utf-8') as af:
            for line in api_set:
                af.writelines(line + '\n')

        print('writing', len(train_set), len(
            val_set), len(test_set), len_origin, len(candidates), 'base_count:', len(base))

        if not with_gbdt:
            return
        # gbdt
        with open(output_files[4], 'w', encoding='utf-8') as f:
            for line in train_gbdt:
                f.writelines(line + '\n')
            # hello
            with open(grandfatherdir + '/data/memn2n/dialog_simulator/greet.txt',
                      'r', encoding='utf-8') as hl:
                for line in hl:
                    line = "plugin:api_call_greet" + '#' + line.strip('\n')
                    f.writelines(line + '\n')
            # qa
            # with open(grandfatherdir + '/data/memn2n/dialog_simulator/qa.txt',
            #           'r', encoding='utf-8') as hl:
            #     for line in hl:
            #         line = "plugin:api_call_qa" + '#' + line.strip('\n')
            #         f.writelines(line + '\n')
            # chat
            if with_base:
                with open(grandfatherdir + '/data/memn2n/train/gbdt/chat.txt',
                          'r', encoding='utf-8') as hl:
                    for line in hl:
                        line = line.strip('\n')
                        cls, sentence = line.split('#')
                        f.writelines('plugin:api_call_base#' + sentence + '\n')


if __name__ == "__main__":
    graph_dir = os.path.join(grandfatherdir, "model/graph/belief_graph.pkl")
    config = dict()
    config['belief_graph'] = graph_dir
    config['solr.facet'] = 'off'
    config['shuffle'] = False
    # memory_dir = os.path.join(grandfatherdir, "model/memn2n/ckpt")
    log_dir = os.path.join(grandfatherdir, "log/test2.log")
    config['template'] = 'register_template.txt'
    tree_simulator = TreeSimilator(config)

    output_files = ['../../data/memn2n/train/tree/origin/candidates.txt',
                    '../../data/memn2n/train/tree/origin/train.txt',
                    '../../data/memn2n/train/tree/origin/val.txt',
                    '../../data/memn2n/train/tree/origin/test.txt',
                    '../../data/memn2n/train/gbdt/train.txt']

    tree_simulator.gen_sessions(output_files)
