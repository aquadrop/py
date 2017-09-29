'''
Created on september 28, 2017

a rule-based business dialog simulator

    user:query                bot:current             bot:accumulate                              bot:response               
ask[category]         |   slot:[category]   |    slot:[category]                       |    api_call request property1
provide[property1]    |   slot:[property1]  |    slot:[category,property1]             |    api_call request property2
provide[property2]    |   slot:[property2]  |    slot:[category,property1,property2]   |    api_call request property3
                                                    .
                                                    .
                                                    .
'''
import os
import sys

grandfatherdir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

from collections import OrderedDict
import numpy as np
from simulator_utils import Tv, Phone, Ac, name_map, template


class BusinessSimulator(object):
    def __init__(self, obj):
        self.category = name_map[obj.name]
        self.obj = obj
        self.necessary_property = obj.get_property()['necessary'].keys()
        self.other_property = obj.get_property()['other'].keys()

    def gen_normal_dialog(self):

        def one_turn(proper=None):
            if proper:  # not turn 1
                chosen_necessary_property = [proper]
                necessary.remove(proper)
                chosen_other_property = list()
            else:
                chosen_necessary_property = list()
                picked_necessary_num = np.random.randint(0, 2)
                for _ in range(picked_necessary_num + 1):
                    proper = np.random.choice(necessary)
                    chosen_necessary_property.append(proper)
                    necessary.remove(proper)

                chosen_other_property = list()
                picked_other_num = np.random.randint(1, 3)
                for _ in range(picked_other_num + 1):
                    proper = np.random.choice(other)
                    chosen_other_property.append(proper)
                    other.remove(proper)

            properties = chosen_necessary_property + chosen_other_property
            queries.append(properties)
            current_slots.append(properties)
            before_property = accumulate_slots[-1] if len(
                accumulate_slots) else accumulate_slots
            accumulate_slots.append(before_property + properties)

            request_property = np.random.choice(
                necessary) if len(necessary) else 'END'
            responses.append(request_property)

            return request_property

        necessary = list(self.necessary_property)
        other = list(self.other_property)
        queries = list()
        current_slots = list()
        accumulate_slots = list()
        responses = list()

        proper = None
        while len(necessary):
            proper = one_turn(proper)

        # print('queries:', queries)
        # print('current_slots', current_slots)
        # print('accumulate_slots:', accumulate_slots)
        # print('responses:', responses)

        dialogs = list()
        for _ in range(5000):
            dialog = self.fill(queries, current_slots,
                               accumulate_slots, responses)
            dialogs.append(dialog)
        # print(dialog)
        self.write_dialog(dialogs)

    def fill(self, queries, current_slots, accumulate_slots, responses):
        def get_property_value(proper, dic):
            value_necess = np.random.choice(
                dic['necessary'].get(proper, ['ERROR']))
            value_other = np.random.choice(dic['other'].get(proper, ['ERROR']))

            return value_necess if value_necess != 'ERROR' else value_other

        property_map = self.obj.get_property()
        property_value = {}
        for p in accumulate_slots[-1]:
            property_value[p] = get_property_value(p, property_map)
        # print(property_value)

        # responses
        responses_filled = ['api_call_request ' + res for res in responses]
        # queries
        queries_filled = list()
        for i, query in enumerate(queries):
            query_filled = ''.join(
                [property_value.get(p, 'ERROR') for p in query])
            if i == 0:
                query_filled = template.replace('[p]', query_filled)
                query_filled = query_filled.replace('[c]', self.category)
            # else:
            #     query_filled += 'provide'
            queries_filled.append(query_filled)
        # current
        current_slots_filled = list()
        for current_slot in current_slots:
            current_slot_filled = 'slot:' + ','.join(
                property_value.get(p, 'ERROR') for p in current_slot)
            current_slots_filled.append(current_slot_filled)
        # accumulate
        accumulate_slots_filled = list()
        for accumulate_slot in accumulate_slots:
            accumulate_slot_filled = 'slot:' + ','.join(
                property_value.get(p, 'ERROR') for p in accumulate_slot)
            accumulate_slots_filled.append(accumulate_slot_filled)

        return (queries_filled, current_slots_filled, accumulate_slots_filled, responses_filled)

    def write_dialog(self, dialogs):
        path = os.path.join(
            grandfatherdir, 'data/memn2n/normal_business_dialog.txt')
        dia = dialogs[0]
        rows = len(dia[0])
        cols = len(dia)

        with open(path, 'a', encoding='utf-8') as f:
            for dialog in dialogs:
                for i in range(rows):
                    for j in range(cols):
                        f.write(dialog[j][i] + '#')
                    f.write('\n')
                f.write('\n')


def main():
    tv = Tv()
    ac = Ac()
    phone = Phone()
    ll = [tv, ac, phone]
    for l in ll:
        print(l.name)
        bs = BusinessSimulator(l)
        bs.gen_normal_dialog()


if __name__ == '__main__':
    main()
