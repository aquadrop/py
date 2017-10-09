import numpy as np
import os

buyQueryMapper = {
    'ask': {
        'user': {
            'brand': ['你们这里有<brand>吗？', '我要买<brand>。'],
            'category': ['<category>都有哪些？', '我要买<category>。'],
            'brand category': ['有<brand><category>吗？', '我要买<brand><category>。']
        },
        'bot': {
            # initiative to request information,brand,category,price et.
            'brand': 'api_call_slot:<brand> ',
            'category': 'api_call_slot:<category>',
            'brand category': 'api_call_slot:<brand>,<category>'
        }
    },
    'rhetorical_ask': {
        'user': {
            'brand': ['你们这都有什么牌子？', '品牌都有哪些？'],
            'category': ['你们这儿都卖什么种类？', '种类都有哪些？'],
            'price': ['都有哪些价格？', '都有什么价啊？']
        },
        'bot': {
            'brand': 'api_call_rhetorical:(brand)',
            'category': 'api_call_rhetorical:(category)',
            'price': 'api_call_rhetorical:(price)'
        }
    },
    'provide': {
        'user': {
            'brand': ['<brand>的。', '喜欢<brand>。'],
            'price': ['大概<price>的吧。', '<price>。'],
            'category': ['<category>类的。']
        },
        'bot': {
            'brand': 'api_call_slot:<brand>',
            'price': 'api_call_slot:<price>',
            'category': 'api_call_slot:<category>'
        }
    },
    'deny': {
        'user': {
            'brand': ['不喜欢这个牌子。', '不喜欢。'],
            'price': ['这个价太贵了。', '有点贵啊。'],
            'general': ['不喜欢。', '还有其它的吗？']
        },
        'bot': {
            'brand': 'api_call_deny:(brand)',
            'price': 'api_call_deny:(price)',
            'general': 'api_call_deny:(general) '
        }
    },
    'accept': {
        'user': {
            'brand': ['我喜欢这个牌子。', '不错的。'],
            'price': ['这个价钱可以接受。', '不算贵，可以的。'],
            'general': ['挺好的。', '不错，就这个了。']
        },
        'bot': {
            'brand': '很高兴您能买到合适的商品',
            'price': '很高兴您能买到合适的商品',
            'general': '很高兴您能买到合适的商品'
        }  # could be '很高兴您能买到合适的[category]'
    }
}


mapper = {
    'greet': '您好，请问有什么可以帮助您的？',
    'chat': 'api_call_embotibot',
    'qa': 'api_call_qa',
    'bye': '再见，谢谢光临！',
    'buy': buyQueryMapper
}

name_map = {'base': '基类', 'tv': '电视', 'ac': '空调', 'phone': '手机'}
template = '我要买[p]的[c]'


class Base(object):
    def __init__(self):
        self.name = 'base'
        self.discount = ["满1000减200", "满2000减300", "十一大促销"]
        self.property_map = dict()
        self.necessary = dict()
        self.other = dict()

    def get_property(self):
        return self.property_map


class Tv(Base):
    def __init__(self):
        self.property_map = dict()
        self.necessary = dict()
        self.other = dict()
        self.name = 'tv'
        self.size = ["#number#"]
        self.type = ["智能", "普通", "互联网"]
        self.price = ["#number#"]
        self.brand = ["索尼", "乐视", "三星", "海信", "创维", "TCL"]
        self.distance = ["#number#"]
        self.resolution = ["4K超高清", "全高清", "高清"]
        self.panel = ["LED", "OLEC", "LCD", "等离子"]
        self.power_level = ["#number#"]

        # index represent priority
        self.necessary_property = ['size', 'type', 'price']

    def get_property(self):
        self.necessary['size'] = self.size
        self.necessary['type'] = self.type
        self.necessary['price'] = self.price

        self.other['brand'] = self.brand
        self.other['distance'] = self.distance
        self.other['resolution'] = self.resolution
        self.other['panel'] = self.panel
        self.other['power_level'] = self.power_level

        self.property_map['necessary'] = self.necessary
        self.property_map['other'] = self.other
        # print(self.property_map)
        return self.property_map


class Ac(Base):
    def __init__(self):
        self.property_map = dict()
        self.necessary = dict()
        self.other = dict()
        self.name = 'ac'
        self.power = ["#number#"]
        self.area = ["#number#"]
        self.type = ["圆柱", "立式", "挂壁式", "立柜式", "中央空调"]
        self.brand = ["三菱", "松下", "科龙", "惠而浦", "大金", "目立", "海尔", "美的", "卡萨帝",
                      "奥克斯", "长虹", "格力", "莱克", "艾美特", "dyson", "智高", "爱仕达", "格兰仕"]
        self.price = ["#number#"]
        self.location = ["一楼", "二楼", "三楼", "地下一楼"]
        self.fr = ["变频", "定频"]
        self.cool_type = ["单冷", "冷暖"]

        # index represent priority
        self.necessary_property = ['type', 'power', 'price']

    def get_property(self):
        self.necessary['type'] = self.type
        self.necessary['power'] = self.power
        self.necessary['price'] = self.price

        self.other['area'] = self.area
        self.other['brand'] = self.brand
        self.other['location'] = self.location
        self.other['fr'] = self.fr
        self.other['cool_type'] = self.cool_type

        self.property_map['necessary'] = self.necessary
        self.property_map['other'] = self.other

        return self.property_map


class Dialogs:
    def __init__(self, userIntentFiles, businessFiles, candidatesFile, outputFiles):
        self.data = self.loadData(userIntentFiles)
        self.businessDialogs = self.loadBusiness(businessFiles)
        self.candidatesSet = self.loadCandidates(candidatesFile)
        self.outputFiles = outputFiles
        self.candidatesFile = candidatesFile

    def loadData(self, userIntentFiles):
        greetFile = userIntentFiles['greet']
        chatFile = userIntentFiles['chat']
        qaFile = userIntentFiles['qa']
        byeFile = userIntentFiles['bye']

        files = [greetFile, chatFile, qaFile, byeFile]
        data = dict()
        for file in files:
            name, _ = os.path.splitext(os.path.basename(file))
            with open(file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            data[name] = [l.strip() for l in lines]

        return data

    def loadBusiness(self, businessFiles):
        businessDialogs = dict()
        for file in businessFiles:
            name, _ = os.path.splitext(os.path.basename(file))
            dialogs = list()
            with open(file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            dialog = list()
            for line in lines:
                if line == '\n':
                    dialogs.append(dialog)
                    dialog = list()
                else:
                    dialog.append(line.strip())

            businessDialogs[name] = dialogs

        return businessDialogs

    def loadCandidates(self, candidatesFile):
        candidatesSet = set()
        with open(candidatesFile, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    candidatesSet.add(line)

        return candidatesSet

    def genDialog(self, new_candidates_file):
        keys = self.businessDialogs.keys()
        frontKeys = ['greet', 'chat', 'qa']
        backKeys = ['bye']
        for key in keys:
            print(key)
            businessDialogs = self.businessDialogs[key]
            with open(self.outputFiles[key], 'w', encoding='utf-8') as f:
                for dialog in businessDialogs:
                    frontDialog = list()
                    for k in frontKeys:
                        if np.random.uniform() < 0.25:
                            query = np.random.choice(self.data[k])
                            response = mapper[k]
                            self.candidatesSet.add(response)
                            frontDialog.append(
                                query + '\t' + response + '\t' + 'placeholder' + '\n')
                    frontDialog.extend(dialog)
                    dialog = frontDialog
                    backDialog = list()
                    for k in backKeys:
                        if np.random.uniform() < 0.25:
                            query = np.random.choice(self.data[k])
                            response = mapper[k]
                            self.candidatesSet.add(response)
                            backDialog.append(
                                query + '\t' + response + '\t' + 'placeholder' + '\n')
                    dialog.extend(backDialog)
                    for l in dialog:
                        f.write(l + '\n')
                    f.write('\n')
        with open(new_candidates_file, 'w', encoding='utf-8') as f:
            for line in self.candidatesSet:
                if line.strip():
                    f.write(line + '\n')


class Entity:
    def __init__(self, data_file):
        # index represent priority
        self.necessary_property = ['category', 'price', 'brand']
        self.required_fields = []
        self.profile = dict()
        self.field_type = dict()
        self.field_trans = dict()
        self.accumulate_slot_values = dict()
        with open(data_file, 'r', encoding='utf-8') as infile:
            line = infile.readline()
            title = line.strip('\n').split("|")[3].split(',')
            for line in infile:
                line = line.replace(' ', '').strip('\n')
                _, a, b, ft, c = line.split("|")
                self.profile[b] = c.split(",")
                self.field_type[b] = ft
                self.field_trans[b] = a

    def init_required_fields(self):
        self.required_fields = []
        self.accumulate_slot_values = dict()
        num = np.random.randint(0, 3)
        options = []
        for key, value in self.profile.items():
            if key not in self.necessary_property:
                options.append(key)
        self.required_fields.extend(self.necessary_property)
        options = np.random.choice(options, num).tolist()
        self.required_fields.extend(options)
        remove_list = ['discount', 'tmarket', 'nation', 'location']
        for rl in remove_list:
            if rl in self.required_fields:
                self.required_fields.remove(rl)

    def random_property(self, required_field):
        current_slot_values = {}
        num_avail = len(self.required_fields) - 1
        user_reply = [self.random_property_value(required_field)]
        self.accumulate_slot_values[required_field] = user_reply[0]
        current_slot_values[required_field] = user_reply[0]
        self.required_fields.remove(required_field)
        num_avail = min([num_avail, 2])
        if num_avail == 0:
            return self.accumulate_slot_values, current_slot_values
        random_num = np.random.randint(0, num_avail)

        for i in range(random_num):
            field = np.random.choice(self.required_fields)
            rnd = self.random_property_value(field)
            user_reply.append(rnd)
            self.accumulate_slot_values[field] = rnd
            current_slot_values[field] = rnd
            self.required_fields.remove(field)

        return self.accumulate_slot_values, current_slot_values

    def get_new_required_field(self):
        if len(self.required_fields) > 0:
            return np.random.choice(self.required_fields)
        else:
            return ''

    def gen_response(self, required_field):
        asv, csv = self.random_property(required_field)
        current_slots = ','.join(
            [key + ':' + value for key, value in csv.items()])
        user_replay = ','.join(csv.values())
        # for_tree_api = ','.join([key + ':' + value for key, value in asv.items()])
        # for_tree_api = 'api_call_slot:' + ','.join(csv.values())
        for_tree_api = 'api_call_slot:' + \
            ','.join([key + ':' + value for key, value in csv.items()])
        new_required = self.get_new_required_field()
        if new_required:
            tree_render_api = 'api_call_request_' + new_required
            # tree_render_api = "(" + new_required + ")"
        else:
            # tree_render_api = 'api_call_search_' + ','.join([key + ':' + value for key, value in asv.items()])
            tree_render_api = 'api_call_search_' + \
                ','.join([key + ':' + value for key, value in asv.items()])
            # tree_render_api = ''

        return user_replay, current_slots, for_tree_api, tree_render_api, new_required

    def random_property_value(self, field):
        if field == 'price':
            value = 'range'
            # if np.random.uniform() < 0.3:
            #     if np.random.uniform() < 0.5:
            #         value += '元'
            #     else:
            #         value += '块'

            return value

        if field == 'ac.power':
            value = 'range'

            # if np.random.uniform() < 0.5:
            #     value += 'P'
            # else:
            #     value += '匹'

            return value

        if self.field_type[field] == 'range':
            return 'range'
        else:
            return np.random.choice(self.profile[field])


def build_corpus(entity, candidate_file, train, val, test):
    entity.init_required_fields()
    required = 'category'
    candidate_set = set()
    train_set = []
    val_set = []
    test_set = []
    mapper = {'train': train_set, 'val': val_set, 'test': test_set}
    which = np.random.choice(['train', 'val', 'test'], p=[0.8, 0.1, 0.1])
    for i in range(2000):
        a, b, c, d, new_required = entity.gen_response(required)
        candidate = c
        # candidate = candidate.replace('api_call_request_', '').replace('api_call_tree_sn_', '')
        candidate_set.add(candidate)
        line = a + '\t' + candidate + '\t' + d
        mapper[which].append(line)

        if new_required:
            required = new_required
        else:
            required = 'category'
            entity.init_required_fields()
            mapper[which].append('')
            which = np.random.choice(
                ['train', 'val', 'test'], p=[0.8, 0.1, 0.1])

    with open(train, 'a', encoding='utf-8') as f:
        for line in mapper['train']:
            f.writelines(line + '\n')

    with open(val, 'a', encoding='utf-8') as f:
        for line in mapper['val']:
            f.writelines(line + '\n')

    with open(test, 'a', encoding='utf-8') as f:
        for line in mapper['test']:
            f.writelines(line + '\n')

    with open(candidate_file, 'a', encoding='utf-8') as f:
        for line in candidate_set:
            f.writelines(line + '\n')


def delete_file(file_path):
    try:
        os.remove(file_path)
    except:
        pass


if __name__ == '__main__':

    data_files = ['../../data/gen_product/shouji.txt',
                  '../../data/gen_product/kongtiao.txt',
                  '../../data/gen_product/bingxiang.txt',
                  '../../data/gen_product/dianshi.txt']

    delete_file('../../data/memn2n/train/candidates.txt')
    delete_file('../../data/memn2n/train/train.txt')
    delete_file('../../data/memn2n/train/val.txt')
    delete_file('../../data/memn2n/train/test.txt')


    for data_file in data_files:
        entity = Entity(data_file)
        build_corpus(entity,
                     '../../data/memn2n/train/candidates.txt',
                     '../../data/memn2n/train/train.txt',
                     '../../data/memn2n/train/val.txt',
                     '../../data/memn2n/train/test.txt')

    # uniq candidates
    candidates = set()
    with open('../../data/memn2n/train/candidates.txt', 'r', encoding='utf-8') as f:
        for line in f:
            candidates.add(line.strip('\n'))

    # candidates = set(candidates)
    # with open('../../data/memn2n/train/complex/candidates.txt', 'w') as f:
    #     for line in candidates:
    #         f.writelines(line + '\n')
    complex_candidate_file = '../../data/memn2n/train/complex/candidates.txt'
# --------------------------------------------------------------------
# generate complex dialogs
    new_candidates_file = '../../data/memn2n/train/complex/candidates.txt'
    outputFiles = {'train': '../../data/memn2n/train/complex/train.txt',
                   'val': '../../data/memn2n/train/complex/val.txt',
                   'test': '../../data/memn2n/train/complex/test.txt'
                   }

    userIntentFiles = {
        'greet': '../../data/memn2n/dialog_simulator/greet.txt',
        'chat': '../../data/memn2n/dialog_simulator/chat.txt',
        'qa': '../../data/memn2n/dialog_simulator/qa.txt',
        'bye': '../../data/memn2n/dialog_simulator/bye.txt'
    }

    businessFiles = ['../../data/memn2n/train/tree/train.txt',
                     '../../data/memn2n/train/tree/val.txt',
                     '../../data/memn2n/train/tree/test.txt']
    candidatesFile = '../../data/memn2n/train/tree/candidates.txt'
    dia = Dialogs(userIntentFiles, businessFiles, candidatesFile, outputFiles)

    dia.genDialog(new_candidates_file)
# --------------------------------------------------------------------

    #         print('------------------')
