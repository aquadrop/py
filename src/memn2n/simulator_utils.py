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


class Phone(Base):
    def __init__(self):
        self.property_map = dict()
        self.necessary = dict()
        self.other = dict()
        self.name = 'phone'
        self.brand = ["华为", "oppo", "苹果", "vivo",
                      "金立", "三星", "荣耀", "魅族", "moto", "小米"]
        self.sys = ["android", "ios"]
        self.net = ["全网通", "移动4G", "联通4G", "电信4G", "双卡双4G", "双卡单4G"]
        self.feature = ["老人手机", "拍照神器", "女性手机", "儿童手机"]
        self.color = ["红", "黑", "白", "深空灰", "玫瑰金"]
        self.mem_size = ["16G", "32G", "64G", "128G", "256G"]
        self.price = ["#number#"]

        # index represent priority
        self.necessary_property = ['price', 'brand']

    def get_property(self):
        self.necessary['price'] = self.price
        self.necessary['brand'] = self.brand

        self.other['sys'] = self.sys
        self.other['net'] = self.net
        self.other['feature'] = self.feature
        self.other['color'] = self.color
        self.other['mem_size'] = self.mem_size

        self.property_map['necessary'] = self.necessary
        self.property_map['other'] = self.other

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
