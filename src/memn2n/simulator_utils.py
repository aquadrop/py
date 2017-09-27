buyQueryMapper = {
    'ask': {
        'user': {
            'brand': ['你们这里有<brand>吗？', '我要买<brand>。'],
            'category': ['<category>都有哪些？', '我要买<category>。'],
            'brand category': ['有<brand><category>吗？', '我要买<brand><category>。']
        },
        'bot': {
            # initiative to request information,brand,category,price et.
            'brand': 'api_call slot <brand> ',
            'category': 'api_call slot <category>',
            'brand category': 'api_call slot <brand> <category>'
        }
    },
    'rhetorical_ask': {
        'user': {
            'brand': ['你们这都有什么牌子？', '品牌都有哪些？'],
            'category': ['你们这儿都卖什么种类？', '种类都有哪些？'],
            'price': ['都有哪些价格？', '都有什么价啊？']
        },
        'bot': {
            'brand': 'api_call rhetorical (brand)',
            'category': 'api_call rhetorical (category)',
            'price': 'api_call rhetorical (price)'
        }
    },
    'provide': {
        'user': {
            'brand': ['<brand>的。', '喜欢<brand>。'],
            'price': ['大概<price>的吧。', '<price>。'],
            'category': ['<category>类的。']
        },
        'bot': {
            'brand': 'api_call slot <brand>',
            'price': 'api_call slot <price>',
            'category': 'api_call slot <category>'
        }
    },
    'deny': {
        'user': {
            'brand': ['不喜欢这个牌子。', '不喜欢。'],
            'price': ['这个价太贵了。', '有点贵啊。'],
            'general': ['不喜欢。', '还有其它的吗？']
        },
        'bot': {
            'brand': 'api_call deny (brand)',
            'price': 'api_call deny (price)',
            'general': 'api_call deny (general) '
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
    'chat': 'api_call embotibot',
    'qa': 'api_call qa',
    'bye': '再见，谢谢光临！',
    'buy': buyQueryMapper
}
