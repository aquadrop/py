import os
import sys
import numpy as np
from collections import OrderedDict

prefix = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

template_path = 'bookstore_qa_template.txt'
faq_floor_template = 'bookstore_faq_template.txt'
outpath = prefix + '/data/memn2n/train/map/map.txt'
REPLACE = '[]'
table = [
    {"title":["苹果专柜"], "brand": ["苹果"], "virtual_category": ["数码产品"], "category": ["手机"],
        "pc.series": ["iphone"], "location": ["一楼"], "store_id":"吴江新华书店","image_key":"吴江新华书店书城一楼"},
    {"title":["苹果专柜"],"brand": ["苹果"], "virtual_category": ["数码产品"], "category": ["平板"],
        "pc.series": ["ipad"], "location": ["一楼"], "store_id":"吴江新华书店","image_key":"吴江新华书店书城一楼"},
    {"title":["oppo专柜"],"brand": ["oppo", "欧珀"], "virtual_category": ["数码产品"],
        "category": ["手机"], "location": ["一楼"], "store_id":"吴江新华书店","image_key":"吴江新华书店书城一楼"},
    {"title":["vivo专柜"],"brand": ["vivo", "维沃"], "virtual_category": ["数码产品"],
        "category": ["手机"], "location": ["一楼"], "store_id":"吴江新华书店","image_key":"吴江新华书店书城一楼"},
    {"title":["华为专柜"],"brand": ["华为"], "virtual_category": ["数码产品"],
        "category": ["手机"], "location": ["一楼"],"image_key":"吴江新华书店书城一楼"},
    {"title":["小天才学习机"],"brand": ["小天才"], "virtual_category": ["数码产品"], "category": [
        "学习机"], "location": ["一楼"], "store_id":"吴江新华书店","image_key":"吴江新华书店书城一楼"},
    {"title":["精品图书展台"],"poi": ["精品图书展台", "图书展台", "精品展台"], "location": [
        "一楼中心区域"], "store_id":"吴江新华书店","image_key":"吴江新华书店书城一楼"},
    {"title":["吴江新华书店"],"poi": ["吴江新华书店"], "location": ["苏州市吴江区"],"store_id":"吴江新华书店"},
    {"title":["茶颜观色餐饮区"],"brand": ["茶颜观色"], "poi": ["餐饮区"],
        "location": ["二楼东侧"], "store_id":"吴江新华书店","image_key":"吴江新华书店书城二楼"},
    {"title":["咖啡馆"],"poi": ["咖啡"], "location": ["二楼东侧"], "store_id":"吴江新华书店","image_key":"吴江新华书店书城二楼"},
    {"title":["奶茶店"],"poi": ["奶茶"], "location": ["二楼东侧"], "store_id":"吴江新华书店","image_key":"吴江新华书店书城二楼"},
    {"title":["简餐"],"poi": ["简餐"], "location": ["二楼东侧"], "store_id":"吴江新华书店","image_key":"吴江新华书店书城二楼"},
    {"title":["小确幸绿植区"],"poi": ["小确幸绿植区"], "location": ["二楼西侧"], "store_id":"吴江新华书店","image_key":"吴江新华书店书城二楼"},
    {"title":["亨通市民书房"],"poi": ["亨通市民书房"], "location": ["二楼西侧"], "store_id":"吴江新华书店","image_key":"吴江新华书店书城二楼"},
    {"title":["文学类图书"],"category": ["图书"], "book.category": ["文学类"],
        "location": ["二楼"], "store_id":"吴江新华书店","image_key":"吴江新华书店书城二楼"},
    {"title":["杂志期刊"],"category": ["图书"], "book.category": ["杂志", "期刊", "杂志期刊"],
        "location": ["二楼"], "store_id":"吴江新华书店","image_key":"吴江新华书店书城二楼"},
    {"title":["社科类图书"],"category": ["图书"], "book.category": ["社会科学", "社科书", "科学书"],
        "location": ["二楼"], "store_id":"吴江新华书店","image_key":"吴江新华书店书城二楼"},
    {"title":["科技图书"],"category": ["图书"], "book.category": ["科技类", "科技书","科学类"],
        "location": ["二楼"], "store_id":"吴江新华书店","image_key":"吴江新华书店书城二楼"},
    {"title":["生活图书"],"category": ["图书"], "book.category": ["生活书"],
        "location": ["二楼"], "store_id":"吴江新华书店","image_key":"吴江新华书店书城二楼"},
    {"title":["少儿图书"],"category": ["图书"], "book.category": ["少儿图书"],
        "location": ["三楼"], "store_id":"吴江新华书店","image_key":"吴江新华书店书城三楼"},
    {"title":["小天才学习机"],"brand": ["诺亚舟"], "virtual_category": ["电教产品"], "category": [
        "电子词典"], "location": ["三楼"], "store_id":"吴江新华书店","image_key":"吴江新华书店书城三楼"},
    {"title":["步步高学习平板电脑"],"brand": ["步步高"], "virtual_category": ["电教产品"], "category": [
        "学习平板电脑"], "location": ["三楼"], "store_id":"吴江新华书店","image_key":"吴江新华书店书城三楼"},
    {"title":["读书郎点读机"],"brand": ["读书郎"], "virtual_category": ["电教产品"], "category": [
        "点读机"], "location": ["三楼"], "store_id":"吴江新华书店","image_key":"吴江新华书店书城三楼"},
    {"title":["创想学习桌"],"brand": ["创想"], "virtual_category": ["电教产品"], "category": [
        "学习桌"], "location": ["三楼"], "store_id":"吴江新华书店","image_key":"吴江新华书店书城三楼"},
    {"title":["文化用品"],"category": ["文具", "文化用品"], "location": ["四楼"], "store_id":"吴江新华书店","image_key":"吴江新华书店书城三楼"},
    {"title":["教辅类图书"],"category": ["图书"], "book.category": ["教辅", "小升初", "中考",
                                           "高考"], "location": ["四楼"], "store_id":"吴江新华书店","image_key":"吴江新华书店书城三楼"},
    {"title":["收银台"],"facility": ["收银台", "服务总台"], "location": [
        "一楼的电梯出来后右手边,在屏幕平面图的正上方"], "store_id":"吴江新华书店","image_key":"吴江新华书店书城三楼"},
    {"title":["卫生间"],"facility": ["卫生间", "厕所"], "location": [
        "在三楼的电梯出来后右手走到底,咨询工作人员"], "store_id":"吴江新华书店","image_key":"吴江新华书店书城三楼"},
    {"title":["电梯"],"facility": ["电梯"], "location": ["在屏幕平面图的右上方"], "store_id":"吴江新华书店","image_key":"吴江新华书店书城三楼"},
    {"title":["楼梯"],"facility": ["楼梯"], "location": ["在屏幕平面图的右上方"], "store_id":"吴江新华书店","image_key":"吴江新华书店书城三楼"}]


label_prefix = 'api_call_query_location'
faq_prefix = 'api_call_faq_info:'
entities = ["一楼","二楼","三楼","四楼"]
tel_info = "客服电话是多少？/客服电话？/客服电话多少啊？/给我报一下客服电话？/给我说一下客服电话？/客服电话是多少啊？/我想问一下客服电话？/我想问下你们客服的电话？/人工服务的电话/人工服务电话？/人工服务的电话多少啊？/我要你们人工服务的电话？/我想拨打你们的人工服务电话，是多少啊？/我想打人工服务，号码多少啊？/我要找客服，电话是多少啊？/我要打给客服，告诉我电话号码？/我需要人工服务，号码是多少？/问一下客服的电话？/问一下客服的号码？/问一下人工服务的号码？/人工服务的电话号码？/客服电话是什么/能告诉我客服电话是多少吗/能告诉我客服电话是什么吗/能跟我说下客服电话是多少嘛/能不能告诉我客服电话/能不能跟我说下客服电话/能跟我说下客服电话是什么吗/能跟我说下客服电话/告诉我客服电话/跟我说下客服电话/跟我讲一下客服电话/能跟我讲一下客服电话吗/能告诉我客服电话吗/能不能告诉我客服电话/能不能跟我讲一下客服电话吗/我要找客服电话/我想让你告诉我客服电话/我想知道客服电话/告诉我客服电话，可以吗/告诉我客服电话，行不行/跟我说一下客服电话行吗/告诉我客服电话行吗/麻烦你跟我说一下客服电话/麻烦你告诉我客服电话/我想让你跟我说一下客服电话/我想让你告诉我客服电话/你知道客服电话吗/跟我说一下客服电话，可行/你是否能告诉我客服电话/你是否知道客服电话/客服电话，谢谢/你能把客服电话告诉我吗/你知不知道客服电话，你能把你知道的客服电话告诉我吗/我想让你跟我说一下客服电话/我现在需要客服电话/我急需客服电话/你能马上告诉我客服电话吗/我在寻找客服电话，你知道吗/我不知道客服电话，你能告诉我吗/我有事要打客服电话，你知道号码吗/能告诉我客服的电话号码吗/客服的电话号码是多少呢/客服的电话号码/告诉我客服的电话号码/跟我说一下客服的电话号码，可以吗/客服的电话号码你知道是多少吗"
store_location_info = "你们店的地址是多少？/你们店在哪里？/你们的地址？/店址?/你们这个店在哪？/店在哪啊？/这个店在哪里啊？/你们的书店在哪里啊？/书店的地址是什么啊？/我想问下店址？/告诉我书店地址？/这个书店在哪里啊？/书店在哪？"

def gen_faq(template_path=faq_floor_template, outpath=outpath):
    with open(template_path, 'r') as f:
        tems = f.readlines()
    tems = [tem.strip() for tem in tems]
    lines = []
    for entity in entities:
        for template in tems:
            question = template.replace('[]', entity)
            cls = faq_prefix + '楼层'
            placeholder = 'placeholder'
            line = question + '\t' + cls + '\t' + placeholder
            lines.append(line)

    tel_info_lines = [question + '\t' + (faq_prefix + '客服电话') + '\t' + placeholder for question in tel_info.split('/')]
    lines.extend(tel_info_lines)

    store_location_lines = [question + '\t' + (label_prefix + '_poi:' + '吴江新华书店') + '\t' + placeholder for question in store_location_info.split('/')]
    lines.extend(store_location_lines)
    with open(outpath, 'a') as f:
        for line in lines:
            f.writelines(line + '\n\n')

def gen2(template_path=template_path, outpath=outpath, mode=True):
    with open(template_path, 'r') as f:
        tems = f.readlines()
    tems = [tem.strip() for tem in tems]
    res = set()
    keys_count = 0
    syn = _load_template()
    for line in table:
        keys = list(line.keys())
        for key in keys:
            if key in ['location', 'store_id', "title", "image_key"]:
                continue
            keys_count += 1
            values = line[key]
            values = values if isinstance(values, list) else [values]
            value = values[0]

            for t in tems:
                for v in values:
                    if v in syn:
                        for sv in syn[v]:
                            query = t.replace(REPLACE, sv)
                            if mode:
                                label = label_prefix + '_' + key + ':' + value
                            else:
                                label = label_prefix
                            res.add(query + '\t' + label + '\t' + 'placeholder')
                    else:
                        query = t.replace(REPLACE, v)
                        if mode:
                            label = label_prefix + '_' + key + ':' + value
                        else:
                            label = label_prefix
                        res.add(query + '\t' + label + '\t' + 'placeholder')

    res = list(res)
    print(len(res), keys_count)
    res.sort()
    with open(outpath, 'w') as f:
        for line in res:
            f.write(line + '\n\n')


def parse(lines):
    keys = lines[0].split(':')[1].split('|')
    lines = lines[1:]
    middle = OrderedDict()
    prefix = OrderedDict()
    postfix = OrderedDict()
    for line in lines:
        name, entities = line.split(':')
        entities = entities.strip().split('|')
        if name.startswith('prefix_'):
            prefix[name] = entities
        elif name.startswith('postfix_'):
            postfix[name] = entities
        else:
            middle[name] = entities

    return middle, prefix, postfix


def gen(outpath=outpath, path=template_path):
    with open(path, 'r') as f:
        lines = f.readlines()[1:]
    middle, prefix, postfix = parse(lines)

    res = list()
    for key in middle.keys():
        middles = middle[key]
        prefixs = prefix['prefix_' + key]
        postfixs = postfix['postfix_' + key]

        for m in middles:
            for pre in prefixs:
                query = pre + m
                label = 'api_call_qa_location_' + key + ':' + m
                line = query.strip() + '\t' + label.strip() + '\t' + 'placeholder'
                res.append(line)
            for post in postfixs:
                query = m + post
                label = 'api_call_qa_location_' + key + ':' + m
                line = query.strip() + '\t' + label.strip() + '\t' + 'placeholder'
                res.append(line)
    with open(outpath, 'w') as f:
        for line in res:
            f.write(line + '\n\n')


def main():
    gen2()
    gen_faq()

def _load_template():
    template_file = 'register_template.txt'
    thesaurus = dict()
    with open(template_file, 'r') as tf:
        for line in tf:
            line = line.strip('\n')
            parsers = line.split('|')
            if parsers[0] == 'thesaurus':
                thesaurus[parsers[1]] = parsers[2].split('/')[:] + [parsers[1]]
    return thesaurus


if __name__ == '__main__':
    main()
