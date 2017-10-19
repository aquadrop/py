import pycnnum

chs_arabic_map = {'零': 0, '一': 1, '二': 2, '三': 3, '四': 4,
                  '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
                  '十': 10, '百': 100, '千': 1000, '万': 10000,
                  '〇': 0, '壹': 1, '贰': 2, '叁': 3, '肆': 4,
                  '伍': 5, '陆': 6, '柒': 7, '捌': 8, '玖': 9,
                  '拾': 10, '佰': 100, '仟': 10000, '萬': 10000,
                  '亿': 100000000, '億': 100000000, '幺': 1,
                  '０': 0, '１': 1, '２': 2, '３': 3, '４': 4,
                  '５': 5, '６': 6, '７': 7, '８': 8, '９': 9, '两': 2}

digit_list = ['零', '一', '二', '三', '四',
              '五', '六', '七', '八', '九',
              '十', '百', '千', '万',
              '〇', '壹', '贰', '叁', '肆',
              '伍', '陆', '柒', '捌', '玖',
              '拾', '佰', '仟', '萬',
              '亿', '億', '幺', '两',
              '点']

skip_gram = ['三星', '一加', '三菱', '三门','万达','一楼','二楼','三楼','四楼','五楼','六楼']

convert_list = {'0':'零','1':'一','2':'二','3':'三','4':'四','5':'五','6':'六','7':'七','8':'八','9':'久'}

lead_digits = ['一', '二', '三', '四',
              '五', '六', '七', '八', '九',
              '壹', '贰', '叁', '肆',
              '伍', '陆', '柒', '捌', '玖',
             '两']


def new_cn2arab(query):

    if query.isdigit():
        return float(query)

    if len(query) == 0:
        return query

    result = []
    numstring = []
    for i in range(len(query)):
        char = query[i]
        if char not in digit_list:
            if len(numstring) > 0:
                numstring = ''.join([str(num) for num in numstring])
                result.append(pycnnum.cn2num(numstring))
                numstring = []
            result.append(char)
        else:
            if char == '点':
                try:
                    pre = query[i - 1]
                    post = query[i + 1]
                    if pre in digit_list and post in digit_list:
                        numstring.append(char)
                    else:
                        result.append(char)
                    continue
                except:
                    continue
            # if char in convert_list:
            #     char = convert_list[char]
            if i < len(query) - 1:
                test = char + query[i + 1]
                if test in skip_gram:
                    result.append(char)
                    continue
            numstring.append(char)

    if len(numstring) > 0:
        numstring = ''.join([str(num) for num in numstring])
        result.append(pycnnum.cn2num(numstring))
    result = [str(r) for r in result]
    return "".join(result)


def cn2arab(chinese_digits):
    if len(chinese_digits) == 0:
        return False, ''

    # chinese_digits = chinese_digits.decode("utf-8")

    prefix = []
    digit = []
    suffix = []
    pre_flag = False
    dig_flag = False
    for char in chinese_digits:
        if char not in digit_list and not pre_flag:
            prefix.append(char)
        elif char in digit_list and not dig_flag:
            digit.append(char)
            pre_flag = True
        else:
            dig_flag = True
            suffix.append(char)

    if len(digit) == 0:
        return False, ''.join(prefix)

    # print 'prefix', _uniout.unescape(str(prefix), 'utf-8')
    # print 'digit', _uniout.unescape(str(digit), 'utf-8')
    # print 'suffix', _uniout.unescape(str(suffix), 'utf-8')

    suffix = ''.join(suffix)
    if suffix:
        transferred, suffix = cn2arab(suffix)
    else:
        transferred = False
    return transferred or pre_flag, ''.join(prefix) + str(cn2arab_core(''.join(digit))) + suffix


def cn2arab_core(chinese_digits):

    if chinese_digits.isdigit():
        return int(chinese_digits)

    dig_mul = 1
    ## 100百万,取出100这个数字
    head_digits = []
    head = False
    for c in chinese_digits:
        if c.isdigit():
            head = True
            head_digits.append(c)
        else:
            break

    if len(head_digits) > 0:
        head_d = ''.join(head_digits)
        chinese_digits = chinese_digits.replace(head_d, '')
        dig_mul = float(head_d)

    if chinese_digits[0] not in lead_digits:
        chinese_digits = u'一' + chinese_digits
    result = 0
    tmp = 0
    hnd_mln = 0
    for count in range(len(chinese_digits)):
        curr_char = chinese_digits[count]
        curr_digit = chs_arabic_map.get(curr_char, None)
        # meet 「亿」 or 「億」
        if curr_digit == 10 ** 8:
            result = result + tmp
            result = result * curr_digit
            # get result before 「亿」 and store it into hnd_mln
            # reset `result`
            hnd_mln = hnd_mln * 10 ** 8 + result
            result = 0
            tmp = 0
        # meet 「万」 or 「萬」
        elif curr_digit == 10 ** 4:
            result = result + tmp
            result = result * curr_digit
            tmp = 0
        # meet 「十」, 「百」, 「千」 or their traditional version
        elif curr_digit >= 10:
            tmp = 1 if tmp == 0 else tmp
            result = result + curr_digit * tmp
            tmp = 0
        # meet single digit
        elif curr_digit is not None:
            tmp = tmp * 10 + curr_digit
        else:
            return float(result)
    result = result + tmp
    result = result + hnd_mln
    return float(result * dig_mul)

if __name__ == '__main__':
    s = ['十五哪点事','那点,42到50买一个三星手机两千一点五','3千','五十点二','三百','3百','两万','2万','2十万','100万','35','两千','1千零1百', '我要买一个两千到三千点二的手机']
    for ss in s:
        # print(ss, cn2arab(ss)[1])
        print(new_cn2arab(ss))
