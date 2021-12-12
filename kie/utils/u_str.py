# coding=utf-8

# 判断一个unicode是否是汉字
def is_chinese(uchar):
    if u'\u4e00' <= uchar <= u'\u9fa5':
        return True
    else:
        return False


# 判断一个unicode是否是数字
def is_number(uchar):
    if u'\u0030' <= uchar <= u'\u0039':
        return True
    else:
        return False


# 判断一个unicode是否是英文字母
def is_alphabet(uchar):
    if (u'\u0041' <= uchar <= u'\u005a') or (u'\u0061' <= uchar <= u'\u007a'):
        return True
    else:
        return False


# 判断是否是汉字、数字或英文字符
def is_valid(uchar):
    if is_chinese(uchar) or is_number(uchar) or is_alphabet(uchar):
        return True
    else:
        return False


# 计算字符串中汉字、数字、英文字符的数量
def calc_valid_char(string):
    cnt = 0
    for s in string:
        if is_valid(s):
            cnt += 1
    return cnt


def str_len(string):
    """
    计算字符串的长度，中文算一个字符，非中文算 0.5 个
    :param string: 字符串
    :return: 字符串长度，浮点数
    """
    ans = 0.0
    for char in string:
        if is_chinese(char):
            ans += 1.0
        else:
            ans += 0.5
    return ans


if __name__ == "__main__":
    str_list = ["项目（英文缩写）", "ID号:000169578", "9 天门冬氨酸氨基转移酶(AST)"]
    for u_str in str_list:
        print("origin_count=%d, valid_count=%d" % (len(u_str), calc_valid_char(u_str)))
    # 化验栏标题
    with open("./doc/dict/test_sheet_head_dict.txt", "r", encoding="utf-8") as f:
        tmp_words = f.read().splitlines()
        head_words = dict()
        standard_head = dict()
        val = 0
        for word in tmp_words:
            if word.startswith('#'):
                val = int(word[-1])
                standard_head[val] = word[1:-2]
                continue
            head_words[word] = val
        print(head_words)
        print(standard_head)
