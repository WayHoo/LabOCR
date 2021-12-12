# coding=utf-8
import json


with open("./doc/dict/test_sheet_head.json", "r", encoding="utf-8") as f:
    print("start load test_sheet_head.json ......")
    GL_HEAD_WORDS = {}
    GL_HEAD_ATTRS = {}
    head_words = json.load(f)
    for item in head_words:
        GL_HEAD_ATTRS[item["id"]] = (item["category"], 
                    item["standard_name"], item["sort_weight"])
        for word in item["words"]:
            GL_HEAD_WORDS[word] = item["id"]


with open("./doc/dict/test_sheet_key_words.json", "r", encoding="utf-8") as f:
    print("start load test_sheet_key_words.json ......")
    key_words = json.load(f)
    GL_KEY_WORDS = set(key_words)


def is_cfg_head_word(word):
    """
    检查给定词语是否在配置的表头关键词中
    :param word: 字符串
    :return: True or False
    """
    return word in GL_HEAD_WORDS


def get_category(word):
    """
    获取表头词语的纠错类别
    :param word: 字符串
    :return: 纠错类别，取值为 ["number", "item", "word", "sign", "unit", "range"]
    """
    if word in GL_HEAD_WORDS:
        id = GL_HEAD_WORDS[word]
        return GL_HEAD_ATTRS[id][0]
    return "other"


def get_std_name(word):
    """
    获取表头词语的标准名称(standard_name)
    :param word: 字符串
    :return: standard_name
    """
    if word in GL_HEAD_WORDS:
        id = GL_HEAD_WORDS[word]
        return GL_HEAD_ATTRS[id][1]
    return word


def get_sort_weight(word):
    """
    获取表头词语的排序权重
    :param word: 字符串
    :return: 整型的排序权重
    """
    if word in GL_HEAD_WORDS:
        id = GL_HEAD_WORDS[word]
        return GL_HEAD_ATTRS[id][2]
    return 0


def is_cfg_key_word(word):
    """
    判断给定词语是否在配置的关键词中
    :param word: 字符串
    :return: True or False
    """
    return word in GL_KEY_WORDS


def remove_dup(file="./doc/dict/medical_lab_items.txt"):
    with open(file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        items = set()
        for line in lines:
            line = line.strip()
            if line not in items:
                items.add(line)
        items = list(items)
        items = sorted(items, key=lambda word : len(word))
        for item in items:
            print(item)

if __name__ == "__main__":
    print("is_cfg_head_word(\"项目全称\")：", is_cfg_head_word("项目全称"))
    print("is_cfg_head_word(\"FLAG\")：", is_cfg_head_word("FLAG"))
    print("get_sort_weight(\"FLAG\")：", get_sort_weight("FLAG"))
    print("get_category(\"参考范围\")：", get_category("参考范围"))
    print("get_std_name(\"FLAG\")：", get_std_name("FLAG"))
    print("get_std_name(\"参考值\")：", get_std_name("参考值"))
    print("is_cfg_key_word(\"送检时间\")：", is_cfg_key_word("送检时间"))
    # remove_dup()
    