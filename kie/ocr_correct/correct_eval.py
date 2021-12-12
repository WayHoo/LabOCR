# coding=utf-8
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

from kie.ocr_correct.correct import single_correct
from kie.ocr_correct.bktree import BKTree

with open("./doc/dict/correct_eval.txt", "r", encoding="utf-8") as f:
    EVAL_ITEMS = {}
    lines = f.readlines()
    for line in lines:
        vals = line.strip().split("\t")
        EVAL_ITEMS[vals[0]] = vals[1]


def test_bk_tree():
    tree = BKTree()
    with open("./doc/dict/medical_lab_items.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            word = line.strip()
            tree.add(word)
    for item, label in EVAL_ITEMS.items():
        res = tree.find(item, 5)
        print(item, label, res)


def test_fuzzy_correct():
    right_cnt = 0
    for item, label in EVAL_ITEMS.items():
        res, _, _ = single_correct(item)
        if res == label:
            right_cnt += 1
            print(item, label, res)
    total = len(EVAL_ITEMS)
    print("right_cnt=%d, total=%d, acc=%.2f" % (right_cnt, total, right_cnt*100/total))


if __name__ == "__main__":
    test_fuzzy_correct()
