# coding=utf-8
"""
Reference: https://fuzzychinese.zenan-wang.com/
GitHub: https://github.com/znwang25/fuzzychinese
"""
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

import logging
import pandas as pd
from kie.ocr_correct.fuzzychinese import FuzzyChineseMatch, Stroke
from sklearn.feature_extraction.text import TfidfVectorizer
default_logger = logging.getLogger(__name__)


class FuzzyCorrector(object):
    _dir = os.path.dirname
    _default_dict_file_path = os.path.join(_dir(_dir(_dir(__file__))),
                                           "doc/dict/medical_lab_items.txt")

    def __init__(self, dict_file_path=None):
        if dict_file_path:
            self._dict_file_path = dict_file_path
        else:
            self._dict_file_path = self._default_dict_file_path
        self._read_dictionary()
        self.fcm = FuzzyChineseMatch(ngram_range=(3, 3), analyzer='stroke')
        self.fcm.fit(self._dictionary)

    def _read_dictionary(self):
        words = set()
        default_logger.debug('Reading test sheet dictionary ...')
        with open(self._dict_file_path, encoding="UTF-8") as f:
            for line in f:
                words.add(line.strip())
        self._dictionary = pd.Series(list(words))

    def get_top_candidates(self, words, n=1):
        if not isinstance(words, list):
            raise Exception('The param word must be type of list.')
        if not isinstance(n, int):
            raise Exception('The param n must be type of int.')
        if n <= 0:
            raise Exception('The param n must be positive integer.')
        top_n_similar = self.fcm.transform(words, n=n)
        top_n_score = self.fcm.get_similarity_score()
        ret = []
        for i in range(0, len(top_n_score)):
            zp = zip(top_n_similar[i], top_n_score[i])
            ret.append(list(zp))
        return ret


if __name__ == "__main__":
    fc = FuzzyCorrector()
    raw_words = ["申该细胞绝对值", "林巴细胞数", "载脂蛋白", "钾",
                 "口嗜碱性粒细胞", "白细胞", "中形粒细胞数"]
    candi = fc.get_top_candidates(raw_words, 1)
    print(candi)
    stroke = Stroke()
    print("载：", stroke.get_stroke("载"))
    print("脂：", stroke.get_stroke("脂"))
    print("蛋：", stroke.get_stroke("蛋"))
    print("白：", stroke.get_stroke("白"))
    # tfidf2 = TfidfVectorizer()
    # strokes = []
    # for word in raw_words:
    #     tmp = ""
    #     for ch in word:
    #         tmp += stroke.get_stroke(ch)
    #     strokes.append(tmp)
    # re = tfidf2.fit_transform(strokes)
    # print("笔划：", strokes)
    # print("关键词：", tfidf2.get_feature_names())
    # matrix = re.toarray()
    # print("词频矩阵：", matrix)
    # print(re)
