import os
import numpy as np

import logging
logging.getLogger(__name__)


def edit_distance(str_a, str_b, name='Levenshtein'):
    """
    >>> edit_distance('abcde', 'avbcude')
    2
    >>> edit_distance(['至', '刂'], ['亻', '至', '刂'])
    1
    >>> edit_distance('fang', 'qwe')
    4
    >>> edit_distance('fang', 'hen')
    3
    """
    size_x = len(str_a) + 1
    size_y = len(str_b) + 1
    matrix = np.zeros((size_x, size_y), dtype=int)
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if str_a[x - 1] == str_b[y - 1]:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1],
                    matrix[x, y - 1] + 1
                )
            elif name == 'Levenshtein':
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1] + 1,
                    matrix[x, y - 1] + 1
                )
            else:  # Canonical
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1] + 2,
                    matrix[x, y - 1] + 1
                )

    return matrix[size_x - 1, size_y - 1]


def load_ids(file_name):
    _dir = os.path.dirname
    file_name = os.path.join(_dir(_dir(_dir(__file__))), file_name)
    data = {}
    with open(file_name, "r", encoding="utf-8") as f:
        for line in f:
            items = line.strip().split('\t')
            code_point = items[0]
            char = items[1]
            decomposition = items[2]
            assert char not in data
            data[char] = {"code_point": code_point, "decomposition": decomposition}
    return data


class CharSim(object):
    def __init__(self, ids_file_name):
        self.data = load_ids(ids_file_name)
        self.char_dict = dict([(c, 0) for c in self.data])

        # to eliminate the bug that, in Windows CMD, char ⿻ and ⿵ are encoded to be the same.
        self.safe = {'\u2ff0': 'A',  # ⿰
                     '\u2ff1': 'B',  # ⿱
                     '\u2ff2': 'C',  # ⿲
                     '\u2ff3': 'D',  # ⿳
                     '\u2ff4': 'E',  # ⿴
                     '\u2ff5': 'F',  # ⿵
                     '\u2ff6': 'G',  # ⿶
                     '\u2ff7': 'H',  # ⿷
                     '\u2ff8': 'I',  # ⿸
                     '\u2ff9': 'J',  # ⿹
                     '\u2ffa': 'L',  # ⿺
                     '\u2ffb': 'M'   # ⿻
                     }

    def _safe_encode_str(self, decomp):
        tree = ""
        for c in decomp:
            if c in self.safe:
                tree += self.safe[c]
            else:
                tree += c
        return tree

    def shape_similarity(self, char1, char2, safe=False):
        """
        >>> c = CharSim('doc/dict/ids.txt')
        >>> c.shape_similarity('田', '由')
        0.33333333333333337
        >>> c.shape_similarity('牛', '午')
        0.4285714285714286
        >>> c.shape_similarity('目', '日')
        0.7777777777777778
        """
        assert char1 in self.data
        assert char2 in self.data

        decomp1 = self.data[char1]["decomposition"]
        decomp2 = self.data[char2]["decomposition"]
        similarity = 0.0

        if not safe:
            ed = edit_distance(decomp1, decomp2)
        else:
            ed = edit_distance(self._safe_encode_str(decomp1), self._safe_encode_str(decomp2))
        norm_ed = ed / max(len(decomp1), len(decomp2))
        similarity = max(similarity, 1 - norm_ed)
        return similarity


if __name__ == '__main__':
    c = CharSim('./doc/dict/ids.txt')
    dis = c.shape_similarity('目', '日', safe=False)
    print(dis)
