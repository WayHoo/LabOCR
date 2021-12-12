import Levenshtein
from collections import deque
from operator import itemgetter


__all__ = ['BKTree']
_getitem0 = itemgetter(0)


class BKTree(object):
    def __init__(self, items=None):
        if items is None:
            items = []
        self.distance_func = Levenshtein.distance
        self.tree = None

        # Slight speed optimization -- avoid lookups inside the loop
        _add = self.add
        for item in items:
            _add(item)

    def add(self, item):
        node = self.tree
        if node is None:
            self.tree = (item, {})
            return

        # Slight speed optimization -- avoid lookups inside the loop
        _distance_func = self.distance_func

        while True:
            parent, children = node
            distance = _distance_func(item, parent)
            node = children.get(distance)
            if node is None:
                children[distance] = (item, {})
                break

    def find(self, item, n):
        if self.tree is None:
            return []

        candidates = deque([self.tree])
        found = []

        # Slight speed optimization -- avoid lookups inside the loop
        _candidates_popleft = candidates.popleft
        _candidates_extend = candidates.extend
        _found_append = found.append
        _distance_func = self.distance_func

        while candidates:
            candidate, children = _candidates_popleft()
            distance = _distance_func(candidate, item)
            if distance <= n:
                _found_append((distance, candidate))
            if children:
                lower = distance - n
                upper = distance + n
                _candidates_extend(c for d, c in children.items() if lower <= d <= upper)

        found.sort(key=_getitem0)
        return found

    def __iter__(self):
        """
        Return iterator over all items in this tree; items are yielded in
        arbitrary order.
        """
        if self.tree is None:
            return

        candidates = deque([self.tree])

        # Slight speed optimization -- avoid lookups inside the loop
        _candidates_popleft = candidates.popleft
        _candidates_extend = candidates.extend

        while candidates:
            candidate, children = _candidates_popleft()
            yield candidate
            _candidates_extend(children.values())

    def __repr__(self):
        """Return a string representation of this BK-tree with a little bit of info.

        >>> BKTree()
        <BKTree using distance with no top-level nodes>
        >>> BKTree(["钾离子", "白细胞数", "球蛋白", "红细胞压积"])
        <BKTree using distance with 3 top-level nodes>
        """
        return '<{} using {} with {} top-level nodes>'.format(
            self.__class__.__name__,
            self.distance_func.__name__,
            len(self.tree[1]) if self.tree is not None else 'no',
        )


if __name__ == '__main__':
    tree = BKTree()
    with open("./doc/dict/medical_lab_items.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            word = line.strip()
            tree.add(word)
    res = tree.find("RH(D)RH血型", 5)
    print(res)
