from typing import List, Tuple, Union
from sortedcontainers import SortedList

class TrieNode:
    def __init__(self):
        self.children: List[Union[TrieNode, None]] = [None] * 256

    def add_element(self, element: int):
        if self.children[element] is not None:
            return
        self.children[element] = TrieNode()

    def get_next(self, element: int) -> Union["TrieNode", None]:
        return self.children[element]


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def add_str(self, s: List[int]):
        cur = self.root
        for i in s:
            cur.add_element(i)
            cur = cur.get_next(i)

    def is_present(self, s: List[int]) -> bool:
        cur = self.root
        for i in s:
            cur = cur.get_next(i)
            if cur is None:
                return False
        return True

    def add_list(self, tokens: List[str]) -> None:
        for token in tokens:
            self.add_str(list(token.encode("utf-8")))


def encode(s: str) -> bytes:
    """
    Encodes a string into bytes(utf-8 encoding)
    """
    return s.encode("utf-8")


def decode(encoded_string: bytes) -> str:
    """
    Decodes utf-8 encoded string into bytes(utf-8 encoding)
    """
    return encoded_string.decode("utf-8")


def get_encoded_byte_tuple(s: str) -> Tuple[int, ...]:
    """
    Returns a list of bytes encoded strings.
    """
    return tuple(encode(s))


class FastMaxPairSorted:
    def __init__(self):
        # We store (count, key) to sort by count
        self.sorted_list = SortedList()
        self.pair_to_count = {}

    def update(self, key, count_change):
        old_count = self.pair_to_count.get(key, 0)
        new_count = old_count + count_change

        if old_count != 0:
            self.sorted_list.remove((old_count, key))

        if new_count != 0:
            self.sorted_list.add((new_count, key))
            self.pair_to_count[key] = new_count
        else:
            del self.pair_to_count[key]

    def get_max(self):
        if not self.sorted_list:
            return None
        # The last element has the highest count
        max_count, max_key = self.sorted_list[-1]
        return max_key
