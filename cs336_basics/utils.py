from typing import List, Tuple, Union
import heapq


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
    def __init__(self, vocab=None):
        self.heap = []
        self.pair_to_count = {}
        self.vocab = vocab

    def _sort_key(self, key, count):
        if self.vocab is None:
            return (count, key)
        return (count, (self.vocab[key[0]], self.vocab[key[1]]), key)

    def update(self, key, count_change):
        old_count = self.pair_to_count.get(key, 0)
        new_count = old_count + count_change

        if new_count != 0:
            self.pair_to_count[key] = new_count
            heapq.heappush(self.heap, _MaxPairEntry(self._sort_key(key, new_count)))
        elif key in self.pair_to_count:
            del self.pair_to_count[key]

    def get_max(self):
        while self.heap:
            entry = self.heap[0]
            key = entry.sort_key[-1]
            count = entry.sort_key[0]
            if self.pair_to_count.get(key) == count:
                return key
            heapq.heappop(self.heap)
        return None


class _MaxPairEntry:
    def __init__(self, sort_key):
        self.sort_key = sort_key

    def __lt__(self, other):
        return self.sort_key > other.sort_key
