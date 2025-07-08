from collections import defaultdict
from typing import Dict, Set

from cs336_basics.byte_pair_encoding_tokenizer import BytePairEncodingTokenizer
from cs336_basics.utils import *

"""
Implementation of BPE tokenizer.
"""


class BytePairEncodingTokenizerInvertedIndex(BytePairEncodingTokenizer):
    def __init__(self, special_tokens: List[str] = None):
        super().__init__(special_tokens)
        self.token_inverted_index: Dict[int, Set[int]] = defaultdict(set)
        self.pre_tokens_inverted_index: List[Tuple[int, ...]] = []
        self.byte_pair_index = FastMaxPairSorted(self._vocab)

    def _merge(self):
        max_pair = self.byte_pair_index.get_max()
        if not max_pair:
            return
        self.merges.append(max_pair)
        self.merges_dict[max_pair] = len(self._vocab)
        self._vocab.append(self._vocab[max_pair[0]] + self._vocab[max_pair[1]])
        self._update_byte_pair_index(max_pair)

    def _generate_byte_pair_index(self) -> None:
        self.pre_tokens_inverted_index = list(self.pre_tokens_dict.keys())
        for word_split, val in self.pre_tokens_dict.items():
            for tok1, tok2 in zip(word_split[:-1], word_split[1:]):
                self.byte_pair_index.update((tok1, tok2), val)
        for idx, word_split in enumerate(self.pre_tokens_inverted_index):
            for tok in word_split:
                self.token_inverted_index[tok].add(idx)

    def _update_byte_pair_index(self, max_pair: Tuple[int, int]) -> None:
        pre_tokens_index_list = list(
            self.token_inverted_index[max_pair[0]].union(
                self.token_inverted_index[max_pair[1]]
            )
        )
        pre_tokens = [
            self.pre_tokens_inverted_index[idx] for idx in pre_tokens_index_list
        ]
        vocab_len = len(self._vocab)
        for pre_token_index, word_split in zip(pre_tokens_index_list, pre_tokens):
            if max_pair[0] not in word_split:
                continue
            idx = 0
            word_split_len, new_word_split_len = len(word_split), 0
            new_word_split = [0] * word_split_len
            while idx < word_split_len - 1:
                if (
                    word_split[idx] == max_pair[0]
                    and word_split[idx + 1] == max_pair[1]
                ):
                    new_word_split[new_word_split_len] = vocab_len - 1
                    new_word_split_len += 1
                    idx += 2
                    continue
                new_word_split[new_word_split_len] = word_split[idx]
                new_word_split_len += 1
                idx += 1
            if idx == word_split_len - 1:
                new_word_split[new_word_split_len] = word_split[idx]
                new_word_split_len += 1
            if new_word_split_len == word_split_len:
                continue
            # Update pre tokens dict
            val = self.pre_tokens_dict.pop(word_split)
            new_word_split = new_word_split[:new_word_split_len]
            self.pre_tokens_inverted_index[pre_token_index] = tuple(new_word_split)
            self.pre_tokens_dict[tuple(new_word_split)] = val
            # Update the byte pair frequency index
            for tok1, tok2 in zip(word_split[:-1], word_split[1:]):
                self.byte_pair_index.update((tok1, tok2), -val)
            for tok1, tok2 in zip(new_word_split[:-1], new_word_split[1:]):
                self.byte_pair_index.update((tok1, tok2), val)
            for tok in new_word_split:
                self.token_inverted_index[tok].add(pre_token_index)
