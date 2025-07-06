import regex as re
from typing import List, Dict, Tuple, Iterable, Union, Iterator

from cs336_basics.pretokenization_example import find_chunk_boundaries

MAX_NUM_CHUNKS = 100  # Spawn maximum 100 processes

"""
Implementation of BPE tokenizer.
"""

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


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

    def add_str(self, s: bytes):
        cur = self.root
        for i in list(s):
            cur.add_element(i)
            cur = cur.get_next(i)

    def is_present(self, s: bytes) -> bool:
        cur = self.root
        for i in list(s):
            cur = cur.get_next(i)
            if cur is None:
                return False
        return True

    def add_list(self, tokens: List[str]) -> None:
        for token in tokens:
            self.add_str(token.encode("utf-8"))


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


def get_encoded_byte_list(s: str) -> Tuple[bytes, ...]:
    """
    Returns a list of bytes encoded strings.
    """
    encoded_string = encode(s)
    return tuple(bytes([x]) for x in encoded_string)


class BytePairEncodingTokenizer:
    def __init__(self, special_tokens: List[str] = None):
        self.special_tokens = special_tokens or ["<|endoftext|>"]
        self._vocab: List[bytes] = [bytes([i]) for i in range(256)]
        self.vocab = {}
        self.rev_vocab = {}
        self.max_vocab_size = None
        self.pre_tokens_dict: Dict[Tuple[bytes, ...], int] = (
            {}
        )  # will be incrementally updated
        self.byte_pair_index = {}

        self.merges: List[Tuple[bytes, bytes]] = (
            []
        )  # List of merges that have taken place
        self.merges_dict: Dict[Tuple[bytes, bytes], int] = {}
        self.trie = Trie()
        self.trie.add_list(self.special_tokens)

    def encode(self, s: str) -> List[int]:
        return list(self.encode_iterable(s))

    def decode(self, s: List[int]) -> str:
        pass

    def from_files(self, vocab_filepath, merge_filepath, special_tokens=None):
        pass

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buffer = ""
        tokens = []
        cur_node = None
        for char in iterable:
            if tokens:
                yield tokens.pop()
            if buffer in self.special_tokens:
                # Special token is found. return that
                buffer = ""
                cur_node = None
                yield self.vocab[buffer]
            if buffer and cur_node is None:
                tokens = self.tokenize(buffer)
                tokens.reverse()
                yield tokens.pop()
            if cur_node is None:
                cur_node = self.trie.root
            cur_node = cur_node.get_next()
            buffer += char

    def _pre_tokenize(self, filename: str):
        """
        Given a file containing text corpus perform pre-tokenization on the corpus :)
        Uses multiprocessing for speedups!.
        """
        chunks = self.get_chunks(filename)
        import multiprocessing

        with multiprocessing.Pool() as pool:
            results = pool.map(self._pre_tokenize_chunk, chunks)
            pool.close()
            for result in results:
                for token, count in result.items():
                    self.pre_tokens_dict[token] = (
                        self.pre_tokens_dict.get(token, 0) + count
                    )

    def _merge(self):
        max_pair = max(self.byte_pair_index, key=self.byte_pair_index.get)
        self.merges.append(max_pair)
        self._update_byte_pair_index(max_pair)
        self._vocab.append(max_pair[0] + max_pair[1])

    def train(
        self, filename: str, max_vocab_size: int, special_tokens: List[str] = None
    ):
        # initialize
        # TODO: Remove default <|endoftext|> special token
        self.special_tokens = special_tokens or ["<|endoftext|>"]
        self._vocab: List[bytes] = [bytes([i]) for i in range(256)] + special_tokens
        self.max_vocab_size = max_vocab_size
        num_iterations = self.max_vocab_size - (len(self.special_tokens) + 256)
        self.pre_tokens_dict: Dict[Tuple[bytes, ...], int] = (
            {}
        )  # will be incrementally updated
        self.byte_pair_index = {}

        self.merges: List[Tuple[bytes, bytes]] = (
            []
        )  # List of merges that have taken place
        # Pre tokenize
        self._pre_tokenize(filename)
        # Generate the frequency index
        self.generate_byte_pair_index()
        # train
        for i in range(num_iterations):
            previous_vocab_len = len(self._vocab)
            self._merge()
            if previous_vocab_len == len(self._vocab):
                break
            if (i + 1) % 100 == 0:
                print("Iteration {}/{} completed".format(i + 1, num_iterations))
        for idx, val in enumerate(self._vocab):
            self.vocab[idx] = val
            self.rev_vocab[val] = idx
        for idx, pair in enumerate(self.merges):
            self.merges_dict[pair] = idx

    def generate_byte_pair_index(self) -> None:
        for word_split in self.pre_tokens_dict:
            for tok1, tok2 in zip(word_split[:-1], word_split[1:]):
                self.byte_pair_index[(tok1, tok2)] = (
                    self.byte_pair_index.get((tok1, tok2), 0) + 1
                )

    def _update_byte_pair_index(self, max_pair: Tuple[bytes, bytes]) -> None:
        # Update the pair frequency dictionary

        previous_tokens = list(self.pre_tokens_dict.keys())

        for word_split in previous_tokens:
            new_word_split = []
            # TODO: Test if adding condition of existence of max_pair[0] in word_split improves performance
            if max_pair[0] not in word_split:
                continue
            idx, updated = 0, False
            word_split_len = len(word_split)
            while idx < word_split_len - 1:
                if (
                    word_split[idx] == max_pair[0]
                    and word_split[idx + 1] == max_pair[1]
                ):
                    new_word_split.append((word_split[idx] + word_split[idx + 1]))
                    updated = True
                    idx += 2
                    continue
                new_word_split.append(word_split[idx])
                idx += 1

            if not updated:
                continue
            # Update pre tokens dict
            val = self.pre_tokens_dict.pop(word_split)
            self.pre_tokens_dict[tuple(new_word_split)] = val
            # Update the byte pair frequency index
            for tok1, tok2 in zip(word_split[:-1], word_split[1:]):
                self.byte_pair_index[(tok1, tok2)] -= val
            for tok1, tok2 in zip(new_word_split[:-1], new_word_split[1:]):
                self.byte_pair_index[(tok1, tok2)] = (
                    self.byte_pair_index.get((tok1, tok2), 0) + val
                )

    def tokenize(self, s: str) -> List[int]:
        """
        Do not use this method. This is supposed to be used only when you are sure that `s` does not contain any special tokens
        """

        def get_pairs(bytes_list: List[bytes]) -> List[Tuple[bytes, bytes]]:
            return list(set(zip(bytes_list[:-1], bytes_list[1:])))

        tokenization = []
        token_iter = re.finditer(s, PAT)
        for token in token_iter:
            byte_str = token.group().encode("uft-8")
            tokens = [bytes([i]) for i in byte_str]
            while len(tokens) >= 2:
                pairs = get_pairs(tokens)
                min_pair = min(
                    pairs, key=lambda x: self.merges_dict.get(x, float("inf"))
                )
                if min_pair not in self.merges:
                    break
                new_tokens = []
                idx = 0
                while idx < len(tokens) - 1:
                    val1, val2 = tokens[idx], tokens[idx + 1]
                    if val1 == min_pair[0] and val2 == min_pair[1]:
                        new_tokens.append(val1 + val2)
                        idx += 2
                    else:
                        new_tokens.append(val1)
                        idx += 1
                tokens = new_tokens
            tokenization += [self.vocab[idx] for idx in tokens]
        return tokenization

    def _pre_tokenize_chunk(self, chunk):
        return self.pre_tokenize_chunk(chunk, self.special_tokens)

    @staticmethod
    def get_chunks(filename):
        chunks = []
        with open(filename, "rb") as f:
            chunk_boundaries = find_chunk_boundaries(
                f, MAX_NUM_CHUNKS, "<|endoftext|>".encode("utf-8")
            )
            for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode(
                    "utf-8", errors="ignore"
                )  # I can run pre-tok here as well.
                chunks.append(chunk)
        return chunks

    @staticmethod
    def pre_tokenize_chunk(s: str, special_tokens) -> Dict[Tuple[bytes, ...], int]:
        split_str = "|".join(map(re.escape, special_tokens))
        splits = re.split(split_str, s)
        tokenization_corpus = {}
        for split in splits:
            token_iter = re.finditer(PAT, split)
            for token in token_iter:
                encode_list = get_encoded_byte_list(token.group())
                tokenization_corpus[encode_list] = (
                    tokenization_corpus.get(encode_list, 0) + 1
                )
        return tokenization_corpus

    @staticmethod
    def get_bpe_tokenizer(
        vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens
    ) -> "BytePairEncodingTokenizer":
        bpe_tokenizer = BytePairEncodingTokenizer(special_tokens=special_tokens or [])
        bpe_tokenizer.vocab = vocab
        for idx, val in vocab.items():
            bpe_tokenizer.rev_vocab[val] = idx
        for idx, merge in enumerate(merges):
            bpe_tokenizer.rev_vocab[merge[0] + merge[1]] = len(vocab)
            bpe_tokenizer.vocab[len(vocab)] = merge[0] + merge[1]
            bpe_tokenizer.merges_dict[merge] = idx
        return bpe_tokenizer
