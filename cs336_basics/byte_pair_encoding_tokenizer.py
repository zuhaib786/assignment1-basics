import regex as re
from typing import List, Dict, Tuple, Iterable, Iterator


from cs336_basics.pretokenization_example import find_chunk_boundaries
from cs336_basics.utils import Trie, get_encoded_byte_tuple

MAX_NUM_CHUNKS = 20  # Spawn maximum 20 processes

"""
Implementation of BPE tokenizer.
"""

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class BytePairEncodingTokenizer:
    def __init__(self, special_tokens: List[str] = None):
        self.special_tokens = special_tokens or ["<|endoftext|>"]
        self._vocab: List[bytes] = [bytes([i]) for i in range(256)] + [
            special_tokens.encode("utf-8") for special_tokens in self.special_tokens
        ]
        self.rev_vocab = {}
        self.max_vocab_size = None
        self.pre_tokens_dict: Dict[Tuple[int, ...], int] = (
            {}
        )  # will be incrementally updated
        self.byte_pair_index = {}

        self.merges: List[Tuple[int, int]] = []  # List of merges that have taken place
        self.merges_dict: Dict[Tuple[int, int], int] = {}
        self.trie = Trie()
        self.trie.add_list(self.special_tokens)
        self.split_special_tokens_pattern = (
            "(" + "|".join(map(re.escape, self.special_tokens)) + ")"
        )

    def encode(self, s: str) -> List[int]:
        return list(self.encode_iterable(s))

    def decode(self, s: List[int]) -> str:
        decoded_bytes = b""
        for idx in s:
            decoded_bytes += self._vocab[idx]
        return decoded_bytes.decode("utf-8")

    def from_files(self, vocab_filepath, merge_filepath, special_tokens=None):
        pass

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buffer = ""
        tokens = []
        for char in iterable:
            while tokens:
                yield tokens.pop()
            buffer += char
            if buffer.encode("utf-8") in self.special_tokens:
                # Special token is found. return that
                encode_val = buffer.encode("utf-8")
                buffer = ""
                yield self.rev_vocab[encode_val]
            if buffer:
                if len(re.split(self.split_special_tokens_pattern, buffer)) <= 1:
                    continue
                tokens = self._tokenize(buffer)
                tokens.reverse()
                buffer = ""

        if buffer:
            last_tokens = self._tokenize(buffer, remove_end=False)
            last_tokens.reverse()
            tokens.extend(last_tokens)
        while tokens:
            yield tokens.pop()

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

    def train(
        self, filename: str, max_vocab_size: int, special_tokens: List[str] = None
    ):
        # initialize
        # TODO: Remove default <|endoftext|> special token
        self.__init__(special_tokens)
        self.max_vocab_size = max_vocab_size
        num_iterations = self.max_vocab_size - (len(self.special_tokens) + 256)

        self.merges = []  # List of merges that have taken place
        # Pre tokenize
        self._pre_tokenize(filename)
        # Generate the frequency index
        self._generate_byte_pair_index()
        # train
        for i in range(num_iterations):
            previous_vocab_len = len(self._vocab)
            self._merge()
            if previous_vocab_len == len(self._vocab):
                break
            if (i + 1) % 100 == 0:
                print("Iteration {}/{} completed".format(i + 1, num_iterations))
        for idx, val in enumerate(self._vocab):
            self.rev_vocab[val] = idx

    def _merge(self):

        max_pair = max(
            self.byte_pair_index,
            key=lambda x: (
                self.byte_pair_index.get(x),
                (self._vocab[x[0]], self._vocab[x[1]]),
            ),
        )
        if self.byte_pair_index[max_pair] == 0:
            return  # Nothing to merge
        self.merges_dict[max_pair] = len(self._vocab)
        self.merges.append(max_pair)
        self._vocab.append(self._vocab[max_pair[0]] + self._vocab[max_pair[1]])
        self._update_byte_pair_index(max_pair)

    def _generate_byte_pair_index(self) -> None:
        for word_split, val in self.pre_tokens_dict.items():
            for tok1, tok2 in zip(word_split[:-1], word_split[1:]):
                self.byte_pair_index[(tok1, tok2)] = (
                    self.byte_pair_index.get((tok1, tok2), 0) + val
                )

    def _update_byte_pair_index(self, max_pair: Tuple[int, int]) -> None:
        # Update the pair frequency dictionary
        # Buffer now
        previous_tokens = list(self.pre_tokens_dict.keys())
        vocab_len = len(self._vocab)
        for word_split in previous_tokens:
            # TODO: Test if adding condition of existence of max_pair[0] in word_split improves performance
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
            self.pre_tokens_dict[tuple(new_word_split)] = val
            # Update the byte pair frequency index
            for tok1, tok2 in zip(word_split[:-1], word_split[1:]):
                self.byte_pair_index[(tok1, tok2)] -= val
            for tok1, tok2 in zip(new_word_split[:-1], new_word_split[1:]):
                self.byte_pair_index[(tok1, tok2)] = (
                    self.byte_pair_index.get((tok1, tok2), 0) + val
                )
        assert (
            self.byte_pair_index[max_pair] == 0
        ), f"{max_pair=}, {self.byte_pair_index[max_pair]=}"

    def _tokenize(self, s: str, remove_end=True) -> List[int]:
        """
        Do not use this method. This is supposed to be used only when you are sure that `s` does not contain any special tokens
        """

        tokenization = []
        splits = re.split(self.split_special_tokens_pattern, s)
        if remove_end:
            splits = splits[:-1]
        else:
            splits += ["dummy"]
        for _split in splits[:-1]:
            pre_tokens = re.findall(PAT, _split)
            for pre_token in pre_tokens:
                tokenization += self._tokenize_pre_token(pre_token)
        if remove_end:
            tokenization += [self.rev_vocab[splits[-1].encode("utf-8")]]
        return tokenization

    def _tokenize_pre_token(self, pre_token: str):

        def get_pairs(bytes_list: List[int]) -> List[Tuple[int, int]]:
            return list(set(zip(bytes_list[:-1], bytes_list[1:])))

        byte_str = pre_token.encode("utf-8")
        tokens = [self.rev_vocab[bytes([i])] for i in byte_str]
        while len(tokens) >= 2:
            pairs = get_pairs(tokens)
            min_pair = min(pairs, key=lambda x: self.merges_dict.get(x, float("inf")))
            if min_pair not in self.merges:
                break
            token = self.merges_dict[min_pair]
            new_tokens = []
            idx = 0
            while idx < len(tokens) - 1:
                val1, val2 = tokens[idx], tokens[idx + 1]
                if val1 == min_pair[0] and val2 == min_pair[1]:
                    new_tokens.append(token)
                    idx += 2
                else:
                    new_tokens.append(val1)
                    idx += 1
            if idx == len(tokens) - 1:
                new_tokens.append(tokens[idx])
            tokens = new_tokens
        return tokens

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
    def pre_tokenize_chunk(s: str, special_tokens) -> Dict[Tuple[int, ...], int]:
        split_str = "|".join(map(re.escape, special_tokens))
        splits = re.split(split_str, s)
        tokenization_corpus = {}
        for split in splits:
            token_iter = re.finditer(PAT, split)
            for token in token_iter:
                encoded_tuple = get_encoded_byte_tuple(token.group())
                tokenization_corpus[encoded_tuple] = (
                    tokenization_corpus.get(encoded_tuple, 0) + 1
                )
        return tokenization_corpus

    @staticmethod
    def get_bpe_tokenizer(
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: List[str],
    ) -> "BytePairEncodingTokenizer":
        bpe_tokenizer = BytePairEncodingTokenizer(special_tokens=special_tokens or [])
        vocab_values = list(vocab.values())
        bpe_tokenizer._vocab = vocab_values + [
            special_token.encode("utf-8")
            for special_token in special_tokens
            if special_token.encode("utf-8") not in vocab_values
        ]
        merged_len = len(bpe_tokenizer._vocab)
        for merge in merges:
            bpe_tokenizer._vocab.append(merge[0] + merge[1])

        for idx, val in enumerate(bpe_tokenizer._vocab):
            bpe_tokenizer.rev_vocab[val] = idx
        for idx, merge in enumerate(merges):
            tup = (bpe_tokenizer.rev_vocab[merge[0]], bpe_tokenizer.rev_vocab[merge[1]])
            bpe_tokenizer.merges.append(tup)
            bpe_tokenizer.merges_dict[tup] = idx + merged_len
        return bpe_tokenizer
