from cs336_basics.byte_pair_encoding_tokenizer import BytePairEncodingTokenizer
import time

# from cs336_basics.bpe_new import BytePairEncodingTokenizer
import json


def tuple_to_str(tup):
    return bytes(tup).decode("utf-8")


if __name__ == "__main__":
    t1 = time.time()
    bpe = BytePairEncodingTokenizer()
    bpe.train("tests/fixtures/tinystories_sample_5M.txt", 10000, ["<|endoftext|>"])
    t2 = time.time()
    print(len(bpe.vocab))
    print("Time", t2 - t1)
    # tokenization_corpus = bpe.pre_tokens_dict
    # token_vs_count = {}
    # for _tup, count in tokenization_corpus.items():
    #     token_vs_count[tuple_to_str(_tup)] = count
    # print(json.dumps(token_vs_count, indent=2))
