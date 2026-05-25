from tests.adapters import run_train_bpe
import logging
import time
from pathlib import Path

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    path = Path("data/TinyStoriesV2-GPT4-train.txt")
    start_time = time.time()
    logging.info("Starting BPE training on %s", path)
    run_train_bpe(path, 10_000, ["<|endoftext|>"])
    logging.info("Finished BPE training in %.2f seconds", time.time() - start_time)
