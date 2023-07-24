import fire
import json
import numpy as np

from typing import *
from utils import Logger


def main(
    data_path: str = "./test.jsonl",
    threshold: int = -1,
    random: int = 0,
    log_path: str = 'inspect_jsonl.txt',
    random_rate: float = 0.5,
):
    logger = Logger(__name__, log_file=log_path, log_mode="file", disable_formatter=True)
    
    n = 0
    with open(data_path, "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                logger.info("Data has the following keys")
                obj = json.loads(line)
                logger.info(obj.keys())
            if threshold > 0 and n > threshold:
                break
            if random and np.random.randint(10) > 10 * random_rate:
                continue
            
            obj = json.loads(line)
            n += 1
            logger.info(f"========== Sample {i} ==========")
            if 'code' in obj:
                try:
                    code_splits = obj['code'].split("\n")
                    logger.info(f"Length of chars: {len(obj['code'])}, length of lines: {len(code_splits)}.")
                except:
                    pass
            for j, k in enumerate(obj.keys()):
                logger.info(f"** Key {j}: {k} **")
                logger.info(obj[k])
    print(f"Log saved in {log_path}")


if __name__ == "__main__":
    fire.Fire(main)