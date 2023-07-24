import os
import re
import sys
import fire
import json
import gzip
import glob
import numpy as np

from typing import *
from tqdm.auto import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from execution import check_correctness
from utils import Logger, IMPORT_HELPER, read_dataset, stream_jsonl_all, estimate_pass_at_k


LANGUAGE_NAME = {
   "CPP"        : "cpp",
   "Go"         : "go",
   "Java"       : "java",
   "JavaScript" : "js",
   "Python"     : "python",
   "Rust"       : "rust",
}


def postprocess_generation(sample, generation_mode="completion"):
    code = sample["generation"]
    if generation_mode == "instruction":
        if "```" in code:
            pattern = r'```(.*?)\n(.*?)```'
            matches = re.findall(pattern, code, re.DOTALL)
            for match in matches:
                code = match[1]
                break
    sample["generation"] = code
    
    return sample


def process_test(sample, problems, dataset_type, language_type, generation_mode):
    if dataset_type == "humanevalx":
        task_id = sample["task_id"]
        prompt = problems[task_id]["prompt"]
        test = problems[task_id]["test"]
        code = sample["generation"]
        
        # Pre-process for different languages
        if language_type == "python":
            test_setup = "\n".join(IMPORT_HELPER["python"]) + "\n"
            test_string = test_setup + prompt + code + "\n" + test + "\n"
        elif language_type == "cpp":
            test_set_up = ""
            for s in IMPORT_HELPER["cpp"]:
                if s not in prompt:
                    test_set_up += s + "\n"
            test_string = test_set_up + "\n" + prompt + code + "\n" + test
        elif language_type == "java":
            test_string = prompt + code + "\n" + test
        elif language_type == "js" or language_type == "javascript":
            test_string = prompt + code + "\n" + test
        elif language_type == "go":
            import_string = problems[task_id]["import"]
            prompt = prompt.replace(import_string, "")
            test = problems[task_id]["test"]
            test_setup = problems[task_id]["test_setup"]
            other_pkgs = []
            for pkg in IMPORT_HELPER["go"]:
                if pkg not in test_setup:
                    p = pkg.split("/")[-1]
                    if p + "." in code:
                        other_pkgs.append(f"\"{pkg}\"")
            if other_pkgs:
                import_other_pkgs = "import (\n" + "    ".join([p + "\n" for p in other_pkgs]) + ")"
                test_string = test_setup + "\n" + import_other_pkgs + "\n" + prompt + code + "\n" + test
            else:
                test_string = test_setup + "\n" + prompt + code + "\n" + test
        elif language_type == "rust":
            main = "\nfn main(){ \n } \n"
            test_string = main + prompt + code + test
    elif dataset_type == "mbpp":
        task_id = sample["task_id"]
        prompt = sample["prompt"]
        test = "\n".join(problems[task_id]["test_list"]) + "\n" + "\n".join(problems[task_id]["challenge_test_list"])
        code = sample["generation"]
        test_setup = "\n".join(IMPORT_HELPER["python"]) + "\n"
        test_string = test_setup + "\n" + prompt + code + "\n" + problems[task_id]["test_setup_code"] + "\n" + test + "\n"

    return test_string


def evaluate_functional_correctness(
    input_path: str = None,
    output_path: str = None,
    log_path: str = None,
    tmp_dir: str = "./",
    n_workers: int = 32,
    timeout: float = 5.0,
    k: List[int] = [1, 10, 100],
    model_name: str = None,
    problem_file: str = None,
    language_type: str = None,
    dataset_type: str = "humanevalx",
    generation_mode: str = "completion",
    test_groundtruth: bool = False,
):
    if log_path is None:
        log_path = os.path.join(output_path, "evaluation.log")
    logger = Logger(__name__, log_file=log_path)
    
    if os.path.isdir(input_path):
        input_list = glob.glob(input_path + '/*generation*.jsonl')
        sample_jsonl = []
        for input_file in input_list:
            sample_jsonl += stream_jsonl_all(input_file)
    else:
        input_file = input_path
        sample_jsonl = stream_jsonl_all(input_file)
    
    problems = read_dataset(problem_file, dataset_type=dataset_type)

    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:

        futures = []
        completion_id = Counter()
        n_samples = 0
        results = defaultdict(list)

        if test_groundtruth:
            logger.info("Testing ground truth...")
        else:
            logger.info("Testing generation...")
        for sample in sample_jsonl:
            task_id = sample["task_id"]
            if language_type is None:
                language_type = LANGUAGE_NAME[task_id.split("/")[0]]
            if test_groundtruth:
                if dataset_type == "humanevalx":
                    sample["generation"] = sample["canonical_solution"]
                    sample["prompt"] = problems[task_id]["prompt"]
                if dataset_type == "mbpp":
                    sample["generation"] = sample["code"]
                    sample["prompt"] = problems[task_id]["prompt"]
            sample = postprocess_generation(sample, generation_mode)
            sample["test_code"] = process_test(sample, problems, dataset_type, language_type, generation_mode)
            if sample["test_code"] is None:
                continue
            if "completion_id" in sample:
                completion_id_ = sample["completion_id"]
            else:
                completion_id_ = completion_id[task_id]
            args = (task_id, sample, language_type, timeout, tmp_dir, completion_id_)
            future = executor.submit(check_correctness, *args)
            futures.append(future)
            completion_id[task_id] += 1
            n_samples += 1

        if len(completion_id) == len(problems):
            evaluate_pass_at_k = True
        else:
            evaluate_pass_at_k = False

        logger.info("Running test suites...")
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            results[result["task_id"]].append((result["completion_id"], result))

    # Calculate pass@k.
    total, correct = [], []
    for result in results.values():
        passed = [r[1]["passed"] for r in result]
        total.append(len(passed))
        correct.append(sum(passed))
    total = np.array(total)
    correct = np.array(correct)
    if evaluate_pass_at_k:
        ks = k
        pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
                     for k in ks if (total >= k).all()}
        logger.info(pass_at_k)
    else:
        logger.info("Total: {}".format(np.sum(total)))
        logger.info("Correct: {}".format(np.sum(correct)))
        
    if test_groundtruth:
        out_file = os.path.join(output_path, "ground_truth.jsonl")
    else:    
        out_file = os.path.join(output_path, "result-" + input_file.split("/")[-2] + "." + input_file.split("/")[-1].split(".")[-1])
    
    logger.info("Writing to: {}".format(out_file))
    if out_file.endswith(".gz"):
        fp = gzip.GzipFile(fileobj=open(out_file, "wb"), mode="wb")
        for res in results.values():
            for r in res:
                fp.write((json.dumps(r[1], ensure_ascii=False) + "\n").encode("utf-8"))
    else:
        fp = open(out_file, 'w')
        for res in results.values():
            for r in res:
                fp.write(json.dumps(r[1], ensure_ascii=False) + "\n")
    fp.close()

    if test_groundtruth:
        logger.info("Ground-truth test finished.")
    else:
        logger.info("Evaluation finished.")


def main():
    fire.Fire(evaluate_functional_correctness)


if __name__ == "__main__":
    sys.exit(main())
