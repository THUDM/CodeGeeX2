import re
import json
import gzip
import torch
import numpy
import random
import logging
import itertools
import numpy as np

from typing import *


LANGUAGE_TAG = {
    "c"            : "// language: C",
    "c++"          : "// language: C++",
    "cpp"          : "// language: C++",
    "c#"           : "// language: C#",
    "csharp"       : "// language: C#",
    "c-sharp"      : "// language: C#",
    "css"          : "/* language: CSS */",
    "cuda"         : "// language: Cuda",
    "dart"         : "// language: Dart",
    "lua"          : "// language: Lua",
    "objectivec"   : "// language: Objective-C",
    "objective-c"  : "// language: Objective-C",
    "objective-c++": "// language: Objective-C++",
    "python"       : "# language: Python",
    "perl"         : "# language: Perl",
    "prolog"       : f"% language: Prolog",
    "swift"        : "// language: swift",
    "lisp"         : "; language: Lisp",
    "java"         : "// language: Java",
    "scala"        : "// language: Scala",
    "tex"          : f"% language: TeX",
    "vue"          : "<!--language: Vue-->",
    "markdown"     : "<!--language: Markdown-->",
    "html"         : "<!--language: HTML-->",
    "php"          : "// language: PHP",
    "js"           : "// language: JavaScript",
    "javascript"   : "// language: JavaScript",
    "typescript"   : "// language: TypeScript",
    "go"           : "// language: Go",
    "shell"        : "# language: Shell",
    "rust"         : "// language: Rust",
    "sql"          : "-- language: SQL",
    "kotlin"       : "// language: Kotlin",
    "vb"           : "' language: Visual Basic",
    "ruby"         : "# language: Ruby",
    "pascal"       : "// language: Pascal",
    "r"            : "# language: R",
    "fortran"      : "!language: Fortran",
    "lean"         : "-- language: Lean",
    "matlab"       : f"% language: Matlab",
    "delphi"       : "{language: Delphi}",
    "scheme"       : "; language: Scheme",
    "basic"        : "' language: Basic",
    "assembly"     : "; language: Assembly",
    "groovy"       : "// language: Groovy",
    "abap"         : "* language: Abap",
    "gdscript"     : "# language: GDScript",
    "haskell"      : "-- language: Haskell",
    "julia"        : "# language: Julia",
    "elixir"       : "# language: Elixir",
    "excel"        : "' language: Excel",
    "clojure"      : "; language: Clojure",
    "actionscript" : "// language: ActionScript",
    "solidity"     : "// language: Solidity",
    "powershell"   : "# language: PowerShell",
    "erlang"       : f"% language: Erlang",
    "cobol"        : "// language: Cobol",
    "alloy"        : "/* language: Alloy */",
    "awk"          : "// language: AWK",
    "thrift"       : "/* language: Thrift */",
    "sparql"       : "# language: SPARQL",
    "augeas"       : "// language: Augeas",
    "cmake"        : "# language: CMake",
    "f-sharp"      : "// language: F#",
    "stan"         : "// language: Stan",
    "isabelle"     : "(*language: Isabelle*)",
    "dockerfile"   : "# language: Dockerfile",
    "rmarkdown"    : "# language: RMarkdown",
    "literate-agda": "-- language: Literate Agda",
    "tcl"          : "// language: Augeas",
    "glsl"         : "// language: GLSL",
    "antlr"        : "// language: ANTLR",
    "verilog"      : "// language: Verilog",
    "racket"       : "; language: Racket",
    "standard-ml"  : "(*language:Standard ML*)",
    "elm"          : "-- language: Elm",
    "yaml"         : "# language: YAML",
    "smalltalk"    : "'' language: Smalltalk",
    "ocaml"        : "(*language: OCaml*)",
    "idris"        : "-- language: Idris",
    "visual-basic" : "' language: Visual Basic",
    "protocol-buffer": "// language: Protocol Buffer",
    "bluespec"     : "// language: Bluespec",
    "applescript"  : "-- language: AppleScript",
    "makefile"     : "# language: Makefile",
    "tcsh"         : "# language: TCSH",
    "maple"        : "# language: Maple",
    "systemverilog": "// language: SystemVerilog",
    "literate-coffeescript": "# language: Literate CoffeeScript",
    "vhdl"         : "-- language: VHDL",
    "restructuredtext": ".. language: reStructuredText",
    "sas"          : "* language: SAS",
    "literate-haskell": "> language: Literate Haskell",
    "java-server-pages": "// language: Java Server Pages",
    "coffeescript" : "# language: CoffeeScript",
    "emacs-lisp"   : "; language: Emacs Lisp",
    "mathematica"  : "// language: Mathematica",
    "xslt"         : "<!--language: XSLT-->",
    "zig"          : "// language: Zig",
    "common-lisp"  : "; language: Common Lisp",
    "stata"        : "* language: Stata",
    "agda"         : "-- language: Agda",
    "ada"          : "-- language: Ada",
}


LANGUAGE_COMMENT_SIGN = {}
for lang in LANGUAGE_TAG:
    LANGUAGE_COMMENT_SIGN[lang] = LANGUAGE_TAG[lang].split("language:")[0].strip()


IMPORT_HELPER = {
    "python": [
        "import math",
        "import re",
        "import sys",
        "import copy",
        "import datetime",
        "import itertools",
        "import collections",
        "import heapq",
        "import statistics",
        "import functools",
        "import hashlib",
        "import numpy",
        "import numpy as np",
        "import string",
        "from typing import *",
        "from collections import *",
    ],
    "go"    : [
        "math",
        "strings",
        "fmt",
        "strconv",
        "time",
        "bytes",
        "regexp",
        "sort",
        "math/rand",
        "crypto/md5",
    ],
    "cpp"   : [
        "#include<stdlib.h>",
        "#include<algorithm>",
        "#include<math.h>",
        "#include<stdio.h>",
        "#include<vector>",
        "#include<string>",
        "#include<climits>",
        "#include<cstring>",
        "#include<iostream>",
    ],
}



def set_random_seed(seed):
    """Set random seed for reproducability."""
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    
    
def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, "rt") as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)
                    

def stream_jsonl_all(filename: str) -> Iterable[Dict]:
    results = []
    if filename.endswith(".gz"):
        fp = gzip.open(open(filename, "rb"), "rt")
    else:
        fp = open(filename, "r")
    for line in fp:
        if any(not x.isspace() for x in line):
            results.append(json.loads(line))
    fp.close()

    return results


def read_dataset(
    data_file: str = None,
    dataset_type: str = "humanevalx",
) -> Dict:
    if "humanevalx" in dataset_type.lower():
        dataset = {task["task_id"]: task for task in stream_jsonl(data_file)}
    elif "mbpp" in dataset_type.lower():
        problems = {task["task_id"]: task for task in stream_jsonl(data_file)}
        task_ids = sorted(problems.keys())[10:510]
        dataset = {}
        for task_id in task_ids:
            sample = problems[task_id]
            description = sample["text"]
            test_example = sample["test_list"][0]
            prompt = f'"""\n{description}\n{test_example}\n"""\n'
            sample["prompt"] = prompt
            dataset[task_id] = sample
    elif "ds1000" in dataset_type.lower():
        # install ds1000 from https://github.com/HKUNLP/DS-1000
        from ds1000 import DS1000Dataset
        ds1000 = DS1000Dataset(source_dir=data_file, libs="all", mode="Completion")
        for lib in ds1000.libs:
            for problem_id in range(len(ds1000[lib])):
                prefix = ""
                suffix = ""
                insert_flag = False
                first_line_flag = True
                # extract prefix and suffix of the prompt
                for line in ds1000[lib][problem_id]["prompt"].split("\n"):
                    if "[insert]" in line:
                        insert_flag = True
                        continue
                    if first_line_flag:
                        first_line_flag = False
                    else:
                        line = "\n" + line
                    if not insert_flag:
                        prefix += line
                    else:
                        suffix += line
            
    else:
        raise f"Dataset: {dataset_type} not supported."

    return dataset


def read_translation_dataset(
    data_file_src: str = None,
    data_file_tgt: str = None,
    lang_src: str = None,
    lang_tgt: str = None,
    dataset_type: str = "humanevalx",
) -> Dict:
    if "humanevalx" in dataset_type.lower():
        dataset_src = {task["task_id"]: task for task in stream_jsonl(data_file_src)}
        dataset_tgt = {task["task_id"].split("/")[-1]: task for task in stream_jsonl(data_file_tgt)}
        for k, sample in dataset_src.items():
            prompt = "code translation\n"
            if lang_src == "cpp":
                prompt += "C++:\n"
            elif lang_src == "js":
                prompt += "JavaScript:\n"
            else:
                prompt += f"{lang_src}:\n".capitalize()
            prompt += dataset_src[k]["declaration"] + "\n" + dataset_src[k]["canonical_solution"].rstrip() + "\n"
            if lang_tgt == "cpp":
                prompt += "C++:\n"
            elif lang_tgt == "js":
                prompt += "JavaScript:\n"
            else:
                prompt += f"{lang_tgt}:\n".capitalize()
            prompt += dataset_tgt[k.split("/")[-1]]["declaration"]
            dataset_src[k]["prompt"] = prompt
    else:
        raise f"Dataset: {dataset_type} not supported."

    return dataset_src


def process_extra_prompt(
    prompt: str,
    language_type: str = "python", 
    dataset_type: str = None,
    generation_mode: str = "completion",
) -> str:
    """
    Processes the extra prompt.
    """
    language = language_type.lower()
    if dataset_type == "humanevalx":
        extra_prompt = ""
        # extra_prompt = LANGUAGE_TAG[language] + "\n"
        prompt = prompt.strip()
        if generation_mode == "instruction":
            return "问：" + extra_prompt + prompt + "\n答："
        return extra_prompt + prompt
    elif dataset_type == "mbpp":
        extra_prompt = ""
        prompt = prompt.strip()
        return extra_prompt + prompt
    else:
        return prompt


def is_code_generation_finished(
    code: str,
    dataset_type: str = None,
    language_type: str = None,
):
    """
    Checks whether the generated code is finished.
    """
    if dataset_type == "mbpp":
        end_words = ["\ndef", "\nassert"]
        for w in end_words:
            if w == "\ndef":
                if code.count(w) > 1:
                    return True
            else:
                if w in code:
                    return True
    else:
        if language_type.lower() == "python":
            for line in code.split("\n"):
                if len(line.strip()) > 0 and line[0] != ' ' and line[0] != '\t':
                    return True
            end_words = ["\ndef", "\nclass", "\nif", "\n#", "\nprint"]
            for w in end_words:
                if w in code:
                    return True
        elif language_type.lower() == "java":
            if code.count("{") + 1 == code.count("}"):
                return True
        elif language_type.lower() == "go":
            if "\nfunc main(" in code:
                return True
            if code.count("{") + 1 == code.count("}"):
                return True
        elif language_type.lower() == "js":
            if code.count("{") + 1 == code.count("}"):
                return True
        elif language_type.lower() == "cpp":
            if "\nint main()" in code:
                return True
            if code.count("{") + 1 == code.count("}"):
                return True
        elif language_type.lower() == "rust":
            if "\nfn main()" in code:
                return True
            if code.count("{") + 1 == code.count("}"):
                return True

    return False


# Modified from https://github.com/bigcode-project/bigcode-evaluation-harness/blob/main/lm_eval/tasks/mbpp.py
stop_words=["\nclass", "\nassert", '\n"""', "\nprint", "\nif"]
def first_block(string, stop_words):
    """Split off first block of code by scanning for class, def etc. on newlines."""
    return re.split("|".join(stop_words), string)[0].rstrip()


def cleanup_code(
    code: str,
    dataset_type: str = None,
    language_type: str = None,
):
    """
    Cleans up the generated code.
    """
    if dataset_type == "mbpp":
        end_words = ["\nassert", "\ndef"]
        for w in end_words:
            if w == "\ndef":
                if code.count(w) > 1:
                    code = code[:code.rfind(w)]
            else:
                code = code[:code.rfind(w)]
        code = first_block(code, stop_words)
    elif dataset_type == "humanevalx":
        if language_type.lower() == "python":
            code_splits = code.split("\n")
            is_empty_line = False
            ind_empty_line = None
            for i, line in enumerate(code_splits):
                if len(line.strip()) > 0 and line[0] != ' ' and line[0] != '\t':
                    is_empty_line = True
                    ind_empty_line = i
                    break
            if is_empty_line:
                code = "\n".join(code_splits[:ind_empty_line])
            else:
                end_words = ["\ndef", "\nclass", "\n#", "\nassert", '\n"""', "\nprint", "\nif", "\n\n\n"]
                for w in end_words:
                    if w in code:
                        code = code[:code.rfind(w)]
        elif language_type.lower() == "java":
            main_pos = code.find("public static void main")
            if main_pos != -1:
                code = code[:main_pos] + '}'
            if '}' in code:
                code = code[:code.rfind('}')] + '}'
            if code.count('{') + 1 == code.count('}'):
                code += "\n}"
        elif language_type.lower() == "go":
            if "\nfunc main(" in code:
                code = code[:code.rfind("func main(")]
            if '}' in code:
                code = code[:code.rfind('}')] + '}'
        elif language_type.lower() == "cpp":
            if "\nint main()" in code:
                code = code[:code.rfind("int main()")]
            if '}' in code:
                code = code[:code.rfind('}')] + '}'
        elif language_type.lower() == "js":
            if '}' in code:
                code = code[:code.rfind('}')] + '}'
        elif language_type.lower() == "rust":
            if '}' in code:
                code = code[:code.rfind('}')] + '}'

    return code


def estimate_pass_at_k(
        num_samples: Union[int, List[int], np.ndarray],
        num_correct: Union[List[int], np.ndarray],
        k: int
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


class Logger:
    def __init__(self, name, log_level=logging.INFO, log_file=None, log_mode="both", disable_formatter=False):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)

        self.formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        
        # Log to console
        if log_mode == "both" or log_mode == "terminal":
            console_handler = logging.StreamHandler()
            if not disable_formatter:
                console_handler.setFormatter(self.formatter)
            self.logger.addHandler(console_handler)

        # Log to file
        if log_file is not None:
            if log_mode == "both" or log_mode == "file":
                file_handler = logging.FileHandler(log_file, mode='w')
                if not disable_formatter:
                    file_handler.setFormatter(self.formatter)
                self.logger.addHandler(file_handler)

    def add_file_handler(self, file_name):
        file_handler = logging.FileHandler(file_name, mode='w')
        file_handler.setFormatter(self.formatter)
        self.logger.addHandler(file_handler)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)
