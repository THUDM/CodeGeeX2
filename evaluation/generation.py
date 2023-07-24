import os
import zmq
import time
import json
import torch
import random
import socket
import argparse

from typing import *
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, StoppingCriteria
from utils import Logger, read_dataset, process_extra_prompt, is_code_generation_finished, cleanup_code

logger = Logger(__name__)


def add_code_generation_specific_args(parser):
    group = parser.add_argument_group("Code Generation")
    group.add_argument(
        "--hostfile",
        type=str,
        default="./hostfile",
    )
    group.add_argument(
        "--channel-ip",
        type=str,
        default=None,
        help="IP for ZeroMQ channel",
    )
    group.add_argument(
        "--channel-port",
        type=int,
        default=5555,
        help="Port for ZeroMQ channel",
    )
    group.add_argument(
        "--master-port",
        type=int,
        default=6007,
        help="Port for distributed channel",
    )
    group.add_argument(
        "--model-per-device",
        type=int,
        default=1,
        help="Number of models per device",
    )
    group.add_argument(
        "--max-length",
        type=int,
        default=8192,
        help="Max sequence length",
    )
    group.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p Probability for sampling",
    )
    group.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top-k for sampling",
    )
    group.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for sampling",
    )
    group.add_argument(
        "--greedy",
        type=int,
        default=0,
        help="Use greedy decoding instead of sampling",
    )
    group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    group.add_argument(
        "--micro-batch-size",
        type=int,
        default=1,
        help="Micro batch size for each GPU",
    )
    group.add_argument(
        "--samples-per-problem",
        type=int,
        default=200,
        help="Number of samples to generate for each problem",
    )
    group.add_argument(
        "--gen-node-world-size",
        type=int,
        default=1,
        help="Number of machines to use for generation",
    )
    group.add_argument(
        '--task-name',
        default="generation",
        help='Name of task',
    )
    group.add_argument(
        '--model-name',
        default="codegeex2-6b",
        help='Name of model, support ["codegeex2-6b", "starcoder", "replit-code-v1-3b", "codegen25-7b-multi", "codegen25-7b-mono", "codegen-16B-multi"]',
    )
    group.add_argument(
        '--data-path',
        required=True,
    )
    group.add_argument(
        '--output-path',
        required=True,
    )
    group.add_argument(
        '--log-path',
        default=None,
        help='Path to log output',
    )
    group.add_argument(
        '--model-path',
        required=True,
    )
    group.add_argument(
        '--dataset-type',
        default="humanevalx",
        help='Identify the evaluation dataset [humanevalx]',
    )
    group.add_argument(
        '--language-type',
        default="python",
        help='Identify the type of programming language to generate',
    )
    group.add_argument(
        '--generation-mode',
        default="instruction",
    )


class CodeStoppingCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the full generated number of tokens exceeds `max_length` or meet the code generation stopping criteria.
    """

    def __init__(
        self, 
        max_length: int, 
        micro_batch_size: int, 
        tokenizer,
        dataset_type: str, 
        language_type: str, 
        prompt: str,
    ):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.dataset_type = dataset_type
        self.language_type = language_type
        self.prompt = prompt
        self.stop_index = [-1 for _ in range(micro_batch_size)]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for i, input_id in enumerate(input_ids):
            if self.stop_index[i] > -1:
                continue
            code = self.tokenizer.decode(input_id)
            code = code[len(self.prompt):]
            if is_code_generation_finished(
                code,
                dataset_type=self.dataset_type,
                language_type=self.language_type) or input_id.shape[-1] >= self.max_length:
                self.stop_index[i] = len(code) + len(self.prompt)
        if all([s != -1 for s in self.stop_index]):
            return True
        
        return False


def run_generation_distributed(args, model, tokenizer):
    logger.info(f"Connecting to tcp://{args.channel_ip}:{args.channel_port}")
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://{args.channel_ip}:{args.channel_port}")
    
    os.makedirs(args.output_path, exist_ok=True)
    output_path = os.path.join(
        args.output_path,
        f"{args.task_name}-t{args.temperature}-topp{args.top_p}-ns{args.samples_per_problem}-rank{args.rank}.jsonl",
    )
    
    def process(obj):
        results = []
        prompt = obj["prompt"]
        if args.generation_mode == "instruction":
            inputs = tokenizer([prompt] * args.micro_batch_size, return_tensors="pt")
            inputs = inputs.to(model.device)
            outputs = model.generate(**inputs,
                                    max_length=args.max_length,
                                    do_sample=True if not args.greedy else False,
                                    use_cache=True,
                                    top_p=args.top_p,
                                    top_k=args.top_k,
                                    temperature=args.temperature,
                                    pad_token_id=tokenizer.eos_token_id)
            for i, output in enumerate(outputs):
                response = tokenizer.decode(output)
                res = obj.copy()
                res["generation"] = response[len(prompt):].strip()
                results.append(res)
        elif args.generation_mode == "completion":
            inputs = tokenizer([prompt for _ in range(args.micro_batch_size)], return_tensors="pt")
            inputs = inputs.to(model.device)
            stop_criteria = CodeStoppingCriteria(
                max_length=args.max_length,
                micro_batch_size=args.micro_batch_size,
                tokenizer=tokenizer,
                dataset_type=args.dataset_type,
                language_type=args.language_type,
                prompt=prompt)
            outputs = model.generate(**inputs,
                                    max_length=args.max_length,
                                    do_sample=True if not args.greedy else False,
                                    use_cache=True,
                                    stopping_criteria=[stop_criteria],
                                    top_p=args.top_p,
                                    top_k=args.top_k,
                                    temperature=args.temperature,
                                    pad_token_id=tokenizer.eos_token_id)
            for i, output in enumerate(outputs):
                response = tokenizer.decode(output)
                res = obj.copy()
                res["generation_raw"] = response
                res["generation"] = cleanup_code(
                    response[len(prompt):], 
                    dataset_type=args.dataset_type,
                    language_type=args.language_type)
                results.append(res)
        
        return results
    
    fout = open(output_path, "w", encoding="utf-8")
    while True:
        socket.send_json({"rank": args.rank, "action": "pull"})
        resp = socket.recv_json()
        try:
            if resp["task_id"] is None:
                break

            current_spec = resp["task_id"]
            results = process(current_spec)
            
            for res in results:
                fout.write(json.dumps(res, ensure_ascii=False) + "\n")
                fout.flush()

            socket.send_json(
                {
                    "rank"   : args.rank,
                    "action" : "success",
                    "task_id": current_spec['task_id']
                }
            )
            socket.recv()

        except Exception as e:
            logger.error(f"*** (rank={args.rank}) crashed.")
            logger.error(f"    error: {repr(e)}")
            socket.send_json(
                {
                    "rank"   : args.rank,
                    "action" : "fail",
                    "task_id": current_spec['task_id']
                }
            )
            socket.recv()
            continue


def main(args, node_rank: int, local_rank: int, master_port: int, num_devices: int):
    world_size = args.gen_node_world_size * num_devices
    args.rank = num_devices * node_rank + local_rank
    args.world_size = world_size
    logger.info(f"Generating on rank {args.rank} of {args.world_size}")
    
    try:
        if args.model_name in ["codegeex2-6b"]:
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, clean_up_tokenization_spaces=False, trust_remote_code=True)
        if args.model_name in ["codegeex2-6b"]:
            model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True).to("cuda:{}".format(local_rank % torch.cuda.device_count()))
        elif args.model_name in ["starcoder", "replit-code-v1-3b", "codegen25-7b-multi", "codegen25-7b-mono", "codegen-16B-multi"]:
            model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True).to("cuda:{}".format(local_rank % torch.cuda.device_count()))
        else:
            try:
                model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True).to("cuda:{}".format(local_rank % torch.cuda.device_count()))
            except:
                logger.error(f"Model {args.model_name} not supported.")
                raise NotImplementedError
    except Exception as e:
        logger.error(e)
    
    model = model.eval()
    # Generate samples.
    run_generation_distributed(args, model, tokenizer)

    logger.info(f"rank={args.rank} worker finished, waiting ...")
    exit(0)


def server(args):
    logger.info(f"[ server ] starting ...")
    entries = read_dataset(args.data_path, dataset_type=args.dataset_type)

    assert args.samples_per_problem % args.micro_batch_size == 0, "samples_per_problem should be divisible by batch_size"

    for entry in entries.values():
        entry["prompt"] = process_extra_prompt(
            entry["prompt"], 
            language_type=args.language_type, 
            dataset_type=args.dataset_type, 
            generation_mode=args.generation_mode,
        )

    res = []
    for entry in entries.values():
        res.extend([entry] * (args.samples_per_problem // args.micro_batch_size))
    random.shuffle(res)
    all_entries = res

    # setup zeromq channel
    logger.info(f"[ server ] starting up on port {args.channel_port}")
    context = zmq.Context()
    logger.info(f"[ server ] creating socket")
    socket = context.socket(zmq.REP)
    logger.info(f"[ server ] binding to port {args.channel_port}")
    socket.bind(f"tcp://*:{args.channel_port}")

    logger.info(
        f"[ server ] loaded {len(entries)} entries, generating {len(entries) * args.samples_per_problem} samples",
    )

    remaining_entries = all_entries.copy()
    running_workers = args.gen_node_world_size * torch.cuda.device_count()
    num_finished = 0

    logger.info(f"[ server ] listening for requests ...")
    start_time = time.perf_counter()
    while True:
        # Wait for next request from client
        msg = socket.recv_json()
        rank = msg["rank"]
        action = msg["action"]

        if action == "pull":
            if len(remaining_entries) == 0:
                socket.send_json({"task_id": None})
                running_workers -= 1
                logger.info(f"[ server ] Shutting down worker {rank}, remaining {running_workers} workers")
                if running_workers == 0 and num_finished == len(all_entries):
                    logger.info(f"[ server ] All workers finished")
                    break
            else:
                entry = remaining_entries.pop()
                time_elapsed = time.perf_counter() - start_time
                logger.info(f"[ server ] Sending entry {entry['task_id']} to worker {rank}")
                remaining = (
                        len(remaining_entries)
                        / (len(all_entries) - len(remaining_entries))
                        * time_elapsed
                )
                time_per_sampple = 0.0 if num_finished == 0 else time_elapsed / num_finished / args.micro_batch_size
                logger.info(
                    f"[ server ] total {len(all_entries)}, assigned {len(all_entries) - len(remaining_entries)}, finished {num_finished}, elapsed {time_elapsed:.4f}, speed {time_per_sampple:.4f}s/sample, remaining {remaining:.4f}",
                )
                socket.send_json({"task_id": entry})
        else:
            if action == "success":
                logger.info(f"[ server ] {msg['task_id']} is finished")
                socket.send_json({"pong": 1})
            else:
                logger.info(f"[ server ] {msg['task_id']} is not finished")
                remaining_entries.append(msg['task_id'])
                socket.send_json({"pong": 1})
                break

            num_finished += 1


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    parser = argparse.ArgumentParser()
    add_code_generation_specific_args(parser)
    args = parser.parse_args()
    
    if args.log_path is None:
        args.log_path = os.path.join(args.output_path, "generation.log")

    logger.info("start method: " + torch.multiprocessing.get_start_method())
    
    processes = []
    num_devices = torch.cuda.device_count()
    hosts = open(args.hostfile, "r").readlines()
    hosts = [host.strip() for host in hosts]
    master_port = args.master_port

    node_rank = None
    for i in range(len(hosts)):
        if hosts[i] == socket.gethostbyname(socket.gethostname()):
            node_rank = i
            break
    assert (
            node_rank is not None
    ), f"Could not find hostname ({socket.gethostbyname(socket.gethostname())}) in hostlist"

    # launch server
    if socket.gethostbyname(socket.gethostname()) == hosts[0]:
        server_process = torch.multiprocessing.Process(target=server, args=(args,))
        logger.info(f"Launching server ...")
        server_process.start()
        processes.append(server_process)

    for i in range(num_devices):
        local_rank = i
        logger.info(f"launching local rank {i}")

        p = torch.multiprocessing.Process(
            target=main,
            args=(args, node_rank, local_rank, master_port, num_devices),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
