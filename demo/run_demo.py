import os
import json
import numpy
import torch
import random
import argparse
import gradio as gr

from transformers import AutoTokenizer, AutoModel

try:
    # Should first install fastllm (https://github.com/ztxz16/fastllm.git)
    from fastllm_pytools import llm
    enable_fastllm = True
except:
    print("fastllm disabled.")
    enable_fastllm = False

try:
    from gpus import load_model_on_gpus
    enable_multiple_gpus = True
except:
    print("Multiple GPUs support disabled.")
    enable_multiple_gpus = False

try:
    import chatglm_cpp
    enable_chatglm_cpp = True
except:
    print("[WARN] chatglm-cpp not found. Install it by `pip install chatglm-cpp` for better performance. "
          "Check out https://github.com/li-plus/chatglm.cpp for more details.")
    enable_chatglm_cpp = False


def get_model(args):
    if not args.cpu:
        if torch.cuda.is_available():
            device = f"cuda:{args.gpu}"
        elif torch.backends.mps.is_built():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    if args.n_gpus > 1 and enable_multiple_gpus:
        # 如需实现多显卡模型加载,传入"n_gpus"为需求的显卡数量 / To enable Multiple GPUs model loading, please adjust "n_gpus" to the desired number of graphics cards.
        print(f"Runing on {args.n_gpus} GPUs.")
        model = load_model_on_gpus(args.model_path, num_gpus=args.n_gpus)
        model = model.eval()
    elif enable_chatglm_cpp and args.chatglm_cpp:
        print("Using chatglm-cpp to improve performance")
        dtype = "f16"
        if args.quantize in [4, 5, 8]:
            dtype = f"q{args.quantize}_0"
        model = chatglm_cpp.Pipeline(args.model_path, dtype=dtype)
    else:
        model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True)
        model = model.eval()

        if enable_fastllm and args.fastllm:
            print("fastllm enabled.")
            model = model.half()
            llm.set_device_map(device)
            if args.quantize in [4, 8]:
                model = llm.from_hf(model, dtype=f"int{args.quantize}")
            else:
                model = llm.from_hf(model, dtype="float16")
        else:
            print("chatglm-cpp and fastllm not installed, using transformers.")
            if args.quantize in [4, 8]:
                print(f"Model is quantized to INT{args.quantize} format.")
                model = model.half().quantize(args.quantize)
            model = model.to(device)

    return tokenizer, model


def add_code_generation_args(parser):
    group = parser.add_argument_group(title="CodeGeeX2 DEMO")
    group.add_argument(
        "--model-path",
        type=str,
        default="THUDM/codegeex2-6b",
    )
    group.add_argument(
        "--example-path",
        type=str,
        default=None,
    )
    group.add_argument(
        "--quantize",
        type=int,
        default=None,
    )
    group.add_argument(
        "--chatglm-cpp",
        action="store_true",
    )
    group.add_argument(
        "--fastllm",
        action="store_true",
    )
    group.add_argument(
        "--n-gpus",
        type=int,
        default=1,
    )
    group.add_argument(
        "--gpu",
        type=int,
        default=0,
    )
    group.add_argument(
        "--cpu",
        action="store_true",
    )
    group.add_argument(
        "--listen",
        type=str,
        default="127.0.0.1",
    )
    group.add_argument(
        "--port",
        type=int,
        default=7860,
    )
    group.add_argument(
        "--username",
        type=str,
        default=None,
    )
    group.add_argument(
        "--password",
        type=str,
        default=None,
    )
    group.add_argument(
        "--auth",
        action="store_true",
    )
    
    
    return parser


# 更完编程语言列表请查看 evaluation/utils.py / Full list of supported languages in evaluation/utils.py
LANGUAGE_TAG = {
    "Abap"         : "* language: Abap",
    "ActionScript" : "// language: ActionScript",
    "Ada"          : "-- language: Ada",
    "Agda"         : "-- language: Agda",
    "ANTLR"        : "// language: ANTLR",
    "AppleScript"  : "-- language: AppleScript",
    "Assembly"     : "; language: Assembly",
    "Augeas"       : "// language: Augeas",
    "AWK"          : "// language: AWK",
    "Basic"        : "' language: Basic",
    "C"            : "// language: C",
    "C#"           : "// language: C#",
    "C++"          : "// language: C++",
    "CMake"        : "# language: CMake",
    "Cobol"        : "// language: Cobol",
    "CSS"          : "/* language: CSS */",
    "CUDA"         : "// language: Cuda",
    "Dart"         : "// language: Dart",
    "Delphi"       : "{language: Delphi}",
    "Dockerfile"   : "# language: Dockerfile",
    "Elixir"       : "# language: Elixir",
    "Erlang"       : f"% language: Erlang",
    "Excel"        : "' language: Excel",
    "F#"           : "// language: F#",
    "Fortran"      : "!language: Fortran",
    "GDScript"     : "# language: GDScript",
    "GLSL"         : "// language: GLSL",
    "Go"           : "// language: Go",
    "Groovy"       : "// language: Groovy",
    "Haskell"      : "-- language: Haskell",
    "HTML"         : "<!--language: HTML-->",
    "Isabelle"     : "(*language: Isabelle*)",
    "Java"         : "// language: Java",
    "JavaScript"   : "// language: JavaScript",
    "Julia"        : "# language: Julia",
    "Kotlin"       : "// language: Kotlin",
    "Lean"         : "-- language: Lean",
    "Lisp"         : "; language: Lisp",
    "Lua"          : "// language: Lua",
    "Markdown"     : "<!--language: Markdown-->",
    "Matlab"       : f"% language: Matlab",
    "Objective-C"  : "// language: Objective-C",
    "Objective-C++": "// language: Objective-C++",
    "Pascal"       : "// language: Pascal",
    "Perl"         : "# language: Perl",
    "PHP"          : "// language: PHP",
    "PowerShell"   : "# language: PowerShell",
    "Prolog"       : f"% language: Prolog",
    "Python"       : "# language: Python",
    "R"            : "# language: R",
    "Racket"       : "; language: Racket",
    "RMarkdown"    : "# language: RMarkdown",
    "Ruby"         : "# language: Ruby",
    "Rust"         : "// language: Rust",
    "Scala"        : "// language: Scala",
    "Scheme"       : "; language: Scheme",
    "Shell"        : "# language: Shell",
    "Solidity"     : "// language: Solidity",
    "SPARQL"       : "# language: SPARQL",
    "SQL"          : "-- language: SQL",
    "Swift"        : "// language: swift",
    "TeX"          : f"% language: TeX",
    "Thrift"       : "/* language: Thrift */",
    "TypeScript"   : "// language: TypeScript",
    "Vue"          : "<!--language: Vue-->",
    "Verilog"      : "// language: Verilog",
    "Visual Basic" : "' language: Visual Basic",
}


def set_random_seed(seed):
    """Set random seed for reproducability."""
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser = add_code_generation_args(parser)
    args, _ = parser.parse_known_args()

    tokenizer, model = get_model(args)

    examples = []
    if args.example_path is None:
        example_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], "example_inputs.jsonl")
    else:
        example_path = args.example_path

    # Load examples for gradio DEMO
    with open(example_path, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(list(json.loads(line).values()))


    def predict(
        prompt, 
        lang,
        seed, 
        out_seq_length, 
        temperature, 
        top_k, 
        top_p,
    ):
        set_random_seed(seed)
        if lang != "None":
            prompt = LANGUAGE_TAG[lang] + "\n" + prompt
        
        if enable_fastllm and args.fastllm:
            model.direct_query = True
            outputs = model.chat(tokenizer, 
                                 prompt,
                                 max_length=out_seq_length,
                                 top_p=top_p,
                                 top_k=top_k,
                                 temperature=temperature)
            response = prompt + outputs[0]
        elif enable_chatglm_cpp and args.chatglm_cpp:
            inputs = tokenizer([prompt], return_tensors="pt")
            pipeline = model
            outputs = pipeline.generate(prompt,
                                        max_length=inputs['input_ids'].shape[-1] + out_seq_length,
                                        do_sample=temperature > 0,
                                        top_p=top_p,
                                        top_k=top_k,
                                        temperature=temperature)
            response = prompt + outputs
        else:
            inputs = tokenizer([prompt], return_tensors="pt")
            inputs = inputs.to(model.device)
            outputs = model.generate(**inputs,
                                     max_length=inputs['input_ids'].shape[-1] + out_seq_length,
                                     do_sample=True,
                                     top_p=top_p,
                                     top_k=top_k,
                                     temperature=temperature,
                                     pad_token_id=2,
                                     eos_token_id=2)
            response = tokenizer.decode(outputs[0])
        
        return response
    
    with gr.Blocks(title="CodeGeeX2 DEMO") as demo:
        gr.Markdown(
            """
            <p align="center">
                <img src="https://raw.githubusercontent.com/THUDM/CodeGeeX2/main/resources/codegeex_logo.png">
            </p>
            """)
        gr.Markdown(
            """
            <p align="center">
                🏠 <a href="https://codegeex.cn" target="_blank">Homepage</a>｜💻 <a href="https://github.com/THUDM/CodeGeeX2" target="_blank">GitHub</a>｜🛠 Tools <a href="https://marketplace.visualstudio.com/items?itemName=aminer.codegeex" target="_blank">VS Code</a>, <a href="https://plugins.jetbrains.com/plugin/20587-codegeex" target="_blank">Jetbrains</a>｜🤗 <a href="https://huggingface.co/THUDM/codegeex2-6b" target="_blank">Download</a>｜📄 <a href="https://arxiv.org/abs/2303.17568" target="_blank">Paper</a>
            </p>
            """)
        gr.Markdown(
            """
            这是 CodeGeeX2 的简易DEMO。请注意：
            * CodeGeeX2 是一个基座模型，它可以完成代码补全/翻译/解释等任务，没有针对聊天进行指令微调。可以在 CodeGeeX 插件[VS Code](https://marketplace.visualstudio.com/items?itemName=aminer.codegeex)、[Jetbrains](https://plugins.jetbrains.com/plugin/20587-codegeex)中体验指令微调后的版本。
            * 可以通过添加`language tag`来控制编程语言，例如`# language: Python`，查看[完整支持语言列表](https://github.com/THUDM/CodeGeeX2/blob/main/evaluation/utils.py#L14)。
            * 按照所选编程语言的格式写注释可以获得更好的结果，请参照下方给出的示例。

            This is the DEMO for CodeGeeX2. Please note that:
            * CodeGeeX2 is a base model, which is not instruction-tuned for chatting. It can do tasks like code completion/translation/explaination. To try the instruction-tuned version in CodeGeeX plugins ([VS Code](https://marketplace.visualstudio.com/items?itemName=aminer.codegeex), [Jetbrains](https://plugins.jetbrains.com/plugin/20587-codegeex)).
            * Programming languages can be controled by adding `language tag`, e.g., `# language: Python`. The format should be respected to ensure performance, full list can be found [here](https://github.com/THUDM/CodeGeeX2/blob/main/evaluation/utils.py#L14).
            * Write comments under the format of the selected programming language to achieve better results, see examples below.
            """)

        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(lines=14, placeholder='Please enter the description or select an example input below.',label='Input')
                with gr.Row():
                    gen = gr.Button("Generate")
                    clr = gr.Button("Clear")

            outputs = gr.Textbox(lines=15, label='Output')

        gr.Markdown(
            """
            Generation Parameter
            """)
        
        with gr.Row():
            with gr.Row():
                seed = gr.Slider(maximum=10000, value=8888, step=1, label='Seed')
                with gr.Row():
                    out_seq_length = gr.Slider(maximum=8192, value=128, minimum=1, step=1, label='Output Sequence Length')
                    temperature = gr.Slider(maximum=1, value=0.2, minimum=0, label='Temperature')
                with gr.Row():
                    top_k = gr.Slider(maximum=100, value=0, minimum=0, step=1, label='Top K')
                    top_p = gr.Slider(maximum=1, value=0.95, minimum=0, label='Top P')
        with gr.Row():
            lang = gr.Radio(
                choices=["None"] + list(LANGUAGE_TAG.keys()), value='None', label='Programming Language')
        inputs = [prompt, lang, seed, out_seq_length, temperature, top_k, top_p]
        gen.click(fn=predict, inputs=inputs, outputs=outputs)
        clr.click(fn=lambda value: gr.update(value=""), inputs=clr, outputs=prompt)

        gr_examples = gr.Examples(examples=examples, inputs=[prompt, lang],
                                  label="Example Inputs (Click to insert an examplet it into the input box)",
                                  examples_per_page=20)
    if not args.auth:
        demo.launch(server_name=args.listen, server_port=args.port)
    else:
        demo.launch(server_name=args.listen, server_port=args.port, auth=(args.username, args.password))
    
    #如果需要监听0.0.0.0和其他端口 可以改成 demo.launch(server_name="0.0.0.0", server_port=6666)
    #如果需要加密码 demo.launch(server_name="0.0.0.0", server_port=6666, auth=("admin", "password"))

if __name__ == '__main__':
    with torch.no_grad():
        main()

