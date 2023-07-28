import os
import json
import numpy
import torch
import random
import gradio as gr

from transformers import AutoTokenizer, AutoModel

def get_model():
    tokenizer = AutoTokenizer.from_pretrained("THUDM/codegeex2-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/codegeex2-6b", trust_remote_code=True).to('cuda:0')
    # 如需实现多显卡模型加载,请将上面一行注释并启用一下两行,"num_gpus"调整为自己需求的显卡数量 / To enable Multiple GPUs model loading, please uncomment the line above and enable the following two lines. Adjust "num_gpus" to the desired number of graphics cards.
    # from gpus import load_model_on_gpus
    # model = load_model_on_gpus("THUDM/codegeex2-6b", num_gpus=2)
    model = model.eval()
    return tokenizer, model

tokenizer, model = get_model()

examples = []
with open(os.path.join(os.path.split(os.path.realpath(__file__))[0], "example_inputs.jsonl"), "r", encoding="utf-8") as f:
    for line in f:
        examples.append(list(json.loads(line).values()))

    
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
                🏠 <a href="https://codegeex.cn" target="_blank">Homepage</a>｜💻 <a href="https://github.com/THUDM/CodeGeeX2" target="_blank">GitHub</a>｜🛠 Tools <a href="https://marketplace.visualstudio.com/items?itemName=aminer.codegeex" target="_blank">VS Code</a>, <a href="https://plugins.jetbrains.com/plugin/20587-codegeex" target="_blank">Jetbrains</a>｜🤗 <a href="https://huggingface.co/THUDM/codegeex2-6b" target="_blank">HF Repo</a>｜📄 <a href="https://arxiv.org/abs/2303.17568" target="_blank">Paper</a>
            </p>
            """)
        gr.Markdown(
            """
            This is the DEMO for CodeGeeX2. Please note that:
            * CodeGeeX2 is a base model, which is not instruction-tuned for chatting. It can do tasks like code completion/translation/explaination. To try the instruction-tuned version in CodeGeeX plugins ([VS Code](https://marketplace.visualstudio.com/items?itemName=aminer.codegeex), [Jetbrains](https://plugins.jetbrains.com/plugin/20587-codegeex)).
            * Programming languages can be controled by adding `language tag`, e.g., `# language: Python`. The format should be respected to ensure performance, full list can be found [here](https://github.com/THUDM/CodeGeeX2/blob/main/evaluation/utils.py#L14).
            * Write comments under the format of the selected programming language to achieve better results, see examples below.
            """)

        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(lines=13, placeholder='Please enter the description or select an example input below.',label='Input')
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
        
    demo.launch(share=True)

if __name__ == '__main__':
    with torch.no_grad():
        main()
