![](resources/codegeex_logo.png)

<p align="center">
    🏠 <a href="https://codegeex.cn" target="_blank">Homepage</a>｜🛠 Extensions <a href="https://marketplace.visualstudio.com/items?itemName=aminer.codegeex" target="_blank">VS Code</a>, <a href="https://plugins.jetbrains.com/plugin/20587-codegeex" target="_blank">Jetbrains</a>｜🤗 <a href="https://huggingface.co/THUDM/codegeex2-6b" target="_blank">HF Repo</a>｜📄 <a href="https://arxiv.org/abs/2303.17568" target="_blank">Paper</a>
</p>

<p align="center">
    👋 Join our <a href="https://discord.gg/8gjHdkmAN6" target="_blank">Discord</a>, <a href="https://join.slack.com/t/codegeexworkspace/shared_invite/zt-1s118ffrp-mpKKhQD0tKBmzNZVCyEZLw" target="_blank">Slack</a>, <a href="https://t.me/+IipIayJ32B1jOTg1" target="_blank">Telegram</a>, <a href="resources/wechat.md"target="_blank">WeChat</a>
</p>

查看[中文版](README.md)<br>
[日本語](README_JA.md)で読む<br>
Lire en [Français](README_FR.md)

# CodeGeeX2: A More Powerful Multilingual Code Generation Model

CodeGeeX2 is the second-generation model of the multilingual code generation model [CodeGeeX](https://github.com/THUDM/CodeGeeX) ([KDD’23](https://arxiv.org/abs/2303.17568)), which is implemented based on the [ChatGLM2](https://github.com/THUDM/ChatGLM2-6B) architecture trained on more code data. Due to the advantage of ChatGLM2, CodeGeeX2 has been comprehensively improved in coding capability (+107% > CodeGeeX; with only 6B parameters, surpassing larger StarCoder-15B for some tasks). It has the following features:

* **More Powerful Coding Capabilities**: Based on the ChatGLM2-6B model, CodeGeeX2-6B has been further pre-trained on 600B code tokens, which has been comprehensively improved in coding capability compared to the first-generation. On the [HumanEval-X](https://huggingface.co/datasets/THUDM/humaneval-x) benchmark, all six languages have been significantly improved (Python +57%, C++ +71%, Java +54%, JavaScript +83%, Go +56%, Rust +321\%), and in Python it reached 35.9% of Pass@1 one-time pass rate, surpassing the larger StarCoder-15B.
* **More Useful Features**: Inheriting the ChatGLM2-6B model features, CodeGeeX2-6B better supports both Chinese and English prompts, maximum 8192 sequence length, and the inference speed is significantly improved compared to the first-generation. After quantization, it only needs 6GB of GPU memory for inference, thus supports lightweight local deployment.
* **Comprehensive AI Coding Assistant**: The backend of CodeGeeX plugin ([VS Code](https://marketplace.visualstudio.com/items?itemName=aminer.codegeex), [Jetbrains](https://plugins.jetbrains.com/plugin/20587-codegeex)) is upgraded, supporting 100+ programming languages, and adding practical functions such as infilling and cross-file completion. Combined with the "Ask CodeGeeX" interactive AI coding assistant, it can be used to solve various programming problems via Chinese or English dialogue, including but not limited to code summarization, code translation, debugging, and comment generation, which helps increasing the efficiency of developpers.
* **Open Liscense**: CodeGeeX2-6B weights are fully open to academic research, and please apply for commercial use by filling in the [registrition form](https://open.bigmodel.cn/mla/form?mcode=CodeGeeX2-6B).


## AI Coding Assistant

![](resources/codegeex_demo.png)

We have developed the CodeGeeX plugin, which supports IDEs such as VS Code, IntelliJ IDEA, PyCharm, GoLand, WebStorm, and Android Studio. The plugin allows you to experience the CodeGeeX2 model's capabilities in code generation and completion, annotation, code translation, and "Ask CodeGeeX" interactive programming, which can help improve your development efficiency. Please download the CodeGeeX plugin in your IDE to get a more comprehensive AI coding experience. You can find more details on our [homepage]( https://codegeex.cn/).

## Get Started

Use `transformers` to quickly launch [CodeGeeX2-6B](https://huggingface.co/THUDM/codegeex2-6b)：

```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("THUDM/codegeex2-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/codegeex2-6b", trust_remote_code=True, device='cuda')
model = model.eval()

# remember adding a language tag for better performance
prompt = "# language: Python\n# write a bubble sort function\n"
inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(inputs, max_length=256, top_k=1)
response = tokenizer.decode(outputs[0])

>>> print(response)
# language: Python
# write a bubble sort function


def bubble_sort(list):
    for i in range(len(list) - 1):
        for j in range(len(list) - 1):
            if list[j] > list[j + 1]:
                list[j], list[j + 1] = list[j + 1], list[j]
    return list


print(bubble_sort([5, 2, 1, 8, 4]))
```

Launch Gradio DEMO:
```
python ./demo/run_demo.py
```

❗️Attention:
* CodeGeeX2 is a base model, which is not instruction-tuned for chatting. It can do tasks like code completion/translation/explaination. To try the instruction-tuned version in CodeGeeX plugins ([VS Code](https://marketplace.visualstudio.com/items?itemName=aminer.codegeex), [Jetbrains](https://plugins.jetbrains.com/plugin/20587-codegeex)).
* Programming languages can be controled by adding `language tag`, e.g., `# language: Python`. The format should be respected to ensure performance, full list can be found [here](https://github.com/THUDM/CodeGeeX2/blob/main/evaluation/utils.py#L14). Please write comments under the format of the selected programming language to achieve better results.
* If the GPU doesn't support `bfloat16` format, it will cause incorrect output. Please convert the model to `float16` format:
    ```python
    model = AutoModel.from_pretrained("THUDM/codegeex2-6b", trust_remote_code=True).half().cuda()
    ```
* If you need to use Multiple GPUs to load the model, you can use the following code:
    ```python
    tokenizer = AutoTokenizer.from_pretrained("THUDM/codegeex2-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/codegeex2-6b", trust_remote_code=True, device='cuda')
    model = model.eval()
    ```
    Replace with

    ```python
    def get_model():
        tokenizer = AutoTokenizer.from_pretrained("THUDM/codegeex2-6b", trust_remote_code=True)
        from gpus import load_model_on_gpus
        # The "gpus" file is located in the demo folder
        model = load_model_on_gpus("THUDM/codegeex2-6b", num_gpus=2)
        model = model.eval()
        return tokenizer, model

    tokenizer, model = get_model()
    ```

## Evaluation

CodeGeeX2 is a base model for multilingual code generation, which has been significantly improved in its coding ability compared to the previous generation. The following are the evaluation results on the HumanEval, HumanEval-X, and DS1000 benchmarks (the evaluation metric Pass@k is the same as in the [paper](https://arxiv.org/abs/2303.17568)):

### HumanEval (Pass@1,10,100)

| **Model**           | **Pass@1** | **Pass@10** | **Pass@100** |
| :-----------------: | :--------: | :---------: | :----------: |
| CodeGen-16B-multi   | 19\.2      | 34\.6       | 55\.2        |
| CodeGeeX-13B        | 22\.9      | 39\.6       | 60\.9        |
| Codex-12B           | 28\.8      | 46\.8       | 72\.3        |
| CodeT5Plus-16B-mono | 30\.9      | 51\.6       | 76\.7        |
| Code-Cushman-001    | 33\.5      | 54\.3       | 77\.4        |
| LLaMA-65B           | 23\.7      | -           | 79\.3        |
| LLaMA2-70B          | 29\.9      | -           | -            |
| CodeGen2\.5-7B-mono | 33\.4      | 58\.4       | 82\.7        |
| StarCoder-15B       | 33\.2      | 61\.0       | 84\.7        |
| **CodeGeeX2-6B**    | **35\.9**  | **62\.6**   | **88\.3**    |
> `n=20, t=0.2, top_p=0.95` for **Pass@1**; `n=200, t=0.8, top_p=0.95` for **Pass@10** and **Pass@100**.

### HumanEval-X (Pass@1)

| **Model**                | **Python** | **C++**   | **Java**  | **JavaScript** | **Go**    | **Rust**  | **Overall** |
| :------------------: | :--------: | :-------: | :-------: | :------------: | :-------: | :-------: | :---------: |
| CodeGen-16B-multi    | 19\.2      | 18\.1     | 15\.0     | 18\.4          | 13\.0     | 1\.8      | 14\.2       |
| CodeGeeX-13B         | 22\.9      | 17\.1     | 20\.0     | 17\.6          | 14\.4     | 4\.3      | 16\.0       |
| Replit-code-v1-3B    | 22\.0      | 20\.1     | 20\.1     | 20\.1          | 12\.2     | 8\.6      | 17\.2       |
| CodeGen2\.5-7B-multi | 30\.6      | 24\.3     | 29\.0     | 27\.5          | 18\.9     | **20\.1** | 25\.1       |
| StarCoder-15B        | 35\.5      | 28\.2     | **31\.5** | **33\.2**      | 21\.3     | 17\.8     | 27\.9       |
| **CodeGeeX2-6B**         | **35\.9**  | **29\.3** | 30\.8     | 32\.2          | **22\.5** | 18\.1     | **28\.1**   |
> `n=20, t=0.2, top_p=0.95` for **Pass@1**.

The above results can be reproduced by running `scripts/run_humanevalx.sh`. Refer to [HumanEval-X environment](https://github.com/THUDM/CodeGeeX/blob/main/codegeex/benchmark/README_zh.md) for the experiment setups.

### DS1000 (Pass@1)

| **Model**            | **Matplotlib** | **Numpy** | **Pandas** | **Pytorch** | **SciPy** | **Scikit-learn** | **TensorFlow** | **Overall** |
| :--------------: | :------------: | :-------: | :--------: | :---------: | :-------: | :--------------: | :------------: | :---------: |
| \# Samples       | 155            | 220       | 291        | 68          | 106       | 115              | 45             | 1000        |
| CodeGen-16B-Mono | 31\.7          | 10\.9     | 3\.4       | 7\.0        | 9\.0      | 10\.8            | 15\.2          | 11\.7       |
| code-cushman-001 | 40\.7          | 21\.8     | 7\.9       | 12\.4       | 11\.3     | 18\.0            | 12\.2          | 18\.1       |
| Codex-001        | 41\.8          | 26\.6     | 9\.4       | 9\.7        | 15\.0     | 18\.5            | 17\.2          | 20\.2       |
| **CodeGeeX2-6B** | 40\.5          | 25\.5     | 14\.5      | 17\.3       | 19\.3     | 24\.0            | 23\.0          | 23\.1       |
| StarCoder-15B    | 51\.7          | 29\.7     | 11\.4      | 21\.4       | 20\.2     | 29\.5            | 24\.5          | 26\.0       |
| Codex-002        | **57\.0**      | **43\.1** | **26\.5**  | **41\.8**   | **31\.8** | **44\.8**        | **39\.3**      | **39\.2**   |
> `n=40, t=0.2, top_p=0.5` for **Pass@1**。

The above results can be reproduced by the code in [DS1000 repo](https://github.com/HKUNLP/DS-1000.git).

## Inference

CodeGeeX2 is more friendly to deployment than the previous generation. Thanks to the use of Multi-Query Attention and Flash Attention, the inference speed is faster, and only 6GB of GPU memory is required after INT4 quantization.

### Quantization

| **Model**        | FP16/BF16 | INT8    | INT4   |
| :--------------: | :-------: | :-----: | :----: |
| CodeGeeX-13B     | 26\.9 GB   | 14\.7 GB | -      |
| **CodeGeeX2-6B** | 13\.1 GB  | 8\.2 GB  | 5\.5 GB |
> Based on PyTorch 2.0, using `torch.nn.functional.scaled_dot_product_attention` for effecient attention mechanism。

### Acceleration

| **Model**        | **Inference speed (token/s)** |
| :--------------: | :-------------: |
| CodeGeeX-13B     | 32              |
| **CodeGeeX2-6B** | 94              |
> `batch_size=1, max_length=2048`, both using acceleration framework, in `GeForce RTX-3090`。

## License

The code in this repository is open source under the [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) license. The model weights are licensed under the [Model License](MODEL_LICENSE). CodeGeeX2-6B weights are open for academic research, and please apply for commercial use by filling in the [registration form](https://open.bigmodel.cn/mla/form?mcode=CodeGeeX2-6B).


## Citation

If you find our work helpful, please feel free to cite the following paper:

```
@inproceedings{zheng2023codegeex,
      title={CodeGeeX: A Pre-Trained Model for Code Generation with Multilingual Evaluations on HumanEval-X},
      author={Qinkai Zheng and Xiao Xia and Xu Zou and Yuxiao Dong and Shan Wang and Yufei Xue and Zihan Wang and Lei Shen and Andi Wang and Yang Li and Teng Su and Zhilin Yang and Jie Tang},
      booktitle={KDD},
      year={2023}
}
```
