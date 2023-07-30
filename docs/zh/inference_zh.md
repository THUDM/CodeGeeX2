# CodeGeeX2推理教程

CodeGeeX2 是多语言代码生成模型 [CodeGeeX](https://github.com/THUDM/CodeGeeX) ([KDD’23](https://arxiv.org/abs/2303.17568)) 的第二代模型，更强，更快，更轻量，是适合本地部署的AI代码生成助手。CodeGeeX2 支持在多种不同平台上进行推理，本教程将会介绍几种不同的推理方式，包括CPU推理，多卡推理，加速推理等。

- [快速开始](#快速开始)
- [多精度/量化推理](#多精度/量化推理)
- [多GPU推理](#多GPU推理)
- [Mac推理](#Mac推理)
- [fastllm加速推理](#fastllm加速推理)

## 快速开始

下载本仓库并使用`pip`安装环境依赖：

```shell
git clone https://github.com/THUDM/CodeGeeX2
cd CodeGeeX2
pip install -r requirements.txt
```

使用`transformers`快速调用[CodeGeeX2-6B](https://huggingface.co/THUDM/codegeex2-6b)，将自动下载权重到本地：

```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("THUDM/codegeex2-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/codegeex2-6b", trust_remote_code=True, device='cuda')  # 如使用CPU推理，device='cpu'
model = model.eval()

# CodeGeeX2支持100种编程语言，加入语言标签引导生成相应的语言
prompt = "# language: Python\n# write a bubble sort function\n"
inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(inputs, max_length=256, top_k=1)  # 示例中使用greedy decoding，检查输出结果是否对齐
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

亦可以手动下载权重：

```shell
# huggingface下载
git clone https://huggingface.co/THUDM/codegeex2-6b
```

将tokenizer和model路径改为本地路径：

```python
model_path = "/path/to/codegeex2-6b"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
```

## 多精度/量化推理

CodeGeeX2 使用BF16训练，推理时支持BF16/FP16/INT8/INT4，可以根据显卡显存选择合适的精度格式：

|    **Model**     | FP16/BF16 |   INT8   |  INT4   |
| :--------------: | :-------: | :------: | :-----: |
|   CodeGeeX-13B   | 26\.9 GB  | 14\.7 GB |    -    |
| **CodeGeeX2-6B** | 13\.1 GB  | 8\.2 GB  | 5\.5 GB |

默认使用BF16精度进行推理，如显卡不支持BF16（❗️如使用错误的格式，推理结果将出现乱码），需要转换为FP16格式：

```python
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().to("cuda")
```

量化推理以INT4为例，可以下载转换好的权重（[INT4权重](https://huggingface.co/THUDM/codegeex2-6b-int4)）或手动转换，如果显卡不支持BF16，也需要先转换为FP16格式：

```python
# 下载转换好的权重
model = AutoModel.from_pretrained("THUDM/codegeex2-6b-int4", trust_remote_code=True)

# 手动转换权重
model = AutoModel.from_pretrained("THUDM/codegeex2-6b", trust_remote_code=True).quantize(4).to("cuda")

# 如果显卡不支持BF16，需要先转换为FP16格式
model = AutoModel.from_pretrained("THUDM/codegeex2-6b", trust_remote_code=True).half().quantize(4).to("cuda")
```

##  多GPU推理

用[gpus.py](https://github.com/THUDM/CodeGeeX2/blob/main/demo/gpus.py)实现多GPU推理：

```python
from gpus import load_model_on_gpus
model = load_model_on_gpus("THUDM/codegeex2-6b", num_gpus=2)
```

## Mac推理

对于搭载了 Apple Silicon 或者 AMD GPU 的 Mac，可以使用 MPS 后端运行。参考 Apple 的 [官方说明](https://developer.apple.com/metal/pytorch) 安装 PyTorch-Nightly（正确的版本号应该是2.x.x.dev2023xxxx，如2.1.0.dev20230729）：

```shell
pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

在 MacOS 上只支持从本地加载模型（提前下载权重[codegeex2-6b](https://huggingface.co/THUDM/codegeex2-6b)，[codegeex2-6b-int4](https://huggingface.co/THUDM/codegeex2-6b-int4)），支持FP16/INT8/INT4格式，并使用 mps 后端：

```python
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().to('mps')
```

## fastllm加速推理

可以使用[fastllm](https://github.com/ztxz16/fastllm)对 CodeGeeX2 进行加速，fastllm是目前支持GLM架构的最快开源框架。首先安装fastllm_pytools：

```shell
git clone https://github.com/ztxz16/fastllm
cd fastllm
mkdir build
cd build
# 使用GPU编译，需要添加CUDA路径：export CUDA_HOME=/usr/local/cuda/bin:$PATH，export PATH=$PATH:$CUDA_HOME/bin
cmake .. -DUSE_CUDA=ON # 如果不使用GPU编译 cmake .. -DUSE_CUDA=OFF
make -j
cd tools && python setup.py install  # 确认安装是否成功，在python中 import fastllm_pytools 不报错
```

如出现架构不支持的报错，需要调整`CMakeLists.txt`，注释掉下面一行：

```shell
# set(CMAKE_CUDA_ARCHITECTURES "native")
```

将huggingface转换成fastllm格式：

```python
# 原本的调用代码
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("THUDM/codegeex2-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/codegeex2-6b", trust_remote_code=True)

# 加入下面这两行，将huggingface模型转换成fastllm模型
from fastllm_pytools import llm
model = llm.from_hf(model, tokenizer, dtype="float16") # dtype支持 "float16", "int8", "int4"
```

fastllm中模型接口和huggingface不完全相同，可以参考[demo/run_demo.py](https://github.com/THUDM/CodeGeeX2/blob/main/demo/run_demo.py)中的相关实现：

```python
model.direct_query = True
outputs = model.chat(tokenizer, 
                     prompt,
                     max_length=out_seq_length,
                     top_p=top_p,
                     top_k=top_k,
                     temperature=temperature)
response = outputs[0]
```
