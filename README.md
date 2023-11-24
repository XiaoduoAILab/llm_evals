<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->

<div align="center">
<h1>
LLM Evals
</h1>
</div>

<div align="center">
[![license](https://img.shields.io/github/license/modelscope/modelscope.svg)](https://github.com/XiaoduoAILab/llm_evals/blob/main/LICENSE)

<h4 align="center">
    <p>
        <b>中文</b> |
        <a href="https://github.com/XiaoduoAILab/llm_evals/blob/main/README_EN.md">English</a>
    <p>
</h4>
</div>

# 介绍

最近，大语言模型（large language models，以下简称LLM）领域取得了很大的进展。许多人声称参数量小于 10B 的小模型可以实现与 GPT-4 相当的性能。这是真的吗？


## 复杂推理评测

关键的区别在于模型是否可以完成复杂的任务。为此我们汇编了一系列复杂推理任务，包括推理（GSM8K）、代码（HumanEval）、心智理论（ToM）、高难度推理（MATH）、符号（BBH）、英文知识（MMLU）、中文知识（C-EVAL）来衡量模型在高难度任务中的表现：

<<<<<<< HEAD
| 模型                      | 规模  | GSM8k | HumanEval | ToM    | MATH | MMLU | BBH     | C-EVAL |
|-------------------------|-----|------| --------- | ------ |------| ---- | ------- | ------ |
| GPT-4                   | 未知  | 92   | 67        | 100/90 | 42.5 | 86.4 | NA      | 68.7   |
| Claude-v1.3             | 未知  | 81.8 | NA        | NA     | NA   | 75.6 | 67.3    | 54.2   |
| PaLM-2-Unicorm          | 未知  | 80.7 | NA        | NA     | 34.3 | 78.3 | 78.1    | NA     |
| ChatGPT (gpt-3.5-turbo) | 未知  | 74.9 | 48.1      | 95/85  | 34.3 | 67.3 | 70.1    | 54.4   |
| LLaMA-2 70B             | 70B | 58.68 | 30.49     | 85/85  | 14.0 | 68.9 | 51.08   | 50.59  |
| Baichuan2 base          | 13B | 51.25 | 23.17     | 85/40  | 8.6  | 58.8 | 36.69   | 57.88  |
| LLaMA-1 65B             | 65B | 49.20 | 20.7      | 95/60  | 11.3 | 63.4 | 46.57   | 41.31  |
| Qwen                    | 7B  | 40.79 | 17        | 15/10  | 7.2  | 22.9 | 18.77   | 58.99  |
| CodeLlama Instruct      | 34B | 40.71 | 38.41     | 55/70  | 9.6  | 53.6 | 49.24   | 44.06  |
| Vicuna-v1.5 13B         | 13B | 34.87 | 20.12     | 60/55  | 6.4  | 54.9 | 41.58   | 40.94  |
| InternLM                | 20B | 34.04 | 26.83     | 70/70  | 8.0  | 59.6 |         | 53.12  |
| Vicuna-v1.3 33B         | 33B | 33.89 | 19.5      | 65/70  | 7.1  | 59.4 | 44.49   | 40.49  |
| LLaMA-1 33B             | 33B | 32.22 | 17.7      | 90/80  | 8.2  | 57.8 | 40.22   | 39.38  |
| CodeLlama Instruct      | 13B | 28.96 | 40.85     | 25/50  | 8.1  | 44.7 | 39.29   | 36.63  |
| ChatGLM2-6B             | 6B  | 26.38 | 0         | 40/15  | 1.1  | 43.2 | 30.66   | 39.52  |
| LLaMA-2 13B             | 13B | 25.63 | 16.46     | 80/55  | 6.5  | 54.8 | 37.44   | 39.67  |
| BELLE-LLaMA-EXT-13B     | 13B | 22.29 | 14.6      | 15/40  | 5.1  | 49.4 | 26.4    | 40.64  |
| Vicuna-v1.5 7B          | 7B  | 20.39 | 17.68     | 70/75  | 4.6  | 49.9 | 36.71   | 37.74  |
| Baichuan2 base          | 7B  | 20.09 | 20.73     | 30/40  | 6.0  | 54   | 32.96   | 55.57  |
| CodeLlama Instruct      | 7B  | 19.86 | 31.1      | 15/25  | 7.0  | 41   | 34.45   | 35.59  |
| Baichuan-13B            | 13B | 19.64 | 14        | 40/30  | 6.2  | 51.3 | 32.85   | 52.6   |
| LLaMA-1 13B             | 13B | 15.09 | 14.6      | 65/50  | 5.9  | 46.7 | 31.25   | 30.24  |
| InternLM                | 7B  | 13.65 | 12.8      | 30/15  | 6.2  | 47.5 | 28.64   | 43.02  |
| LLaMA-2 7B              | 13B | 12.21 | 13.4      | 50/50  | 5.2  | 46.1 | 33.51   | 30.53  |
| LLaMA-1 7B              | 7B  | 10.84 | 11.6      | 45/20  | 4.7  | 34.1 | 27.2    | 27.41  |
| BELLE-LLAMA-7B-0.6M     | 7B  | 7.96 | 15.2      | 30/5   | 4.6  | 40.5 | 28.61   | 28.53  |
| ChatGLM-6B              | 6B  | 6.29 | 0         | 20/20  | 0.4  | 32.9 | 24.45   | 36.85  |
| Baichuan-7B             | 7B  | 5.53 | 7.9       | 20/10  | 4.8  | 42.5 | 26.75   | 42.57  |
| MOSS                    | 16B | 5.23 | 0         | 10/0   | 4.4  | 27.1 | 5.87    | 28.31  |
| BELLE-LLAMA-7B-2M       | 7B  | 2.35 | 11        | 20/10  | 2.7  | 32.6 | 25.65   | 30.39  |
| Tencent Hunyuan                | 未知  | 70.03 | 60.36 | 0/0  | 3.6 | 32.8 | 26.08 | 39.38 |
| xinghuov3-xfyun                   | 未知  | 2.27 | 18.29 | 10/30 | NA   | NA | NA   | NA  |
## 如何使用

要运行评估，需要下载 Hugging Face 兼容的模型，并置于 PATH_TO_CONVERTED_WEIGHTS 目录。然后执行如下脚本：

```
bash scripts/test_reason.sh PATH_TO_CONVERTED_WEIGHTS
```

你会看到七个评测依次进行，并将结果输出。

# 协议

对本仓库源码的使用遵循开源许可协议 [Apache 2.0](https://github.com/XiaoduoAILab/llm_evals/blob/main/LICENSE)。

