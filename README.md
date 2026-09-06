# nanochat-guide

[karpathy/nanochat](https://github.com/karpathy/nanochat) 用一份可改的小仓库，把分词、预训练、SFT、可选强化学习、评测和带 KV 的推理串成一条能跑通的链路。它不是训练框架，也不是生产推理引擎；它是目前少有的、认知负担足够低的全栈基线。

本仓库对照这份代码精读：每段在全链路的哪一步、为什么这样写。不提供代码副本。

建议有 PyTorch 基础。不要求NLP背景。若对 PyTorch 训练循环还不熟，可先看 [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)。PyTorch 仍吃力的话，可选看一份最小的 [MNIST 训练示例](https://github.com/smiletrl/machine-learning/blob/main/01_nn_from_scratch_mnist/code/pytorch_mnist.py)（不是本仓库内容）。

## 这不是什么

- 不是 nanochat 的 fork，也不维护一份代码副本
- 不是从 NLP 讲起的系统课
- 不是 vLLM / SGLang 生产系统教程

## 怎么读

1. Clone 上游，不必先跑完 `runs/speedrun.sh`：

   ```bash
   git clone https://github.com/karpathy/nanochat.git
   ```

2. 按下面顺序读。已完成的章节链到正文，未写的只标对应文件。
3. 读设计即可，不要求 8×H100。

## 目录

状态：✅ 已有正文 · 🚧 草稿 · ⏳ 未写

| # | 章节 | 状态 | 上游主要文件 |
|:---|:---|:---:|:---|
| **第 0 阶** | **全局视角** | | |
| 0 | 仓库地图：一次训练到能聊天经过哪些阶段 | ⏳ | `runs/speedrun.sh`，`scripts/` |
| **第 1 阶** | **数据与模型骨架 (Model)** | | |
| 1 | Tokenizer | ⏳ | `scripts/tok_train.py`，`scripts/tok_eval.py`，`nanochat/tokenizer.py` |
| 2 | 模型结构 | ⏳ | `nanochat/gpt.py` |
| 2.1 | [Grouped-Query Attention](GroupQueryAttention.md) | ✅ | `nanochat/gpt.py` ， `scripts/base_train.py` |
| 3 | [旋转位置编码 RoPE](RotaryEmbedding.md) | ✅ | `nanochat/gpt.py` |
| **第 2 阶** | **训练流水线 (Training)** | | |
| 4 | [预训练参数与训练步数计算](BaseTrainParameter.md) | 🚧 | `scripts/base_train.py` |
| 5 | 优化器、常规精度与 checkpoint | ⏳ | `nanochat/optim.py`，`nanochat/common.py`，`nanochat/checkpoint_manager.py` |
| 6 | [FP8 混合精度训练](FP8.md) | ✅ | `nanochat/fp8.py` |
| **第 3 阶** | **对齐与评测 (Post-Training)**| | |
| 7 | SFT / 对话对齐 | ⏳ | `scripts/chat_sft.py`，`tasks/` |
| 8 | 强化学习 | ⏳ | `scripts/chat_rl.py` |
| 9 | 评测 | ⏳ | `scripts/base_eval.py`，`scripts/chat_eval.py`，`nanochat/core_eval.py`，`tasks/` |
| **第 4 阶** | **推理引擎与极致效率 (Inference & Systems)**| | |
| 10| [KV Cache 与推理引擎](KVCache.md) | ✅ | `nanochat/engine.py`，`nanochat/gpt.py` |
| 11| [FlashAttention](FlashAttention.md) | 🚧 | `nanochat/flash_attention.py` |
| 12| 对话入口 | ⏳ | `scripts/chat_cli.py` |

## 进度

进行中。已完成 RoPE、KV Cache、FP8；FlashAttention 为草稿。

每章固定四块：问题是什么、最小直觉、nanochat 里对应哪几行、工业系统通常还会多什么。最后一块目前可能很短。

## 上游

- 代码：[karpathy/nanochat](https://github.com/karpathy/nanochat)
- 原作者说明：[Discussions](https://github.com/karpathy/nanochat/discussions)

本仓库与 Andrej Karpathy / nanochat 没有官方关系。