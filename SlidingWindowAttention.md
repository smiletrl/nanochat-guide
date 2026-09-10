# 滑动窗口注意力

滑动窗口注意力（Sliding Window Attention）用在 [Flash Attention V3](FlashAttentionV3.md) 中，将注意力计算的复杂度从 $O(N^2)$ 降到 $O(N*W)$ 。极大降低显存占用，提高大模型训练跟推理速度。

在如下具体百万token上下文的案例中，KV 缓存的 **显存占用下降 99.6%**。

### 具体案例数据对比

全局全注意力（Full Attention），单头计算量的核心操作是 $Q \times K^T$，需要计算的点积次数与上下文长度的平方成正比。以序列长度 $N = 1,000,000$（$10^6$），窗口大小 $W = 4,096$ 为例：

| 指标 | 全局自注意力（Full Attention） | 滑动窗口注意力（Sliding Window Attention） | 优化效果 |
| --- | --- | --- | --- |
| **理论复杂度** | $O(N^2)$ | $O(N \times W)$ | 随长度线性增长 |
| **点积计算次数**（单头近似） | $N^2 = (10^6)^2 = \mathbf{10^{12}}$ | $N \times W = 10^6 \times (4 \times 10^3) \approx \mathbf{4 \times 10^9}$ | **计算量降为原来的 $\approx 1/250$** |
| **Decode 阶段 KV Cache 显存** | 需维护全量 $N$ 个 token 的 KV | 仅需循环缓存最近 $W$ 个 token 的 KV | **单请求 KV 显存占用直接降低 99.6%** |

---

### 案例总结

在处理 **1M（$10^6$）Tokens** 的超长上下文场景中：
* **全量注意力（Full Attention）：** 点积矩阵规模为 $N^2 = (10^6)^2 = \mathbf{10^{12}}$，无论在预填充（Prefill）阶段的 FLOPs 还是解码（Decode）阶段的 KV Cache 显存占用，都会成为瓶颈。
* **滑动窗口注意力（Sliding Window Attention）：** 若设置窗口大小 $W = 4096 \approx 4 \times 10^3$（如 Mistral 或 Qwen 早期长文本所用配置），每个 Token 仅与局部前 $W$ 个 Token 交互，计算规模降为 $N \times W \approx 10^6 \times 4096 \approx \mathbf{4 \times 10^9}$，计算量相比全注意力直接下降了 **两个数量级以上（约 250 倍）**。
* **推理优势：** 解码阶段不再需要线性增长的全局 [KV Cache](KVCache.md)，可采用 **循环缓冲区（Ring Buffer）** 只保留最近 $W$ 个 Token，彻底消除了超长对话中显存溢出 OOM 的问题。 

*(注：在主流开源模型中，如 Mistral-7B 首次大范围推广了 $W=4096$ 的 SWA；而在 Qwen / DeepSeek 的超长上下文架构中，通常会采用**混合策略**——底层使用滑动窗口截断低频长距离依赖，顶层或部分特定层保留全注意力，以在保持 $O(N \cdot W)$ 整体吞吐优势的同时不损失全局检索能力。)*

## 实现步骤

1. 启动命令

参数 `--window-pattern` 可选，默认值为 `SSSL`。`S` 对应了局部的上下文，`L` 对应全部的上下文，也就是保留全部注意力。

这个配置模式，对应了大模型中的模块层。比如默认的 `SSSL`, 表示模型从1-3层采用 S 模式，第4层采用 L 模式。第5-7层继续采用 S 模式，第8层采用 L 模式。依次顺延。

实践中，大模型最后一层固定采用 L 模式，无论这里的参数配置如何。

```bash
python -m scripts.base_train \
  --max-seq-len=512 \
  --window-pattern=SSL
```

注意，滑动窗口注意力（SWA）应该跟 Flash Attention v3 一起使用。如果环境不支持 FA3，那么SWA会回退到默认的缩放点积注意力（Scaled Dot-Product Attention - SDPA） 注意力计算模式。这种模式实际不支持 SWA。内存跟计算没有提升。

2. 预先计算每层的窗口长度

L 注意力则是全部的sequence 长度。S 局部注意力，计算把全部上下文的 1/4 作为短窗口大小，然后向上取整到 128 的整数倍。然后每层依次按照窗口模式，配置该层选 S 还是选 L。

`nanochat/gpt.py` 中的实现见

```python
class GPT(nn.Module):
  def __init__(self, config, pad_vocab_size_to=64):
    # 提前计算每层的滑动窗口
    self.window_sizes = self._compute_window_sizes(config)

  def _compute_window_sizes(self, config):
```

这里所谓`每层`在代码中指的是每个 `class Block` 模块层。而滑动窗口仅对 Block 层内部的 `class CausalSelfAttention` 发挥作用。对 `class MLP(nn.Module)` 无效。

```python
class GPT(nn.Module):
  def forward():
    # 依次读取每个 Block 层，分配对应层数的窗口值。
    for i, block in enumerate(self.transformer.h):
        x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)

class Block(nn.Module):
  def forward(self, x, ve, cos_sin, window_size, kv_cache):
    # 传给 `CausalSelfAttention`。
    x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
```

3. 稀疏矩阵掩码的滑动窗口实现

在 `class CausalSelfAttention` 内部，经 `forward()` 函数，最终进入 `nanochat/flash_attention.py` 中。

FA3的实现在接口内部，不在nanochat的实现范围内。调用方法类似 `_fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)`。

SDPA 的实现方式，虽然工程实践中不会用于工业生产环境，但是可以帮助我们理解滑动窗口的具体用途。调用详情见 `def _sdpa_attention()`