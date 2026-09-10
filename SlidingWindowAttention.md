# 滑动窗口注意力

滑动窗口注意力（Sliding Window Attention）是一种 注意力模式：每个 token 只和自己附近一段历史做注意力，而不是和整段上下文。复杂度从 $O(N^2)$ 降到约 $O(N⋅W)$ 。

在下面的具体百万token上下文的案例中，KV 缓存理论上 **显存占用下降 99.6%**。

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
* **推理优势：** 解码阶段不再需要线性增长的全局 [KV Cache](KVCache.md)，可采用 **循环缓冲区（Ring Buffer）** 只保留最近 $W$ 个 Token，彻底消除了超长对话中显存溢出 OOM 的问题。注意，循环缓冲区不是nanochat的实现方式。 

**注：** 在主流开源模型中，如 Mistral-7B 首次大范围推广了 $W=4096$ 的 SWA；而在 Qwen / DeepSeek 的超长上下文架构中，通常会采用**混合策略**——底层使用滑动窗口截断低频长距离依赖，顶层或部分特定层保留全注意力，以在保持 $O(N \cdot W)$ 整体吞吐优势的同时不损失全局检索能力。

## 实现步骤

1. 启动命令

参数 `--window-pattern` 可选，默认值为 `SSSL`。`S` 对应了局部的上下文，`L` 对应全部的上下文，也就是保留全部注意力。

这个配置模式，对应了大模型中的模块层。比如默认的 `SSSL`, 表示模型从1-3层采用 S 模式，第4层采用 L 模式。第5-7层继续采用 S 模式，第8层采用 L 模式。依次顺延。

实践中，大模型最后一层固定采用 L 模式，无论这里的参数配置如何。

混合 S/L 的意义：多数层局部、隔几层（以及最后一层）全注意力，用来补长距离，和 Gemma/Mistral-style hybrid 同一思路。nanochat 并不是每层都 SWA。

```bash
python -m scripts.base_train \
  --max-seq-len=512 \
  --window-pattern=SSL
```

注意，滑动窗口注意力（SWA）应该跟 Flash Attention v3 一起使用。如果环境不支持 FA3，那么SWA会回退到默认的缩放点积注意力（Scaled Dot-Product Attention - SDPA） 注意力计算模式。

2. 预先计算每层的窗口长度

L 注意力则是全部的sequence 长度。S 局部注意力，计算把全部上下文的 1/4 作为短窗口大小，然后向上取整到 128 的整数倍。比如`max-seq-len=512`, 那么 S 为 128。接下来每层依次按照窗口模式，配置该层选 S 还是选 L。

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

SDPA 的实现方式虽然在生产环境中效率不如原生算子，但非常直观地展示了滑动窗口的数学本质。

在因果注意力（Causal Attention）中，由于只能看过去和当前的 token（即列索引 $\le$ 行索引），允许计算的有效区域实际是**下三角矩阵**（右上角的未来 token 被遮蔽为 0）。

以序列长度 $T=6$ 为例，其中行表示当前 Query 的 Token 绝对位置，列表示目标 Key 的 Token 绝对位置。

```text
       K0  K1  K2  K3  K4  K5
Q0 [   1   0   0   0   0   0  ]
Q1 [   1   1   0   0   0   0  ]
Q2 [   1   1   1   0   0   0  ]
Q3 [   1   1   1   1   0   0  ]
Q4 [   1   1   1   1   1   0  ]
Q5 [   1   1   1   1   1   1  ]
```

加了滑动窗口（如设置窗口大小 `window = 2`）后，每个 Token 最多只能看自己以及前 2 个 Token，超出窗口太久远的历史信息（矩阵左下角）也被遮蔽为 `0`，掩码矩阵收缩为一条**沿主对角线的斜带状矩阵（Band Matrix）**：

```text
       K0  K1  K2  K3  K4  K5
Q0 [   1   0   0   0   0   0  ]
Q1 [   1   1   0   0   0   0  ]
Q2 [   1   1   1   0   0   0  ]
Q3 [   0   1   1   1   0   0  ]   <- Q3 丢失了对 K0 的关注
Q4 [   0   0   1   1   1   0  ]   <- Q4 只能看到 [K2, K3, K4]
Q5 [   0   0   0   1   1   1  ]   <- Q5 只能看到 [K3, K4, K5]
```

对应实现见 `nanochat/flash_attention.py` 中的 `def _sdpa_attention()`：

```python
def _sdpa_attention(q, k, v, window_size, enable_gqa):
  # 1. 构造标准下三角因果掩码 (保留自己及历史 Token，右上角未来置 False)
  row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
  col_idx = torch.arange(Tk, device=device).unsqueeze(0)
  mask = col_idx <= row_idx

  # 2. 考虑滑动窗口截取 (左侧超出窗口大小的历史 Token 置 False)
  if window >= 0 and window < Tk:
      mask = mask & ((row_idx - col_idx) <= window)
  return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)
```

4. 单步解码（Decode）的缓存截取

滑动窗口在推理时主要体现在算子层面的计算量截断与读带宽节省（Compute & Memory Bandwidth Bound）：

```python
def _sdpa_attention(q, k, v, window_size, enable_gqa):
  # Single token generation (单 token 解码阶段)
  if Tq == 1:
      if window >= 0 and window < Tk:
          # 只截取最后 (window + 1) 个 key 和 value，更早的直接切片抛弃
          start = max(0, Tk - (window + 1))
          k = k[:, :, start:, :]
          v = v[:, :, start:, :]
      return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
```

**注意：** 这里的SDPA+滑动窗口，事实上没有加速，反而在底层让GPU性能变差。这里我们的讨论仅仅是理解滑动窗口的实现。
