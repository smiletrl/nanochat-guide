# GQA Grouped-Query Attention

GQA 的全称 Grouped-Query Attention（分组查询注意力）。

它是介于 MHA（Multi-Head Attention）和 MQA（Multi-Query Attention）之间的一种折中方案：

- MHA（标准多头注意力）：每个 Query head 对应独立的 Key/Value head（例如 10 个 Q 对应 10 个 KV，比例 1:1）。

- MQA（多查询注意力）：所有 Query head 共享唯一 1 个 KV head（例如 10 个 Q 共享 1 个 KV）。KV Cache 最省，但大模型上容量和精度更容易掉。

- GQA（分组查询注意力）：把 Query heads 分组，同一组共享一个 KV head（例如 10 个 Q 分成 5 组，每 2 个 Q 共享 1 个 KV，比例 2:1）。

GQA 在几乎不损失模型表现的前提下，大幅削减 Decode 阶段 KV Cache 的显存和访存带宽，是 Llama 2/3、Mistral 等开源大模型的常见配置。

## 代码微调

nanochat 架构支持 GQA，但默认没有启用：`n_kv_head` 被设成和 `n_head` 一样，等价于 MHA。`scripts/base_train.py` 原始代码：

```python
def build_model_meta(depth):
    num_heads = model_dim // args.head_dim
    config = GPTConfig(
        sequence_len=args.max_seq_len, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=args.window_pattern,
    )
```

这里的 `GPTConfig` 内部的属性 `n_kv_head` 跟 `n_head` 是共用的 `head` 数量。我们可以修改如下，新建一个 `n_kv_heads` 变量，并将这个变量覆盖 `GPTConfig` 默认的 `n_kv_head`。

```python
def build_model_meta(depth):
    num_heads = model_dim // args.head_dim
    n_kv_heads = num_heads // 2           # 这是新增加的变量
    config = GPTConfig(
        sequence_len=args.max_seq_len, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=n_kv_heads, n_embd=model_dim,
        window_pattern=args.window_pattern,
    )
```

`num_heads // 2` 只是演示。真实约束在 `nanochat/gpt.py`:

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
```

即：KV 头数不能多于 Q 头数，且 Q 头数必须是 KV 头数的整数倍。相等时是 MHA，为 1 时是 MQA，介于两者之间才是 GQA。


## 代码执行验证

为了减少干扰，我们可以使用类似命令执行，不传参数 `--depth`（默认值20）：

```bash
python -m scripts.base_train \
  --max-seq-len=512 \
  --device-batch-size=4 \
  --eval-tokens=512 \
  --core-metric-every=-1 \
  --total-batch-size=2048 \
  --num-iterations=1 \
  --eval-every=-1 
```

这样在屏幕上输出如下结果, `"n_head": 10` 跟 `"n_kv_head": 5`:

```bash
Model config:
{
  "sequence_len": 512,
  "vocab_size": 32768,
  "n_layer": 20,
  "n_head": 10,
  "n_kv_head": 5,
  "n_embd": 1280,
  "window_pattern": "SSSL"
}
```

## 多头计算

`nanochat/gpt.py` 的 `class CausalSelfAttention` 里 Q 和 KV 的投影宽度已经不同：

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        # 见 scripts/base_train.py -> `def build_model_meta(depth)`
        #   -> num_heads = model_dim // args.head_dim
        # q 投影后的维度 self.n_head * self.head_dim == self.n_embd，即 10 * 128 = 1280。
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        # k/v 投影后的维度缩小到了 self.n_kv_head * self.head_dim，即 5 * 128 = 640.
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        B, T, C = x.size()
        # q/k/v 向量形状分头. 显然 k/v 的头部数量比 q 要少，是 q 的一半。
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)     # (B, T, 10, 128)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)  # (B, T, 5, 128)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)  # (B, T, 5, 128)
```

最终计算注意力机制时 `nanochat/flash_attention.py` 中的两个分支如下：

```python
# sdpa 分支
def _sdpa_attention(q, k, v, window_size, enable_gqa):
    # enable_gqa = boolean (q 的 head 数 != k 的 head 数)
    return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

# Flash Attention v3 分支
def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    if USE_FA3:
        return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)
```
这两个分支， q, k/v 的形状，q 的 head 数量，按照上面演示的数据，都是 k/v head 数量的两倍。Pytorch 跟 FlashAttentionV3 内部都可以自动执行分组查询注意力GQA。

训练和推理的投影逻辑相同。推理时 kv cache 按 n_kv_head 分配，不按 n_head：

```python
class KVCache:
    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers, device, dtype):
        self.batch_size = batch_size
        self.max_seq_len = seq_len
        self.n_layers = num_layers
        self.n_heads = num_heads
        self.head_dim = head_dim
        # Pre-allocate cache tensors: (n_layers, B, T, H, D)
        self.k_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)

class Engine:
    @torch.inference_mode()
    def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42):
        # "num_heads": m.n_kv_head 使用了跟 `class CausalSelfAttention` 中相同的 n_kv_head。
        kv_model_kwargs = {"num_heads": m.n_kv_head, "head_dim": m.n_embd // m.n_head, "num_layers": m.n_layer}
        kv_cache_prefill = KVCache(
            batch_size=1,
            seq_len=len(tokens),
            device=device,
            dtype=dtype,
            **kv_model_kwargs,
        )
```

注意：这里的 head_dim 依然是 m.n_embd // m.n_head；GQA 降低的是 KV 头的数量，但单个注意力头的维度大小必须与 Q 保持一致，否则点积 $QK^T$ 会因维度不匹配而报错。

上面 10/5 的例子里，KV cache 体积和带宽相对 MHA 大约减半。nanochat 这种小模型对 decode cache 不敏感，所以默认关着；长 decode 的大模型才更需要它。

更多见[KVCache](KVCache.md)。
