# KV cache

大模型在推理阶段，做注意力计算时，我们将公式中的 $K/V$ 数据缓存起来，提高计算效率。训练阶段不需要 $K/V$ 缓存。

## 推理为什么需要 KV cache

大模型计算注意力的公式，

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) \cdot V
$$

在模型推理阶段，token是一个接一个地生成的。每生成一个新的token，它的 $Q$ 需要跟已经生成的内容中的每一个token的 $K$ 以及 $V$ 进行计算。如果每生成一个新的token，就把现有token的 $K V$ 都算一遍，将是极大的计算浪费。

举例：当前已经推理出来的结果是 `[12, 34, 178, 19]`, 这里用一个整数表示 Vocabulary 里的一个token id。在预测新的token时，将最后的token `19` 的 $Q$ 跟 `[12, 34, 178, 19]` 的 $K/V$ 做运算。假定新的token的id是 `17`， 我们得到推理结果为 `[12, 34, 178, 19, 17]`。 继续预测下一个token时，我们需要将 `17` 的 $Q$ 跟 `[12, 34, 178, 19, 17]` 的 $K/V$ 做运算。显然，token `[12, 34, 178, 19]` 的 $K/V$ 需要重复计算两次。

由此， 我们可以提前将推理结果的每个token的 $K/V$ 保存起来。每生成一个新的token，就将这个新token的 $K/V$ 追加到前面token的缓存中。预测下一个token的时候，就可以直接拿来用了。

## 大模型推理两阶段

1. Prefill 阶段（处理prompt）

提前将提示词（prompt token）的 $K/V$ 缓存起来，然后复制一份或者多份（生成样本的数量）。这样每一个样本都可以直接使用相同的 $K/V$ 缓存。这种策略一般称为预填充(Prefill)。

假设用户输入的 prompt 转换成 token id 是 `[12, 34]` 。模型首先一次性读取这两个 prompt token，算出它们的 $K$ 和 $V$，并且存入 KV Cache 中。此时cache里只有 `[12, 34]`。得到最后一位 `34` 的logits。

2. Decode阶段（自动回归生成）

这里假定只生成一条回答，不考虑并发。

单条生成时 `num_samples=1`；多条采样只是把同一份 prompt cache 复制多份，每步仍然 `T=1`。也就是每次只生成一个token。

* **循环1**：用Prefill 阶段 `34` 的logits 采出 `178`，并加入生成结果 → 序列是 `[12, 34, 178]`。再 $forward(178)$，在forward内先追加 `178` 的 $K/V$, $cache = [12, 34, 178]$，然后得到最后一位 `178` 的logits。

* **循环2**：用 `178` 的logits 采出 `19`，并加入生成结果 → 序列是 `[12, 34, 178, 19]`。再 $forward(19)$，在forward内追加 `19` 的 $K/V$, $cache = [12, 34, 178, 19]$，然后得到最后一位 `19` 的logits。

Decode 小结：

- **Engine 循环**：先采样并写入生成序列，再 forward 这个 token。
- 一次 forward 内部：先追加这个 token 的 $K/V$，再产出它的 logits。

这里的推理流程严格按照 `nanochat/engine.py` 中推理引擎的实现方式整理的。具体详情见下文 **推理代码调用路径** 的第3，4，5节。

## 训练为什么不需要 KV cache

1. 模型训练的调用方式：`scripts/base_train.py`，

```python
    for micro_step in range(grad_accum_steps):
        loss = model(x, y)
```

此处 `model()` 对应 `nanochat/gpt.py` 中 `class GPT(nn.Module)` 的方法 `def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):`。这里传递的参数 `x, y` 对应了 `idx, targets`。`kv_cache` 使用的是默认值 None。

2. 模型推理的调用方式：`nanochat/engine.py`：

```python
logits = self.model.forward(ids, kv_cache=kv_cache_decode)[:, -1, :]
```

此处 `self.model.forward()`，同样对应了 class GPT 的 forward 方法。传递的参数对应了 `idx, kv_cache`。而 `targets`为None。

训练过程的一次`forward`调用是完整的一次 batch 以及 sequence 的自注意力计算。不同调用之间的token是独立的。比如sequence token的length是128K，甚至1M，那这一次前向forward的计算，就会把这个batch里每个sequence的注意力计算全部完成。跟下一次前向计算的token之间就没有关系了。

推理过程的一次`forward`调用，则不同。在 Prefill 阶段，推理的一次调用，batch 固定为1，sequence 的length 为 prompt token的长度 > 1。 而在Decode阶段，batch可能 > 1 (并发采样)，但是每个 sequence 的 length 固定为 1，因为每次只预测下一个token。这就意味着，第 (N) 次 Decode 要看见 Prefill 以及前 N−1次 Decode 写入的 (KV)。那前面的Prefill 以及前 N-1 次的 $KV$ 都需要保存在显存中。

3. 不要跟 `nanochat/gpt.py` 模型中的 `generate()` 方法混淆：

```python
class GPT(nn.Module):
    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
```

这个方法同样是做推理，但是没有传 kv_cache。 这是因为这个方法是用来作为一个基准测试。它在 `nanochat/engine.py` 中作为行内测试，用来校准手写的推理引擎 `class Engine` 是否推理正确。

```python
if __name__ == "__main__":
    stream = model.generate(prompt_tokens, **kwargs)
    stream = engine.generate(prompt_tokens, num_samples=1, **kwargs)
    print(f"Match: {reference_ids == generated_tokens}")
```

## Q 为什么不需要缓存

训练和推理里， $Q$ 都是由当前这次 前向传播（`forward`）的 token 算出来、用完即弃。

Decode 下一步会换一个新 token，它的 $Q$ 必须重算，和上一拍的 $Q$ 没有复用关系。当前这拍的 $Q$ 却要和已经出现过的所有 token 的 $K/V$ 做注意力，所以要缓存的是 $KV$，不是 $Q$。

Prefill 时 prompt 每个位置也会算 $Q$，但只在这一次 `forward` 里用；Decode 不会再拿这些 $Q$ 出来，故也不存。

## 推理引擎 Engine

文件`nanochat/engine.py`中，定义了一个 `KVCache` 类:

```python

class KVCache:
    """
    这个类的写法，是为了适配 Flash Attention 3的flash_attn_with_kvcache 接口
    """

    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers, device, dtype):
        self.batch_size = batch_size
        self.max_seq_len = seq_len
        self.n_layers = num_layers
        self.n_heads = num_heads
        self.head_dim = head_dim
        # 提前分配好缓存的张量: (n_layers, B, T, H, D)
        self.k_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        self.v_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        # 对于每个batch，seq已经生成好的长度(FA3 需要int32)
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        # 这个是把上一个 token 的 embedding 混进当前 token。可以先不用管
        self.prev_embedding = None
    
    def get_layer_cache(self, layer_idx):
        """Return (k_cache, v_cache) views for a specific layer."""
        return self.k_cache[layer_idx], self.v_cache[layer_idx]
    
    def prefill(self, other):
        """
         从另一个缓存中复制数据到当前缓存中。主要用于先做batch=1的prefill，然后并行生成多个回答结果。
        """
        assert self.get_pos() == 0, "Cannot prefill a non-empty KV cache"
        assert self.n_layers == other.n_layers and self.n_heads == other.n_heads and self.head_dim == other.head_dim
        assert self.max_seq_len >= other.max_seq_len
        other_pos = other.get_pos()
        self.k_cache[:, :, :other_pos, :, :] = other.k_cache[:, :, :other_pos, :, :]
        self.v_cache[:, :, :other_pos, :, :] = other.v_cache[:, :, :other_pos, :, :]
        self.cache_seqlens.fill_(other_pos)
    
    def advance(self, num_tokens):
        """将缓存位移，加上Decode新的token长度(Prefill 时加的是整段 prompt 长度)"""
        self.cache_seqlens += num_tokens
```

这个类用来存储 kv 缓存信息。同时提供了读取缓存跟 prefill 缓存的方法。这个类没有提供写入缓存的方法。

## 推理代码调用路径

阅读代码过程中，可以实际运行代码，在调用栈里打上断点，或者日志，加深理解。

1. 执行类似如下命令，`--num-iterations=1` 是指仅训练一次更新模型的权重参数，用于演示。

```bash
python -m scripts.base_train \
  --depth=4 \
  --max-seq-len=512 \
  --device-batch-size=4 \
  --eval-tokens=512 \
  --core-metric-every=-1 \
  --total-batch-size=2048 \
  --num-iterations=1 \
  --eval-every=-1 
```
2. `scripts/base_train.py`, 在训练过程中，执行一次推理采样。

```python
if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
    prompts = [
        "The capital of France is", # 这里用一个prompt提示词作为演示
    ]
    # 调用推理引擎开始推理
    sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
```

3. `nanochat/engine.py`, 推理引擎 `class Engine` 根据上一个步骤的prompt提示词, 开始推理。首先用提示词的token 做一次prefill。将提示词的 kv 保存到`kv_cache_prefill`里。

    这里的`self.model` 对应的是 `nanochat/gpt.py`中的 `class GPT`。理论上说，这里的调用也可以写成 `self.model(ids, kv_cache=kv_cache_prefill)`。直接用forward更直观些。

```python
def generate_batch(self, tokens, num_samples=1, **kwargs):
def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42):
        # 运行 batch = 1的提示词的预填充（prefill）
        m = self.model.config
        kv_model_kwargs = {"num_heads": m.n_kv_head, "head_dim": m.n_embd // m.n_head, "num_layers": m.n_layer}
        kv_cache_prefill = KVCache(
            batch_size=1,
            seq_len=len(tokens),
            device=device,
            dtype=dtype,
            **kv_model_kwargs,
        )
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        # 这一步主要是计算提示词的kv_cache。logits的最后一个token的结果会被用来预测推理的第一个token
        # ids 的形状是 (1, T_prompt)
        logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
        logits = logits[:, -1, :].expand(num_samples, -1)  # (num_samples, vocab_size)
        
        # 将prefill的cache复制，同时删掉prefill cache
        kv_length_hint = (len(tokens) + max_tokens) if max_tokens is not None else self.model.config.sequence_len
        kv_cache_decode = KVCache(
            batch_size=num_samples,
            seq_len=kv_length_hint,
            device=device,
            dtype=dtype,
            **kv_model_kwargs,
        )
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill
        
        while True:
            # 根据上一轮的logits采样新的token
            next_ids = sample_next_token(logits, rng, temperature, top_k)  # (B, 1)
            sampled_tokens = next_ids[:, 0].tolist()

            # 开始执行实际的推理。ids的shape是torch.Size([num_samples, 1])。T为1，且只有一个token进入模型的前向forward过程。
            # token_column 来自本轮采样。
            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)

            # 传入的这个ids token 依赖kv_cache_decode中保存的前面已经推理出来的token的kv cache。
            # 同时会先把这个token追加到kv_cache_decode中，然后再计算ids的logits。
            # ids 的形状是(num_samples, 1)
            logits = self.model.forward(ids, kv_cache=kv_cache_decode)[:, -1, :]  # (B, vocab_size)
```

4. `nanochat/gpt.py`，模型的前向传播，最终进入 `class CausalSelfAttention()`的`def forward()`方法。

```python
class CausalSelfAttention(nn.Module):
    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        # 进入推理的分支
        k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
        y = flash_attn.flash_attn_with_kvcache(
            q, k_cache, v_cache,
            k=k, v=v,
            cache_seqlens=kv_cache.cache_seqlens,
            causal=True,
            window_size=window_size,
        )
        # 将缓存位移往前更新T（sequence 长度）步。 
        # 如果是prefill阶段，T 为 prompt token length。如果是Decode 阶段，T 固定为 1。
        if self.layer_idx == kv_cache.n_layers - 1:
            kv_cache.advance(T)
```

5. `nanochat/flash_attention.py`, 这里计算注意力，同时更新kv cache。

如果使用flash attention v3，kv cache 在flash attention 内部更新

```python
def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                            causal=False, window_size=(-1, -1)):
    if USE_FA3:
        # 这个函数内部直接将k,v 更新到 kv cache中。
        return _fa3.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
            causal=causal, window_size=window_size
        )
```

如果使用Pytorch的SDPA(Scaled_dot_product_attention), 则需要手动管理kv cache.

```python
def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                            causal=False, window_size=(-1, -1)):
    # 首先将新的kv原地更新到 kv cache中， 跟 Flash Attention v3中的行为一致。
    if k is not None and v is not None:
        k_cache[:, pos:pos+T_new, :, :] = k
        v_cache[:, pos:pos+T_new, :, :] = v

    end_pos = pos + T_new
    k_full = k_cache[:, :end_pos, :, :]
    v_full = v_cache[:, :end_pos, :, :]

    # 然后再计算注意力。这个计算好的注意力，在神经网络中，继续沿着module层往下走，用来计算logits。
    y_sdpa = _sdpa_attention(q_sdpa, k_sdpa, v_sdpa, window_size, enable_gqa)
```
