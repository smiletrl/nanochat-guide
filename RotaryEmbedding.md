# 旋转位置编码

旋转位置编码取代Google 2017《Attention Is All You Need》论文中的绝对位置编码，采用相对位置的编码方式，是当前大模型训练的位置编码的常见训练方式。

## 前置几何基础： 在二维平面上旋转一个点

假设在平面坐标系中，我们有一个点（或者一个向量），坐标是 $(x_1, x_2)$。
它的长度是 $r$，它与横轴的夹角是 $\alpha$。

根据最基础的三角函数定义，它的坐标可以写成：

1. $x_1 = r \cos\alpha$
2. $x_2 = r \sin\alpha$

现在，我们要把这个向量**旋转一个角度 $\theta$**，得到一个新点 $(y_1, y_2)$。
旋转后，这个向量的长度 $r$ 保持不变，但新的夹角变成了 $(\alpha + \theta)$。

那么，新点的坐标就是：

* $y_1 = r \cos(\alpha + \theta)$
* $y_2 = r \sin(\alpha + \theta)$

### 套用三角函数展开公式（和角公式）：

* $\cos(\alpha + \theta) = \cos\alpha \cos\theta - \sin\alpha \sin\theta$
* $\sin(\alpha + \theta) = \sin\alpha \cos\theta + \cos\alpha \sin\theta$

我们把它们分别乘以 $r$：


$$y_1 = (r \cos\alpha) \cos\theta - (r \sin\alpha) \sin\theta$$

$$y_2 = (r \sin\alpha) \cos\theta + (r \cos\alpha) \sin\theta$$

在开始时我们已经定义了 $x_1 = r \cos\alpha ，x_2 = r \sin\alpha$ 。把它们直接替换进去：


$$y_1 = x_1 \cos\theta - x_2 \sin\theta$$

$$y_2 = x_1 \sin\theta + x_2 \cos\theta$$

得到**二维旋转矩阵**形式就是：

$$\begin{bmatrix} y_1 \cr y_2 \end{bmatrix} = \begin{bmatrix} \cos\theta & -\sin\theta \cr \sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} x_1 \cr x_2 \end{bmatrix}$$

## 旋转核心思想

- 原始 Google 论文采用可训练的绝对位置编码，直接与词嵌入相加，然后一起进入神经网络学习。

- 旋转编码的设计非常巧妙，计算注意力的公式

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) \cdot V
$$

$Q K ^T $, 提前将 $Q$ 跟 $K$ 的语义编码按照不同token所在的位置，旋转一个角度。比如位置 $m$ 的token 旋转 $\theta_m$, 位置 $n$ 的token旋转 $\theta_n$ 。然后点积运算 

 $$\vec{q} \cdot \vec{k} = \vert{}\vec{q}\vert{} \vert{}\vec{k}\vert{} \cos(\theta_m - \theta_n)$$

- 因为 $Q/K $ 分别根据各自的位置旋转了特定的角度，导致旋转后的向量做点积的时候，自动算进去了它们之间的旋转夹角 $(\theta_m - \theta_n)$，而这个夹角就代表了它们的位置差异。

- 角度的计算规则，以 $Q$ 为例。它的最后一个维度，拆分为两两一对向量，比如 $[X_1, X_2, X_3, \dots, X_n]$，被拆成 $n / 2$ 对，
$[[X_1, X_{n/2}], [X_2, X_{n/2+1}], \dots, [X_{n/2-1}, X_n]]$。这是参照 Hugging Face Transformers 库中 [LLaMA](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) 的实现风格（出于计算效率考虑），也是目前大模型工程代码库的通用标准。在[苏剑林的原始 RoFormer 论文](https://arxiv.org/abs/2104.09864)中，采用的是相邻维度拆分，也就是 $[[X_1, X_2], [X_3, X_4], \dots]$。拆分后每对向量的旋转有不同的频率。

 $$\theta_i = \text{base}^{-2i/d}$$

- 上式中， $i$ 为拆分后的对数的索引。对于同一个token而言，对数索引越大，频率越小。 $d$ 为维度，一般是 head dimension 的一半。base 是一个常数。

- 拿token在sequence中的位置索引乘以该token语义编码（word embedding）内每一对二维特征的旋转频率，得到token内每对二维特征真实的旋转角度

 $$\theta_t = t * \theta_i $$
 
- $t$ 表示位置的步长（stride），之所以写 $t$， 主要是nanochat中写的 $t$（应该是借鉴了RNN/LSTM中的写法）。实际或许写 $p$ 更合理些，因为这代表一个token 在sequence序列中的位置。我们得到的 $\theta_t$ ， 就是不同位置 $m, n$ 的token的旋转角度 $\theta_m , \theta_n$。

参考**前置基础几何**中的角度旋转公式：

$$y_1 = x_1 \cos\theta - x_2 \sin\theta$$

$$y_2 = x_1 \sin\theta + x_2 \cos\theta$$

对应 $Q/K$ 的旋转后每一对二维特征的新语义编码（注意，nanochat中实际是对角度 $-\theta$ 旋转）：

* $\tilde{q}_m = (x_1 \cos\theta_m + x_2 \sin\theta_m, \ -x_1 \sin\theta_m + x_2 \cos\theta_m)$
* $\tilde{k}_n = (k_1 \cos\theta_n + k_2 \sin\theta_n, \ -k_1 \sin\theta_n + k_2 \cos\theta_n)$


应用注意力机制的核心矩阵乘法：**计算旋转后的 $\tilde{q}_m$ 和 $\tilde{k}_n$ 的点积**（第一项乘第一项 + 第二项乘第二项）：


$$\tilde{q}_m \cdot \tilde{k}_n = (x_1 \cos\theta_m + x_2 \sin\theta_m)(k_1 \cos\theta_n + k_2 \sin\theta_n) + (-x_1 \sin\theta_m + x_2 \cos\theta_m)(-k_1 \sin\theta_n + k_2 \cos\theta_n)$$

把上面的式子完全乘开并合并同类项，所有带 $\theta_m$ 和 $\theta_n$ 的单项都会利用三角恒等式组合成：

$$\cos\theta_m\cos\theta_n + \sin\theta_m\sin\theta_n = \mathbf{\cos(\theta_m - \theta_n)}$$

$$\sin\theta_m\cos\theta_n - \cos\theta_m\sin\theta_n = \mathbf{\sin(\theta_m - \theta_n)}$$

最终化简出的结果为：

$$\tilde{q}_m \cdot \tilde{k}_n = (x_1 k_1 + x_2 k_2) \mathbf{\cos((m - n)\theta)} + (x_2 k_1 - x_1 k_2) \mathbf{\sin((m - n)\theta)}$$

这个结果有一个非常好的性质， $(x_1 k_1 + x_2 k_2)$ 正好是原始 $Q*K$ 没有旋转前的点积运算结果。这说明旋转后的点积，保留了原始语义编码的点积，又加上了相对位置差 $(m - n)$ 的性质。

**注：以上推导基于实数域的简单计算，实际推导参考论文中的复数域计算。*

## 实现步骤

`karpathy/nanochat`项目中的步骤，主要在文件`nanochat/gpt.py`中

对于每个sequence而言，它的内部每个token的位置，token内部二维特征拆分后的每对旋转频率都是一样的，我们可以提前将旋转角度的正弦/余弦值计算出来。 在 `gpt.py` 内部，专门预先计算角度的函数 `_precompute_rotary_embeddings`：

```python
    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        # 准备二维特征对的索引i
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        # 得到二维特征对的旋转频率
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # 位置索引
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # 计算不同token位置下，每个二特征对的旋转角度。
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        # 增加batch 跟 head dims，方便后续广播运算 broadcasting。
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] 
        return cos, sin
```

在对每一个sequance进行旋转计算的时候，用不同sequance里token的原始语义编码拆分后的二维特征对，乘以旋转角度的正弦余弦值，计算一个张量的旋转编码：

```python
def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # 多头注意力
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # 最后一维分为两批
    # 注意：由于 nanochat 在底层实现时对旋转角度取了负号（即旋转 -θ），因此下方代码中的 y1 公式与上文标准推导的符号相反，这是完全等价的。
    y1 = x1 * cos + x2 * sin # 旋转特征组
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)
```

分别旋转 $q, k$

```python
q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
```

计算注意力

```python
y = flash_attn.flash_attn_func(q, k, v, ...)
```

## References

- [苏剑林. (Mar. 23, 2021). 《Transformer升级之路：2、博采众长的旋转式位置编码 》](https://spaces.ac.cn/archives/8265)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Hugging Face Rotary Pos Embedding](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)

```python

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@use_kernel_forward_from_hub("rotary_pos_emb")
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```