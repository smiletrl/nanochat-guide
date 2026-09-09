# FP8

FP8是在一种在大模型线性层（Linear Layer）内，进行矩阵运算时，对数据进行量化quantize的一种技术。将浮点数从16位转为8位，使用FP8硬件单元（CUDA设备， H100或更新架构），加速大矩阵运算。相对于BF16浮点数，**算力提升近一倍**。

## 矩阵量化数值演示

假设我们有一个待量化的 2×2 矩阵 **X**（存储为 FP32）：

$$
X = \begin{bmatrix}
1.0 & -2.0 \\
3.0 & -6.0
\end{bmatrix}
$$

**步骤 1：寻找绝对最大值**
矩阵中绝对值最大的元素为 `-6.0`，即 `amax = 6.0`。

**步骤 2：计算缩放因子**
`float8_e4m3fn` 的表示上限为 `fp8_max = 448`。
缩放因子 `scale = 448 / 6.0 ≈ 74.6667`。

**步骤 3：矩阵缩放（FP32 精度下计算）**
将原矩阵所有元素乘以缩放因子，使其尽量填满 FP8 的表示范围：

$$
X_{scaled} = X \times 74.6667 = \begin{bmatrix}
1.0 \times 74.6667 & -2.0 \times 74.6667 \\
3.0 \times 74.6667 & -6.0 \times 74.6667
\end{bmatrix} = \begin{bmatrix}
74.67 & -149.33 \\
224.0 & -448.0
\end{bmatrix}
$$

**步骤 4：截断与量化（转换为 FP8）**
将缩放后的值限制在 `[-448, 448]` 区间，并转换为 FP8 格式（`float8_e4m3fn`）。该过程会触发“舍入到最近偶数”的硬件级四舍五入，引入精度损失。

量化后的矩阵 **X_fp8**（以离散数值表示）为：

$$
X_{fp8} \approx \begin{bmatrix}
74.67 & -149.33 \\
224.0 & -448.0
\end{bmatrix}
$$

（注：实际硬件中 `74.67` 和 `-149.33` 会被舍入到 FP8 格式下最接近的离散网格点，例如 `74.67` 可能变为 `76.0` 或 `72.0`，取决于具体的步长，此处为了清晰展示缩放原理，保留了数学上的中间值。）

**步骤 5：保存逆缩放因子**
供 `torch._scaled_mm` 在矩阵乘法后还原数值幅度：

$$
inv\_scale = \frac{1}{74.6667} \approx 0.01339
$$

#### 关键观察
- **最大值对齐**：原始矩阵中绝对值最大的 `-6.0` 被精准映射到了 FP8 的边界 `-448`，从而**最大化利用了 FP8 的动态范围**。
- **有损压缩**：原本连续的浮点数（如 `74.6667`）被强制转换，只能近似为 FP8 网格上的离散点，这是精度损失的主要来源。
- **反向传播差异**：注意在反向传播中，代码使用了 `float8_e5m2`（范围更大但精度更低）来量化梯度 `grad_output`，以适应梯度可能出现的较大数值波动。
- **缩放粒度的选择：Tensorwise vs. Rowwise（工程实践关键）**：
  虽然上述演示基于 `tensorwise`（这也是 `--fp8-recipe` 的默认值，主打速度优势），但在实际大模型训练中，**`rowwise` 策略通常更为通用和稳健**。
  
  - **`tensorwise`（默认）**：全矩阵共享一个缩放因子。计算开销极低（只需一次 `amax` 归约），但如果矩阵中存在极端离群值（Outliers），整个矩阵的数值范围会被该离群值“撑开”，导致其余大部分正常数值的量化分辨率降低。
  
  - **`rowwise`（更通用）**：针对矩阵的 *每一行* 独立计算 `amax` 和缩放因子（即 `amax = x.abs().max(dim=-1, keepdim=True)`）。这使得每一行都能独立地最大化利用 FP8 的 `[-448, 448]` 动态范围，对异常值有极强的鲁棒性，从而获得更低的量化误差和更稳定的训练收敛性。
  
  不过，`rowwise` 的计算模式需要启动更多的 CUDA 内核来处理逐行缩放，吞吐量会略低于 `tensorwise`。nanochat 保留了这两种选择，允许用户在**速度（tensorwise）**与**精度/稳定性（rowwise）**之间根据硬件（H100+）情况进行权衡。

## fp8实现流程

大模型内部的FP8实现，依赖硬件支持，自定义线性层，手动执行张量矩阵的量化。

1. **开启训练**

执行如下命令。附带参数 `--fp8`，启用fp8。

```bash
python -m scripts.base_train \
  --max-seq-len=512 \
  --device-batch-size=4 \
  --eval-tokens=512 \
  --core-metric-every=-1 \
  --total-batch-size=2048 \
  --num-iterations=1 \
  --eval-every=-1 
  --fp8
```

可选参数 `--fp8-recipe`，默认值 `tensorwise`, 也可以传 `rowwise`。但是实际上代码内部仅支持 `tensorwise`。见 `nanochat/fp8.py` 的 `class Float8LinearConfig`. 

2. **检测系统环境是否支持**

`scripts/base_train.py` 检测是否启用

```python
if args.fp8:
    if device_type != "cuda":
        print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag")
    else:
        # 启用fp8
```

代码中检测硬件环境是否 `cuda`。实际上最好是检测下硬件的版本。因为 cuda 从第4代tensor cores（H100）开始支持 fp8。如果当前cuda版本低于H100，fp8 是不支持的。

3。 **替换特定的线性层**

`scripts/base_train.py` 中定义了替换条件

```python
def fp8_module_filter(mod: nn.Module, fqn: str) -> bool:
    if not isinstance(mod, nn.Linear):
        return False
    # fp8 要求线性层的输入维度跟输出维度是16的倍数
    if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
        return False
    # 这两个维度各自不能低于128。
    if min(mod.in_features, mod.out_features) < 128:
        return False
    return True
```

不是所有的线性层都应该替换。条件不满足的话，矩阵的量化会干扰结果，性能没有提升。

`nanochat/fp8.py` 执行线性层替换。 通过扫描大模型的模块树，从树的叶子节点开始，依次往上搜索，将linear层，替换为FP8的linear层。

```python
def convert_to_float8_training(module, *, config=None, module_filter_fn=None):
    """递归性地将满足条件的 nn.Linear 线性层替成自定义的 Float8Linear 层.
    """
    def _convert(mod, prefix=""):
        for name, child in mod.named_children():
            fqn = f"{prefix}.{name}" if prefix else name
            # 执行递归逻辑
            _convert(child, fqn)
            if isinstance(child, nn.Linear) and not isinstance(child, Float8Linear):
                if module_filter_fn is None or module_filter_fn(child, fqn):
                    # 执行线性层的替换
                    setattr(mod, name, Float8Linear.from_float(child))
```

4. **自定义fp8线性层**

`nanochat/fp8.py` 中自定义FP8的线性层 `Float8Linear`， 拓展 pytorch `nn.Linear` 层，重写前向传播 forward、反向传播 backward 方法。

```python
class Float8Linear(nn.Linear):
    def forward(self, input):
        output = _Float8Matmul.apply(input_2d, self.weight)
```

- 前向/反向传播的fp8的精度不同

```python
@torch._dynamo.allow_in_graph
class _Float8Matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_2d, weight):
        # 前向传播需要更高的精度 torch.float8_e4m3fn
        input_fp8, input_inv = _to_fp8(input_2d, torch.float8_e4m3fn)
        weight_fp8, weight_inv = _to_fp8(weight, torch.float8_e4m3fn)
    
    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播需要更大的数据范围 torch.float8_e5m2
        go_fp8, go_inv = _to_fp8(grad_output, torch.float8_e5m2)
```

- 量化步骤（匹配文档初始的矩阵示例）

FP8 量化的核心是将高精度（如 FP32/BF16）矩阵中的元素，通过缩放映射到低精度的 FP8 表示范围内（float8_e4m3fn 范围为 [-448, 448]）。以下以 nanochat/fp8.py 中的 _to_fp8 函数逻辑为例，演示矩阵的数值变化过程。

`nanochat/fp8.py` 数据缩放:

```python
@torch.no_grad()
def _to_fp8(x, fp8_dtype):
    fp8_max = torch.finfo(fp8_dtype).max          # e4m3fn 最大值为 448
    amax = x.float().abs().max()                  # 找出矩阵绝对值的最大值
    scale = fp8_max / amax.double().clamp(min=EPS) # 计算缩放因子
    scale = scale.float()
    x_scaled = x.float() * scale                  # 放大矩阵元素
    x_clamped = x_scaled.clamp(-fp8_max, fp8_max) # 截断越界值
    x_fp8 = x_clamped.to(fp8_dtype)               # 转换为 FP8（发生四舍五入/有损压缩）
    inv_scale = scale.reciprocal()                # 保存逆缩放因子，用于后续矩阵乘法的还原
    return x_fp8, inv_scale
```

- 计算使用 `torch._scale_mm()`, 在前向/反向传播的计算过程中，第一个参数要求数据内存是行连续（row-major），而第二个参数要求内存是列连续（col-major）. 在反向传播的计算过程中，这里用到张量tensor的转置 `.t()`，重写分配连续内存`contiguous()`两个方法。

```python
@torch._dynamo.allow_in_graph
class _Float8Matmul(torch.autograd.Function):
    @staticmethod
    def backward(ctx, grad_output):
        # === 矩阵乘法 1: grad_input = grad_output @ weight ===
        # Shapes: [B, N] @ [N, K] -> [B, K]
        # 梯度使用 e5m2 (更大区间), 权重使用 e4m3 (更高精度)
        go_fp8, go_inv = _to_fp8(grad_output, torch.float8_e5m2)
        # go_fp8  [B, N] 在内存中是行连续, 适配_scaled_mm()的第一个参数要求
        # w_fp8 [N, K] 在内存中是行连续，需要转为列连续的方式，适配_scaled_mm()的第二个参数要求
        w_col = _to_col_major(w_fp8)
        grad_input = torch._scaled_mm(...)
```

## References

- [Training Deep Neural Networks with 8-bit Floating Point Numbers](https://arxiv.org/abs/1812.08011)
- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)
