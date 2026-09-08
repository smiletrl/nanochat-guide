# FP8

FP8是在一种在神经网络线性层（Linear Layer）内，进行矩阵运算时，对数据进行量化quantize的一种技术。该技术使用专属的FP8硬件单元，加速矩阵运算。相对于BF16，算力提升近一倍。

数据从原始的FP32/BF16被（有损）压缩到8位，进行计算。计算完成后，得到的weights/Biase，再根据缩放比例，复原到原来数据的32/16位数后保存。

这种量化quantize策略，对模型训练影响不是很大。因为深度学习本来也是一种统计概率科学，缩减精度有点像某种数据正则化。但是可以极大提高显存使用率，提升数据传输速率跟训练速度。

## 检测系统环境是否支持

`scripts/base_train.py` 检测是否启用

```python
if args.fp8:
    if device_type != "cuda":
        print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag")
    else:
        # 启用fp8
```

代码中没有做硬件的强检测。启动命令 `python -m scripts.base_train` 传递的参数 `--fp8` 是否启用, 以及当前环境是否是 `cuda`。实际上最好是检测下硬件的版本。因为 cuda 从第4代tensor cores（H100）开始支持 fp8。

不是所有的线性层都应该替换。fp8 要求线性层的输入维度跟输出维度是16的倍数，且这两个维度各自不能低于128。如果维度低于128，矩阵的量化损失可能干扰结果，而且性能提升也不明显。

## FP8实现流程


### 替换模型内部满足条件的线性层

启动命令 `python -m scripts.base_train` 后，`scripts/base_train.py` 在满足系统环境后，调用函数 `def convert_to_float8_training()`。

`nanochat/fp8.py` ，在模型被初始化到内存以后，通过扫描model的子树，从树的叶子节点开始，依次往上搜索，将linear层，替换为FP8的linear层。

```python
def convert_to_float8_training(module, *, config=None, module_filter_fn=None):
    """递归性地将满足条件的 nn.Linear 线性层替成自定义的 Float8Linear 层.
    """
    def _convert(mod, prefix=""):
        for name, child in mod.named_children():
            fqn = f"{prefix}.{name}" if prefix else name
            _convert(child, fqn)
            if isinstance(child, nn.Linear) and not isinstance(child, Float8Linear):
                if module_filter_fn is None or module_filter_fn(child, fqn):
                    setattr(mod, name, Float8Linear.from_float(child))

    _convert(module)
    return module
```

### 自定义fp8 线性层

`nanochat/fp8.py` 中的实现。

- 自定义的FP8的线性层 `Float8Linear` 拓展pytorch里的nn.Linear 层，重写前向传播forward跟反向传播backward 方法。

```python
class Float8Linear(nn.Linear):
    def forward(self, input):
        output = _Float8Matmul.apply(input_2d, self.weight)
```

- 前向传播跟后向传播用的fp8的精度不同，前向需要更高的精度，是 `torch.float8_e4m3fn`, 而反向传播对于梯度，需要更大的范围，是`torch.float8_e5m2`.

```python
@torch._dynamo.allow_in_graph
class _Float8Matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_2d, weight):
        # 需要更高的精度
        input_fp8, input_inv = _to_fp8(input_2d, torch.float8_e4m3fn)
        weight_fp8, weight_inv = _to_fp8(weight, torch.float8_e4m3fn)
    
    @staticmethod
    def backward(ctx, grad_output):
        # 需要更大的数据范围
        go_fp8, go_inv = _to_fp8(grad_output, torch.float8_e5m2)
```


- 在进行庞大的GEMM运算之前，先找到整个矩阵中最大的值，然后将float8的最大值除以矩阵的最大值，得到缩放比例scale。然后对矩阵元素乘以缩放比例，使得矩阵元素都被压缩到float8的表示范围内。

`nanochat/fp8.py`， 数据缩放

```python
@torch.no_grad()
def _to_fp8(x, fp8_dtype):
    fp8_max = torch.finfo(fp8_dtype).max
    # 找到张量中绝对值最大的数字
    amax = x.float().abs().max()
    # 得到缩放比例
    scale = fp8_max / amax.double().clamp(min=EPS)
    scale = scale.float()
    # 量化Quantize，得到（有损）压缩的值
    x_scaled = x.float() * scale
    x_clamped = x_scaled.clamp(-fp8_max, fp8_max)
    x_fp8 = x_clamped.to(fp8_dtype)
    # _scaled_mm 计算要求的缩放比例格式
    inv_scale = scale.reciprocal()
    return x_fp8, inv_scale
```

- 计算使用`torch._scale_mm()`, 在前向跟反向传播的计算过程中，第一个参数要求数据内存是行连续（row-major），而第二个参数要求内存是列连续（col-major）. 在反向传播的计算过程中，这里用到张量tensor的转置 `.t()`，重写分配连续内存`contiguous()`两个方法。

示例在Linear线性层反向传播时，

```python
@torch._dynamo.allow_in_graph
class _Float8Matmul(torch.autograd.Function):
    @staticmethod
    def backward(ctx, grad_output):
        in_fp8, in_inv, w_fp8, w_inv = ctx.saved_tensors

        # === 矩阵乘法 1: grad_input = grad_output @ weight ===
        # Shapes: [B, N] @ [N, K] -> [B, K]
        # 梯度使用 e5m2 (更大区间), 权重使用 e4m3 (更高精度)
        go_fp8, go_inv = _to_fp8(grad_output, torch.float8_e5m2)
        # go_fp8  [B, N] 在内存中是行连续, 适配_scaled_mm()的第一个参数要求
        # w_fp8 [N, K] 在内存中是行连续，需要转为列连续的方式，适配_scaled_mm()的第二个参数要求
        w_col = _to_col_major(w_fp8)
        grad_input = torch._scaled_mm(
            go_fp8,
            w_col,
            scale_a=go_inv,
            scale_b=w_inv,
            out_dtype=grad_output.dtype,
            use_fast_accum=False,
        )

        # === 矩阵乘法 2: grad_weight = grad_output.T @ input ===
        # Shapes: [N, B] @ [B, K] -> [N, K]
        # go_fp8 [B, N] 在内存中是行连续， go.T = [N, B] 转置后，就变成了列连续，需要加contiguous()，在内存中重新分配，变为行连续，适配_scaled_mm()的第一个参数要求。
        go_T = go_fp8.t().contiguous()  # [N, B] row-major 行连续
        in_col = _to_col_major(in_fp8)    # [B, K] column-major 列连续
        grad_weight = torch._scaled_mm(
            go_T,
            in_col,
            scale_a=go_inv,
            scale_b=in_inv,
            out_dtype=grad_output.dtype,
            use_fast_accum=False,
        )

        return grad_input, grad_weight
```

## References

- [Training Deep Neural Networks with 8-bit Floating Point Numbers](https://arxiv.org/abs/1812.08011)
- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)
