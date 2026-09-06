# FlashAttention

Flash Attention是一种加速attention计算的技术。FA仅仅支持BF16/FP8两种数据计算格式。

在nanochat中，FlashAttention的实现是，使用现成已编译完成的文件来完成attention计算。因为FlashAttention编译对环境要求极高。所以预先编译好，可以避免本地编译的各种问题。在 [Kernels](https://huggingface.co/kernels)中有已经编译完成的kernel，类似`varunneal/flash-attention-3`, `kernels-community/flash-attn3`. 代码主要在 `nanochat/flash_attention.py`中。

用法类似

```python
from kernels import get_kernel, has_kernel

flashAttn = None

hf_kernel = "kernels-community/flash-attn3"
if has_kernel(hf_kernel):
    # 下载已经预先编译好的文件到本地
    flashAttn = get_kernel(hf_kernel).flash_attn_interface

# 执行attention计算
flashAttn.flash_attn_func()
```

## References

不同版本的论文在以下GitHub repository中有链接。

- [dao-ailab/flash-attention](https://github.com/dao-ailab/flash-attention)
