# FlashAttention

Flash Attention是一种加速attention计算的技术。FA仅仅支持BF16/FP8两种数据计算格式。

在nanochat中，FlashAttention的实现是专门针对 V3。

使用现成已编译完成的文件来完成attention计算。因为FlashAttention编译对环境要求极高。所以预先编译好，可以避免本地编译的各种问题。在 [Kernels](https://huggingface.co/kernels)中有已经编译完成的kernel，类似`varunneal/flash-attention-3`, `kernels-community/flash-attn3`. 

### 实际用法

环境应该使用 bf6/fp8 数据格式。

代码主要在 `nanochat/flash_attention.py`中。用法类似:

```python
def _load_flash_attention_3():
    major, _ = torch.cuda.get_device_capability()
    # FA3 kernels 目前支持 Hopper (sm90), Ada (sm89) and Ampere (sm80/sm86)
    if major == 9:
        hf_kernel = "varunneal/flash-attention-3"
        # 下载线上的 FA3 算子
        return get_kernel(hf_kernel).flash_attn_interface
    else:
        hf_kernel = "kernels-community/flash-attn3"
        if has_kernel(hf_kernel):
            # 下载线上的 FA3 算子
            return get_kernel(hf_kernel).flash_attn_interface
        else:
            return None

_fa3 = _load_flash_attention_3()

def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    if USE_FA3:
        # 执行attention计算
        return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)
```

## References

不同版本的论文在以下GitHub repository中有链接。

- [dao-ailab/flash-attention](https://github.com/dao-ailab/flash-attention)
