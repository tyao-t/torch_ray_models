import torch
import torch.nn as nn
from typing import Optional, Union

class RoPE(nn.Module):
    """
    Rotary Positional Embedding (RoPE)

    Args:
        dims:          head_dim（必须为偶数）
        context_length:预计算的最大序列长度
        base:          频率底数（一般 10000）
        traditional:   True=even-odd(相邻维成对)；False=split-half(前后半维成对)

    Input:
        x: (batch_size, num_tokens, num_heads, head_dim)
        offset: None / int / slice
            - None: 使用 [0:num_tokens)
            - int:  使用 [offset:offset+num_tokens)
            - slice: 直接用于切片（长度需等于 num_tokens）
    """
    def __init__(
        self,
        dims: int,
        context_length: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        super().__init__()
        assert dims % 2 == 0, "dims must be even"
        self.dims = dims
        self.half_dims = dims // 2
        self.context_length = context_length
        self.base = base
        self.traditional = traditional

        # ω_i = base^{-(2i/d)}，用 i/half_dims 等价于 (2i)/d
        i = torch.arange(self.half_dims, dtype=torch.float32) / self.half_dims  # (half_dims,)
        freqs = base ** (-i)                                                    # (half_dims,)

        # n = token positions
        positions = torch.arange(context_length, dtype=torch.float32)           # (context_length,)

        # θ_{n,i} = n * ω_i
        angles = torch.outer(positions, freqs)                                  # (context_length, half_dims)

        # 预存 cos/sin(θ)
        self.register_buffer("cos_freqs", torch.cos(angles), persistent=False)  # (T, half_dims)
        self.register_buffer("sin_freqs", torch.sin(angles), persistent=False)  # (T, half_dims)

    def forward(self, x: torch.Tensor, offset: Optional[Union[int, slice]] = None) -> torch.Tensor:
        # x: (batch_size, num_tokens, num_heads, head_dim)
        batch_size, num_tokens, num_heads, head_dim = x.shape
        assert head_dim == self.dims, f"last dim {head_dim} must equal dims={self.dims}"
        assert num_tokens <= self.context_length, (
            f"seq len {num_tokens} > context_length {self.context_length}"
        )

        # 选取本序列位置对应的 cos/sin 基底
        if offset is None:
            cos = self.cos_freqs[:num_tokens]     # (num_tokens, half_dims)
            sin = self.sin_freqs[:num_tokens]
        elif isinstance(offset, int):
            cos = self.cos_freqs[offset: offset + num_tokens]
            sin = self.sin_freqs[offset: offset + num_tokens]
            assert cos.size(0) == num_tokens, "offset + num_tokens exceeds precomputed context_length"
        elif isinstance(offset, slice):
            cos = self.cos_freqs[offset]
            sin = self.sin_freqs[offset]
            assert cos.size(0) == num_tokens, "slice length must equal num_tokens"
        else:
            raise TypeError("offset must be None, int, or slice")

        # 广播到 (batch_size, num_tokens, num_heads, half_dims)
        cos = cos.view(1, num_tokens, 1, self.half_dims).to(dtype=x.dtype, device=x.device)
        sin = sin.view(1, num_tokens, 1, self.half_dims).to(dtype=x.dtype, device=x.device)

        # 取“实/虚部”两路（两种配对方式）
        if self.traditional:
            # even-odd：相邻维成对 → real=x[...,0::2], imag=x[...,1::2]
            x_real = x[..., 0::2]                 # (batch_size, num_tokens, num_heads, half_dims)
            x_imag = x[..., 1::2]                 # (batch_size, num_tokens, num_heads, half_dims)
        else:
            # split-half：前半=real，后半=imag
            x_real = x[..., :self.half_dims]      # (batch_size, num_tokens, num_heads, half_dims)
            x_imag = x[..., self.half_dims:]      # (batch_size, num_tokens, num_heads, half_dims)

        # 真正旋转：real' = real·cos − imag·sin；imag' = imag·cos + real·sin
        real = x_real * cos - x_imag * sin
        imag = x_imag * cos + x_real * sin

        # 拼回最后一维
        if self.traditional:
            # 将 (real, imag) 交替还原为 even/odd 排列
            y = torch.stack((real, imag), dim=-1).reshape(batch_size, num_tokens, num_heads, head_dim)
        else:
            y = torch.cat((real, imag), dim=-1)   # 直接拼接回 (batch_size, num_tokens, num_heads, head_dim)

        return y.to(dtype=x.dtype)
