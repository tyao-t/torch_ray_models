import torch

class RoPE:
    def __init__(self, dims: int, context_length: int,
        base: int = 10000, traditional: bool = False,
    ):
        assert dims % 2 == 0, "dims must be even"
        self.dims = dims
        self.context_length = context_length
        half_dims = dims // 2
        inner = torch.arange(0, half_dims, dtype=torch.float32) / half_dims # torch.arange(...) * 2 / half_dims
        freqs = torch.pow(base, -inner)
        token_positions = torch.arange(context_length)
        angles = torch.outer(token_positions, freqs)
        self.register_buffer("cos_freqs", torch.cos(angles))
        self.register_buffer("sin_freqs", torch.sin(angles))
        self.base = base
        self.half_dims = half_dims
        self.traditional = traditional

    def __call__(self, x: torch.Tensor, offset: slice | None = None) -> torch.Tensor:
        batch_size, num_tokens, num_heads, head_dim = x.shape
        
        cos_basis = self.cos_freqs[:num_tokens, :] if offset is None else self.cos_freqs[offset, :]
        sin_basis = self.sin_freqs[:num_tokens, :] if offset is None else self.sin_freqs[offset, :]
        
        if self.traditional:
            x = x.view(batch_size, num_tokens, num_heads, self.half_dims, 2)
            x1 = x[..., 0]
            x2 = x[..., 1]
        else:
            x1 = x[..., 0 : self.half_dims]
            x2 = x[..., self.half_dims : self.dims]
        
        cos_basis = cos_basis.reshape(num_tokens, 1, self.half_dims)
        sin_basis = sin_basis.reshape(num_tokens, 1, self.half_dims)
        
        real = x1 * cos_basis - x2 * sin_basis
        imag = x2 * cos_basis + x1 * sin_basis
        
        if self.traditional:
            y = torch.stack([real, imag], dim=-1)
            y = y.reshape(N, S, H, D)
        else:
            y = torch.cat([real, imag], dim=-1)
            y = y.reshape(N, S, H, D)
            
        return y.to(x.dtype)