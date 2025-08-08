import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import MultiHeadAttention
from extensions.efficient_attention import MHAFlashPyTorchSDP

class FeedForward(nn.Module):
    def __init__(self, emb_dim, *, dropout, bias):
        super().__init__()
        self.out_proj = nn.Linear(4 * emb_dim, emb_dim, bias=config.bias)
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim, bias=config.bias),
            nn.GELU(approximate="none"), # nn.GELU(approximate="tanh")
            self.out_proj,
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg["emb_dim"], bias=True)
        self.ln_2 = nn.LayerNorm(cfg["emb_dim"], bias=True)
        self.attn = MHAFlashPyTorchSDP(d_in=cfg["emb_dim"], d_out=cfg["emb_dim"], \
            num_heads=cfg["n_heads"], dropout=cfg["drop_rate"], qkv_bias=cfg["bias"])
        self.ff = FeedForward(cfg["emb_dim"], dropout=cfg["drop_rate"], bias=cfg["bias"])

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.ff(self.ln_2(x))
        return x

class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.emb_dim = emb_dim
        self.weight = nn.Parameter(torch.ones(emb_dim, dtype=torch.float32))

    def forward(self, x):
        means = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(means + self.eps)
        return (x_normed * self.weight).to(dtype=x.dtype)

# torch.manual_seed(123)
# example_batch = torch.randn(2, 3, 4)
# rms_norm = RMSNorm(emb_dim=example_batch.shape[-1])
# rmsnorm_pytorch = torch.nn.RMSNorm(example_batch.shape[-1], eps=1e-5)
# assert torch.allclose(rms_norm(example_batch), rmsnorm_pytorch(example_batch))

class RMSNormQwen3(nn.Module):
    def __init__(self, emb_dim, eps=1e-6, bias=False, qwen3_compatible=True):
        super().__init__()
        self.eps = eps
        self.qwen3_compatible = qwen3_compatible
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        input_dtype = x.dtype

        if self.qwen3_compatible:
            x = x.to(torch.float32)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        norm_x = norm_x * self.scale

        if self.shift is not None:
            norm_x = norm_x + self.shift

        return norm_x.to(input_dtype)

class FeedForwardLlamaQwen(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.input_proj  = nn.Linear(embed_dim, hidden_dim * 2)
        self.output_proj = nn.Linear(hidden_dim, embed_dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        candidate, gate = self.input_proj(x).chunk(2, dim=-1)
        gated = self.act(candidate) * gate # SwiGLU
        return self.output_proj(gated)