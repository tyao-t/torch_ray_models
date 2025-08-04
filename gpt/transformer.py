import torch
import torch.nn as nn
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