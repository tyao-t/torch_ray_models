import torch.nn as nn
import torch

class TransformerBlockGPT2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg["emb_dim"], bias=True)
        self.ln_2 = nn.LayerNorm(cfg["emb_dim"], bias=True)
        # self.attn = MHAFlashPyTorchSDP(d_in=cfg["emb_dim"], d_out=cfg["emb_dim"], \
        #     num_heads=cfg["n_heads"], dropout=cfg["drop_rate"], qkv_bias=cfg["bias"])
        self.ff = FeedForwardGPT2(cfg["emb_dim"], dropout=cfg["drop_rate"], bias=cfg["bias"])

    def forward(self, x):
        x_normalized = self.ln_1(x)
        x = x + self.attn(x_normalized, x_normalized, x_normalized)
        x = x + self.ff(self.ln_2(x))
        return x

class FeedForwardGPT2(nn.module):
    def __init__(self, emb_dim, drop_rate, bias=False):
        self.upward_project = nn.Linear(emb_dim, 4*emb_dim, bias=bias)
        self.downward_project = nn.Linear(4*emb_dim, emb_dim, bias=bias)
        self.gelu = nn.GELU(approximate="tanh")
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.upward_project(x)
        x = self.gelu(x)
        x = self.downward_project(x)
        x = self.dropout(x)
        return x

class FeedForwardLlamaQwen3(nn.module):
    def __init__(self, emb_dim, hidden_dim):
        self.in_projs = nn.linear(emb_dim, 2*hidden_dim)
        self.act = nn.SiLU() # self.swish
        self.out_proj = nn.Linear(hidden_dim, emb_dim)

    def forward(self, x):
        hidden1, hidden2 = self.in_projs(x).chunk(2, dim=-1)
        candidate, gate = hidden1, hidden2
        return self.out_proj(self.act(gate) * candidate)
    
class RMSNorm(nn.module):
    def __init__(self, eps, emb_dim, use_bias):
        self.eps = eps
        self.shift = nn.Parameter(torch.ones(emb_dim))
        self.bias = nn.Parameter(torch.zeros(emb_dim)) if use_bias else None
    
    def forward(self, x):
        var = x.pow(2).mean(dim=-1, keep_dim=True)
        x_normalized = x * torch.rsqrt(var + self.eps)
        x_normalized *= self.shift
        if self.bias is not None:
            x_normalized += self.bias

        return x_normalized