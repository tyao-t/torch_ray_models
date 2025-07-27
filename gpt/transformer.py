import torch
import torch.nn as nn
from attention import MultiHeadAttention

gelu_exact = nn.GELU(approximate='none')  
gelu_approx = nn.GELU(approximate='tanh') 

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"], bias=cfg["bias"]),
            nn.GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"], bias=cfg["bias"]),
            nn.Dropout(cfg["drop_rate"])
        )

    def forward(self, x):
        return self.layers(x)

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class TransformerBlockBootstrap(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            bias=cfg["bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"], bias=cfg["bias"])
        self.norm2 = LayerNorm(cfg["emb_dim"], bias=cfg["bias"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
    
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        # x = self.drop_shortcut(x)
        x = x + shortcut
        
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        # x = self.drop_shortcut(x)
        x = x + shortcut
        return x

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg["emb_dim"], bias=cfg["bias"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"], bias=cfg["bias"])
        
        self.att = nn.MultiheadAttention(
            embed_dim=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            bias=cfg["bias"],
            batch_first=True
        )

        self.out_proj = nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"], bias=cfg["bias"])
        self.ff = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"], bias=cfg["bias"]),
            nn.GELU(),
            self.out_proj,
            nn.Dropout(cfg["drop_rate"])
        )
        self.resid_dropout = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        num_tokens = x.size(1)
        mask = torch.triu(torch.ones(num_tokens, num_tokens), diagonal=1).bool()
        
        x_norm = self.norm1(x)
        # attn_out, attn_w = self.att(
        #     query=x_norm,
        #     key=x_norm,
        #     value=x_norm,
        #     is_causal = True
        #     # attn_mask=mask.to(x.device)
        # )
        # x = x + self.resid_dropout(attn_out)
        x = x + self.resid_dropout(self.att(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            is_causal = True,
            need_weights = False
        )[0])
        x = x + self.ff(self.norm2(x))

        return x