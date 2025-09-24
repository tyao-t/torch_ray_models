# Efficent attentions: https://github.com/rasbt/LLMs-from-scratch/blob/main/ch03/02_bonus_efficient-multihead-attention/mha-implementations.ipynb

import torch
import torch.nn as nn
class SelfAttention_v1_v2(nn.Module):
    def __init__(self, d_in, d_out, bias=False):
        super().__init__()
        # self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        # self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        # self.W_value = nn.Parameter(torch.rand(d_in, d_out))
        self.W_query = nn.Linear(d_in, d_out, bias=bias)
        self.W_key = nn.Linear(d_in, d_out, bias=bias)
        self.W_value = nn.Linear(d_in, d_out, bias=bias)
        
    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        
        attn_scores = queries @ keys.T # keys.transpose(-1, -2)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )

        context_vec = attn_weights @ values
        return context_vec

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length,
                 dropout, bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=bias)
        self.W_value = nn.Linear(d_in, d_out, bias=bias)
        self.dropout = nn.Dropout(dropout)
        # Auto goes to device and gets saved in state dict
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1)) # New

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(-1, -2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec


class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, bias=False):
        super().__init__()
        self.head_models = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, bias) 
             for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head_model(x) for head_model in self.head_models], dim=-1)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be a multiple of num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=bias)
        self.W_key = nn.Linear(d_in, d_out, bias=bias)
        self.W_value = nn.Linear(d_in, d_out, bias=bias)
        self.out_proj = nn.Linear(d_out, d_out, bias=bias)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x_q, x_k=None, x_v=None, *, is_causal=True):
        if x_k is None: x_k = x_q
        if x_v is None: x_v = x_q

        b, num_tokens_q, _ = x_q.shape
        _, num_tokens_k, _ = x_k.shape
        _, num_tokens_v, _ = x_v.shape
        assert num_tokens_k == num_tokens_v, "num_tokens_k and num_tokens_v must match"

        queries = self.W_query(x_q).view(b, num_tokens_q, self.num_heads, self.head_dim).transpose(1, 2)
        keys    = self.W_key(x_k).view(b, num_tokens_k, self.num_heads, self.head_dim).transpose(1, 2)
        values  = self.W_value(x_v).view(b, num_tokens_v, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = queries @ keys.transpose(-1, -2)  # (b, num_heads, num_tokens_q, num_tokens_k)

        if is_causal and num_tokens_q == num_tokens_k:
            mask_bool = self.mask.bool()[:num_tokens_q, :num_tokens_k]
            attn_scores.masked_fill_(mask_bool, float('-inf'))

        attn_weights = torch.softmax(attn_scores / (self.head_dim ** 0.5), dim=-1)
        attn_weights = self.attn_dropout(attn_weights) # Still # (b, num_heads, num_tokens_q, num_tokens_k)

        context_vec = attn_weights @ values  # (b, num_heads, num_tokens_q, head_dim)
        context_vec = context_vec.transpose(1, 2).contiguous().view(b, num_tokens_q, self.d_out)
        context_vec = self.resid_dropout(self.out_proj(context_vec))

        return context_vec
