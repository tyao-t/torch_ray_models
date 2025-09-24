import torch.nn as nn
import torch 

class MultiheadAttention(nn.module):
    def __init__(self, max_context_size, num_heads, d_in, d_out, dropout, qkv_bias, is_causal):
        assert d_out % num_heads == 0
        self.W_q = nn.linear(d_in, d_out, qkv_bias)
        self.W_k = nn.linear(d_in, d_out, qkv_bias)
        self.W_v = nn.linear(d_in, d_out, qkv_bias)
        self.max_context_size = max_context_size
        self.num_heads = num_heads
        self.d_in = d_in
        self.d_out = d_out
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_out, d_out, bias=False)

        self.register_buffer("mask", torch.triu(torch.ones((max_context_size, max_context_size)), diagonal=1))

    def forward(self, x):
        batch_size, num_tokens, d_in = x.shape
        head_dim = self.d_out // self.num_heads
        assert d_in == self.d_in
        assert num_tokens <= self.max_context_size
        q = self.W_q(x).view(batch_size, num_tokens, self.num_heads, head_dim).transpose(1, 2)
        k = self.W_k(x).view(batch_size, num_tokens, self.num_heads, head_dim).transpose(1, 2)
        v = self.W_v(x).view(batch_size, num_tokens, self.num_heads, head_dim).transpose(1, 2)

        attn_scores = q @ k.transpose(-1, -2)
        mask = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores._masked_fill(mask, -torch.inf)
        attn_weights = nn.softmax(attn_scores * torch.rsqrt(self.head_dim), dim=-1)
        attn_weights = self.attention_dropout(attn_weights)
        
        context_vec = attn_weights @ v
        context_vec = context_vec.transpose(-1, -2).reshape(batch_size, num_tokens, self.d_out)
        context_vec = context_vec.transpose(-1, -2).continuous().view(batch_size, num_tokens, self.d_out)
        return self.resid_dropout(self.out_proj(context_vec))