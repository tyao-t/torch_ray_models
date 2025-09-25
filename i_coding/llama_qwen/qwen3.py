import torch.nn as nn
import torch

class GroupedQueryAttention(nn.Module):
    def __init__(
        self, d_in, d_out, num_heads, num_kv_groups, qk_norm=False, dtype=None
    ):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        assert d_out % num_heads == 0, "`d_in` must be divisible by `num_heads` if `head_dim` is not set"
        head_dim = d_out // num_heads

        self.head_dim = head_dim
        self.d_out = d_out # num_heads * head_dim

        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)

        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        if qk_norm:
            self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
            self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, mask, cos, sin):
        batch_size, num_tokens, d_in = x.shape
        
        queries = self.W_query(x)  # (b, num_tokens, num_heads * head_dim)
        keys = self.W_key(x)       # (b, num_tokens, num_kv_groups * head_dim)
        values = self.W_value(x)   # (b, num_tokens, num_kv_groups * head_dim)

        # Reshape
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        # Optional normalization
        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys = self.k_norm(keys)

        # Apply RoPE
        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)

        # Expand K and V to match number of heads
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        # Attention
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)

        context = (attn_weights @ values).transpose(1, 2).reshape(batch_size, num_tokens, self.d_out)
        return self.out_proj(context)

class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention (MQA): many query heads (H), single shared K/V head.
    - Queries: (B, H, T, D)
    - Keys/Values: (B, 1, T, D)  (shared across heads)
    Broadcasting handles head expansion during matmuls—no repeat_interleave needed.
    """
    def __init__(self, d_in, d_out, num_heads, qk_norm=False, dtype=None):
        super().__init__()
        assert d_out % num_heads == 0, "`d_out` must be divisible by `num_heads`"
        head_dim = d_out // num_heads

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.d_out = d_out  # == num_heads * head_dim

        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        # Single shared K/V head in MQA
        self.W_key   = nn.Linear(d_in, head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, head_dim, bias=False, dtype=dtype)

        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        if qk_norm:
            self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
            self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(self, x, mask, cos, sin):
        """
        x:    (B, T, d_in)
        mask: broadcastable to (B, H, T, T) (e.g., (B, 1, 1, T, T) or (B, 1, T, T) or (1, 1, T, T))
        cos/sin: RoPE caches compatible with apply_rope
        """
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        # Projections
        q = self.W_query(x)                 # (B, T, H*D)
        k = self.W_key(x)                   # (B, T, D)
        v = self.W_value(x)                 # (B, T, D)

        # Reshape to heads
        q = q.view(B, T, H, D).transpose(1, 2)   # (B, H, T, D)
        k = k.view(B, 1, T, D)                   # (B, 1, T, D)
        v = v.view(B, 1, T, D)                   # (B, 1, T, D)

        # Optional Q/K RMSNorm (acts on last dim)
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        # RoPE (assumed to work with shapes (B, H|1, T, D))
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Scaled dot-product attention (broadcasting over the singleton K/V head)
        # (B, H, T, D) @ (B, 1, D, T) -> (B, H, T, T)
        attn_scores = torch.matmul(q, k.transpose(2, 3)) / (D ** 0.5)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)

        # (B, H, T, T) @ (B, 1, T, D) -> (B, H, T, D)
        context = torch.matmul(attn_weights, v)

        # Merge heads and project out
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_out)  # (B, T, H*D)
        return self.out_proj(context)
