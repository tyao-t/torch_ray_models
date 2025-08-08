class MHAFlashPyTorchSDP(nn.Module):
    def __init__(self, d_in, d_out, num_heads, dropout=0, qkv_bias=False):
        super().__init__()

        assert d_out % num_heads == 0, "d_out must be a multiple of num_heads"

        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape

        # (batch_size, num_tokens, 3 * embed_dim)
        qkv = self.qkv(x)

        # (batch_size, num_tokens, 3, num_heads, head_dim)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

        # (3, batch_size, num_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # 3 x (batch_size, num_heads, num_tokens, head_dim)
        queries, keys, values = qkv.unbind(0)

        # (batch_size, num_heads, num_tokens, head_dim)
        context_vec = nn.functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)

        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)

        return self.resid_dropout(self.out_proj(context_vec))

class MQAFlashPyTorchSDP(nn.Module):
    def __init__(self, d_in, d_out, num_heads, dropout=0, qkv_bias=False):
        super().__init__()

        assert d_out % num_heads == 0, "d_out must be a multiple of num_heads"

        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        self.q_proj = nn.Linear(d_in, d_out, bias=qkv_bias)  # (num_heads × head_dim)
        self.kv_proj = nn.Linear(d_in, 2 * self.head_dim, bias=qkv_bias)  # (2 × head_dim)
        
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape

        queries = self.q_proj(x)
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        queries = queries.permute(0, 2, 1, 3)  # (batch_size, num_heads, num_tokens, head_dim)

        kv = self.kv_proj(x)  # (batch_size, num_tokens, 2 × head_dim)
        kv = kv.view(batch_size, num_tokens, 2, self.head_dim)
        kv = kv.permute(2, 0, 1, 3)  # (2, batch_size, num_tokens, head_dim)
        keys, values = kv.unbind(0)  # (batch_size, num_tokens, head_dim)

        keys = keys.unsqueeze(1)  # (batch_size, 1, num_tokens, head_dim)
        values = values.unsqueeze(1)  # (batch_size, 1, num_tokens, head_dim)

        context_vec = nn.functional.scaled_dot_product_attention(
            queries, keys, values,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True,
        )  # (batch_size, num_heads, num_tokens, head_dim)

        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)
        return self.resid_dropout(self.out_proj(context_vec))

class GQAFlashPyTorchSDP(nn.Module):
    def __init__(self, d_in, d_out, num_heads, num_kv_heads=None, dropout=0, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be a multiple of num_heads"
        
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        self.num_kv_heads = num_kv_heads if num_kv_heads else 1
        assert num_heads % self.num_kv_heads == 0, "num_heads must be a multiple of num_kv_heads"
        self.num_queries_per_kv = num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.kv_proj = nn.Linear(d_in, 2 * self.num_kv_heads * self.head_dim, bias=qkv_bias)
        
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape

        queries = self.q_proj(x)
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        queries = queries.permute(0, 2, 1, 3)  # (batch_size, num_heads, num_tokens, head_dim)

        kv = self.kv_proj(x)
        kv = kv.view(batch_size, num_tokens, 2, self.num_kv_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)  # (2, batch_size, num_kv_heads, num_tokens, head_dim)
        keys, values = kv.unbind(0)  # (batch_size, num_kv_heads, num_tokens, head_dim)

        keys = keys.repeat_interleave(self.num_queries_per_kv, dim=1)  # (batch_size, num_heads, num_tokens, head_dim)
        values = values.repeat_interleave(self.num_queries_per_kv, dim=1)  # (batch_size, num_heads, num_tokens, head_dim)

        context_vec = nn.functional.scaled_dot_product_attention(
            queries, keys, values,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True,
        )  # (batch_size, num_heads, num_tokens, head_dim)

        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)
        return self.resid_dropout(self.out_proj(context_vec))