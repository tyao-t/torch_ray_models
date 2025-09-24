import torch
import torch.nn as nn

class MoEFeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_experts_per_tok = cfg["num_experts_per_tok"]
        self.num_experts = cfg["num_experts"]
        self.gate = nn.Linear(cfg["emb_dim"], cfg["num_experts"], bias=False, dtype=cfg["dtype"])

        # meta device to reduce memory pressure when initializing the model before loading weights
        meta_device = torch.device("meta")
        self.fc1 = nn.ModuleList([
            nn.Linear(
                cfg["emb_dim"], cfg["moe_intermediate_size"],
                bias=False, dtype=cfg["dtype"], device=meta_device)
            for _ in range(cfg["num_experts"])]
        )
        self.fc2 = nn.ModuleList([
            nn.Linear(
                cfg["emb_dim"], cfg["moe_intermediate_size"],
                bias=False, dtype=cfg["dtype"], device=meta_device
                )
            for _ in range(cfg["num_experts"])]
        )
        self.fc3 = nn.ModuleList([
            nn.Linear(
                cfg["moe_intermediate_size"], cfg["emb_dim"],
                bias=False, dtype=cfg["dtype"], device=meta_device
                )
            for _ in range(cfg["num_experts"])]
        )

    def forward(self, x):
        b, seq_len, embed_dim = x.shape
        scores = self.gate(x)  # (b, seq_len, num_experts)
        topk_scores, topk_indices = torch.topk(scores, self.num_experts_per_tok, dim=-1)
        topk_probs = torch.softmax(topk_scores, dim=-1)
        
        expert_outputs = []
        for e in range(self.num_experts):
            hidden = torch.nn.functional.silu(self.fc1[e](x)) * self.fc2[e](x)
            out = self.fc3[e](hidden)
            expert_outputs.append(out.unsqueeze(-2))
        expert_outputs = torch.cat(expert_outputs, dim=-2)  # (b, t, num_experts, emb_dim)

        gating_probs = torch.zeros_like(scores)

        for i in range(self.num_experts_per_tok):
            indices = topk_indices[..., i:i+1]
            prob = topk_probs[..., i:i+1]
            gating_probs.scatter_(dim=-1, index=indices, src=prob)
        gating_probs = gating_probs.unsqueeze(-1)  # (b, t, num_experts, 1)
        
        # Weighted sum over experts
        y = (gating_probs * expert_outputs).sum(dim=-2)
        return y


        # For some reason, the version below is slower than the naive version
        # above that computes all experts, even the unused ones

        # def forward(self, x):
        #     scores = self.gate(x)  # (b, seq_len, num_experts)
        #     topk_scores, topk_indices = torch.topk(scores, self.num_experts_per_tok, dim=-1)
        #     topk_probs = torch.softmax(topk_scores, dim=-1)
        #     y = torch.zeros_like(x)
        #
        #     for i in range(self.num_experts_per_tok):
        #         # expert_indices is (b, seq_len) with values in [0, num_experts)
        #         expert_indices = topk_indices[..., i]
        #         prob = topk_probs[..., i].unsqueeze(-1)  # (b, seq_len, 1)
        #
        #         # For each expert, process only the tokens assigned to it
        #         for e in range(self.num_experts):
        #             mask = (expert_indices == e)  # (b, seq_len) boolean mask
        #             if mask.any():
        #                 selected = x[mask]  # (num_tokens_e, emb_dim)
        #                 out = self.fc3[e](torch.nn.functional.silu(self.fc1[e](selected)) * self.fc2[e](selected))
        #                 y[mask] += prob[mask] * out
        #     return y

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            head_dim=cfg["head_dim"],
            num_kv_groups=cfg["n_kv_groups"],
            qk_norm=cfg["qk_norm"],
            dtype=cfg["dtype"]
        )
        if cfg["num_experts"] > 0:
            self.ff = MoEFeedForward(cfg)
        else:
            self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-6)

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x, next_cache = self.att(x, mask, cos, sin, start_pos=start_pos, cache=cache)  # Shape [batch_size, num_tokens, emb_size]
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut  # Add the original input back

        return x, next_cache


class Qwen3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Main model parameters
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        self.trf_blocks = nn.ModuleList(  # ModuleList since Sequential can only accept one input, and we need `x, mask, cos, sin`
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        # Reusuable utilities
        if cfg["head_dim"] is None:
            head_dim = cfg["emb_dim"] // cfg["n_heads"]
        else:
            head_dim = cfg["head_dim"]
        cos, sin = compute_rope_params(
            head_dim=head_dim,
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"]
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg
        self.current_pos = 0  # Track current position in KV cache


    def forward(self, in_idx, cache=None):
        # Forward pass
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds

        num_tokens = x.shape[1]
        if cache is not None:
            pos_start = self.current_pos
            pos_end = pos_start + num_tokens
            self.current_pos = pos_end
            mask = torch.triu(
                torch.ones(pos_end, pos_end, device=x.device, dtype=torch.bool), diagonal=1
            )[pos_start:pos_end, :pos_end]
        else:
            pos_start = 0  # Not strictly necessary but helps torch.compile
            mask = torch.triu(
                torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), diagonal=1
            )
        # Shape (1, 1, num_tokens, num_tokens) to broadcast across batch and heads
        mask = mask[None, None, :, :]

        next_cache = []
        for i, block in enumerate(self.trf_blocks):
            blk_cache = cache.get(i) if cache else None
            x, new_blk_cache = block(x, mask, self.cos, self.sin,
                                     start_pos=pos_start,
                                     cache=blk_cache)
            if cache is not None:
                cache.update(i, new_blk_cache)
            next_cache.append(new_blk_cache)

        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))
        return logits

    def reset_kv_cache(self):
        self.current_pos = 0