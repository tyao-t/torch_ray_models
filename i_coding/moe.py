import torch.nn as nn
import torch
class MoEFeedForwardSimplest(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.top_k: int = cfg["num_experts_per_tok"]
        self.num_experts: int = cfg["num_experts"]
        self.embed_dim: int = cfg["emb_dim"]
        self.hidden_dim: int = cfg["moe_intermediate_size"]
        self.dtype = cfg["dtype"]

        self.fc12 = nn.ModuleList([
            nn.Linear(self.embed_dim, 2*self.hidden_dim, bias=False, dtype=self.dtype)
            for _ in range(self.num_experts)
        ])

        self.fc3 = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.embed_dim, bias=False, dtype=self.dtype)
            for _ in range(self.num_experts)
        ])
        self.gate = nn.Linear(self.embed_dim, self.num_experts, bias=False)

    def forward(self, x):
        batch_size, num_tokens, emb_dim = x.shape
        scores = self.gate(x) # [batch_size, num_tokens, num_experts]

        topk_scores, topk_indices = torch.topk(             # topk_scores:  [batch, num_tokens, top_k]
            scores, k=self.top_k, dim=-1                    # topk_indices: [batch, num_tokens, top_k]  (专家下标)
        )                                                      
                                                               
        topk_probs = torch.softmax(topk_scores, dim=-1)     # [batch, num_tokens, top_k]

        expert_outputs = []
        for e in range(self.num_experts):
            # x_fc1/x_fc2/hidden/out 的维度分别是：
            #   x_fc1:  [batch, num_tokens, hidden_dim]
            #   x_fc2:  [batch, num_tokens, hidden_dim]
            #   hidden: [batch, num_tokens, hidden_dim]
            #   out:    [batch, num_tokens, embed_dim]
            x_fc1, x_fc2 = self.fc12[e](x).chunk(2, dim=-1)
            hidden = nn.SiLU(x_fc1) * x_fc2
            out = self.fc3[e](hidden)
            expert_outputs.append(out.unsqueeze(-2))           # [batch, num_tokens, 1, embed_dim]

        expert_outputs = torch.cat(expert_outputs, dim=-2)     # [batch, num_tokens, num_experts, embed_dim]

        total_probs = torch.zeros_like(scores)
        total_probs.scatter_(dim=-1, index=topk_indices, src=topk_probs)
        
        y = (total_probs.unsqueeze(-1) * expert_outputs).sum(dim=-2)
        return y

class MoEFeedForwardOptimized(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.top_k: int = cfg["num_experts_per_tok"]
        self.num_experts: int = cfg["num_experts"]
        self.embed_dim: int = cfg["emb_dim"]
        self.hidden_dim: int = cfg["moe_intermediate_size"]
        self.dtype = cfg["dtype"]

        self.fc12 = nn.ModuleList([
            nn.Linear(self.embed_dim, 2*self.hidden_dim, bias=False, dtype=self.dtype)
            for _ in range(self.num_experts)
        ])

        self.fc3 = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.embed_dim, bias=False, dtype=self.dtype)
            for _ in range(self.num_experts)
        ])
        self.gate = nn.Linear(self.embed_dim, self.num_experts, bias=False)

    def forward(self, x):
        scores = self.gate(x) # [batch_size, num_tokens, num_experts]

        topk_scores, topk_indices = torch.topk(             # topk_scores:  [batch, num_tokens, top_k]
            scores, k=self.top_k, dim=-1                    # topk_indices: [batch, num_tokens, top_k]  (专家下标)
        )                                                      
                                                               
        topk_probs = torch.softmax(topk_scores, dim=-1)     # [batch, num_tokens, top_k]
        batch_size, num_tokens, emb_dim = x.shape
        x_flat = x.reshape(batch_size * num_tokens, -1)
        out_flat = torch.zeros_like(x_flat, device=x.device, dtype=x.dtype)

        topk_probs_flat = topk_probs.reshape(batch_size * num_tokens, self.top_k)
        topk_indices_flat = topk_indices.reshape(batch_size * num_tokens, self.top_k)
        unique_experts = torch.unique(topk_indices_flat)

        for expert_id_tensor in unique_experts:
            expert_id = int(expert_id_tensor.item())
            mask = topk_indices_flat == expert_id # [batch_size * num_tokens, top_k]
            # if not mask.any(): continue

            token_mask = mask.any(dim=-1)
            selected_idx = token_mask.nonzero(as_tuple=False).squeeze(-1)
            if selected_idx.numel() == 0: continue

            x_selected_input = x_flat.index_select(dim=0, index=selected_idx)
            x_fc1, x_fc2 = self.fc12[expert_id](x_selected_input).chunk(2, dim=-1)
            hidden = nn.SiLU(x_fc1) * x_fc2
            expert_out = self.fc3[expert_id](hidden) # [num_relevant_indices, emb_dim]

            mask_selected = mask.index_select(dim=0, index=selected_idx) # mask[selected_idx] # [num_relevant_indices, top_k]
            slot_indices = mask_selected.int().argmax(dim=-1, keepdim=True) # [num_relevant_indices, 1]
            selected_probs = torch.gather(topk_probs_flat.index_select(dim=0, index=selected_idx), index=slot_indices, dim=-1)
            out_flat.index_add_(0, selected_idx, expert_out * selected_probs)

        probs_full = torch.softmax(scores, dim=-1) # [batch_size, num_tokens, num_experts]
        importance = probs_full.mean(sum=(0, 1)) / (batch_size * num_tokens) # [num_experts]，sum=1

        if self.load_use_weight:
            # 概率加权：基于 gating_probs 的概率质量统计
            # gating_probs 在每个 token 上对所有专家之和为 1
            load_raw = topk_probs.sum(dim=(0, 1))             # [num_experts]，和为 batch*seq_len
            load = load_raw / (batch_size * num_tokens)                 # [num_experts]，sum=1
        else:
            dispatch_one_hot = torch.zeros_like(scores)
            dispatch_one_hot.scatter_(dim=-1, index=topk_indices, src=torch.ones_like(topk_indices))
            load = dispatch_one_hot.sum(dim=(0, 1)) / (batch_size * num_tokens * self.top_k)  # [num_experts]，sum=1
        
        lb_loss = self.num_experts * torch.sum(importance * load)
        probs_full = torch.softmax(scores, dim=-1) # [batch_size, num_tokens, num_experts]
        # 熵正则：ent_loss = E_{batch,seq_len}[ sum_e p_e * log(p_e) ]  （= 负熵）
        ent_loss = (probs_full * (probs_full.clamp_min(eps)).log()).sum(dim=-1).mean()  # 标量

        aux_total = lb_coef * lb_loss + ent_coef * ent_loss
        aux: Dict[str, torch.Tensor] = {
            "lb_loss": lb_loss,
            "ent_loss": ent_loss,
            "total": aux_total,
        }

        return out_flat.reshape(batch_size, num_tokens, emb_dim)
