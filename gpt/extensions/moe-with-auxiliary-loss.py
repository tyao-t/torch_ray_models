import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class MoEFeedForwardWithAuxiliaryLoss(nn.Module):
    r"""
    Mixture-of-Experts 版本的前馈网络（FFN），带可选的 auxiliary losses：
      - lb_loss  : 负载均衡损失（Switch/GShard 风格），鼓励每个专家使用更均匀
      - ent_loss : 熵正则，鼓励路由分布更分散（更高熵）

    记号（只用于注释/说明，不用于变量名）：
      - batch         : batch size
      - seq_len       : 序列长度
      - embed_dim     : 词嵌入/特征维度
      - num_experts   : 专家数量（E）
      - top_k         : 每个 token 激活的专家数（K）
      - hidden_dim    : MoE 内部中间维度（H）

    输入/输出：
      - 输入 x:  [batch, seq_len, embed_dim]
      - 输出 y:  [batch, seq_len, embed_dim]
      - 当 return_aux=True 时，额外返回 aux 字典：
          aux = {
            "lb_loss": 标量张量（负载均衡损失，值越大表示越均衡；训练时通常直接以小系数加到总损失中）,
            "ent_loss": 标量张量（负熵；系数为正时等价于鼓励更高熵分布）,
            "total": lb_coef * lb_loss + ent_coef * ent_loss
          }

    说明：
      - probs_full：对“全部专家”做 softmax 得到的概率；用于统计 importance 与熵
      - gating_probs：仅在 top-k 专家位置非零（按 top-k 概率散射到 E 维），用于加权汇总专家输出
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        self.top_k: int = cfg["num_experts_per_tok"]          # top_k
        self.num_experts: int = cfg["num_experts"]            # E
        self.embed_dim: int = cfg["emb_dim"]                  # D
        self.hidden_dim: int = cfg["moe_intermediate_size"]   # H
        self.dtype = cfg["dtype"]

        # 路由器：根据 token 表示生成每个专家的打分
        self.gate = nn.Linear(self.embed_dim, self.num_experts, bias=False, dtype=self.dtype)

        # 使用 "meta" 设备占位；第一次前向时再物化到真实 device（降低初始化显存压力）
        meta = torch.device("meta")
        self.fc1 = nn.ModuleList([
            nn.Linear(self.embed_dim, self.hidden_dim, bias=False, dtype=self.dtype, device=meta)
            for _ in range(self.num_experts)
        ])
        self.fc2 = nn.ModuleList([
            nn.Linear(self.embed_dim, self.hidden_dim, bias=False, dtype=self.dtype, device=meta)
            for _ in range(self.num_experts)
        ])
        self.fc3 = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.embed_dim, bias=False, dtype=self.dtype, device=meta)
            for _ in range(self.num_experts)
        ])

    @torch.no_grad()
    def _materialize_if_needed(self, device: torch.device) -> None:
        """把 meta 权重移动到实际 device（仅在第一次前向时需要）。"""
        if self.fc1[0].weight.device.type != "meta":
            return
        for m in list(self.fc1) + list(self.fc2) + list(self.fc3):
            m.to(device)

    def forward(
        self,
        x: torch.Tensor,                                       # [batch, seq_len, embed_dim]
        *,
        return_aux: bool = False,                              # 是否返回辅助损失
        lb_coef: float = 1e-2,                                 # 负载均衡损失系数（常取很小值）
        ent_coef: float = 0.0,                                 # 熵正则系数
        eps: float = 1e-9,                                     # 数值稳定项
        load_use_weight: bool = False,                         # True: 用概率加权的 load；False: 用 one-hot 计数
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        返回：
          - 若 return_aux=False: y
          - 若 return_aux=True : (y, aux) ，aux 见类注释
        """
        batch, seq_len, embed_dim = x.shape
        assert embed_dim == self.embed_dim, f"Expected embed_dim={self.embed_dim}, got {embed_dim}"
        device = x.device
        self._materialize_if_needed(device)

        # ---------------------------------------------------------
        # 1) 路由打分与 softmax 概率（对“全部专家”）
        # ---------------------------------------------------------
        scores = self.gate(x)                                  # [batch, seq_len, num_experts]
        probs_full = F.softmax(scores, dim=-1)                 # [batch, seq_len, num_experts]
        # 注：probs_full 仅用于统计 importance/熵等；不改变 top-k 路由决策

        # 选择 top_k 个专家（仅用于派发与最终加权）
        topk_scores, topk_indices = torch.topk(
            scores, k=self.top_k, dim=-1
        )                                                      # topk_scores:  [batch, seq_len, top_k]
                                                               # topk_indices: [batch, seq_len, top_k]  (专家下标)
        topk_probs = F.softmax(topk_scores, dim=-1)            # [batch, seq_len, top_k]  (top_k 内归一化)

        # ---------------------------------------------------------
        # 2) 计算所有专家输出（朴素实现：每个专家都各算一遍）
        #    SwiGLU: silu(fc1(x)) * fc2(x) → fc3(...)
        # ---------------------------------------------------------
        expert_outputs = []
        for e in range(self.num_experts):
            # x_fc1/x_fc2/hidden/out 的维度分别是：
            #   x_fc1:  [batch, seq_len, hidden_dim]
            #   x_fc2:  [batch, seq_len, hidden_dim]
            #   hidden: [batch, seq_len, hidden_dim]
            #   out:    [batch, seq_len, embed_dim]
            x_fc1 = self.fc1[e](x)
            x_fc2 = self.fc2[e](x)
            hidden = F.silu(x_fc1) * x_fc2
            out = self.fc3[e](hidden)
            expert_outputs.append(out.unsqueeze(-2))           # [batch, seq_len, 1, embed_dim]

        # 拼成一个专家维度的张量
        expert_outputs = torch.cat(expert_outputs, dim=-2)     # [batch, seq_len, num_experts, embed_dim]

        # ---------------------------------------------------------
        # 3) 构造 gating_probs：把 top_k 概率散射到 num_experts 维度
        #    gating_probs[b, t, e] = 该 token 分配到专家 e 的概率（其余专家为 0）
        # ---------------------------------------------------------
        gating_probs = torch.zeros_like(scores)                 # [batch, seq_len, num_experts]
        gating_probs.scatter_(dim=-1, index=topk_indices, src=topk_probs)
        # 最终输出：按专家维度做加权求和
        y = (gating_probs.unsqueeze(-1) * expert_outputs).sum(dim=-2)  # [batch, seq_len, embed_dim]

        if not return_aux:
            return y, None

        # ---------------------------------------------------------
        # 4) auxiliary losses（可选）：
        #    - importance: 各专家的平均“概率质量”（来自 probs_full；和为 1）
        #    - load      : 各专家的平均“实际指派份额”（和为 1）
        #    - lb_loss   : num_experts * sum_i importance[i] * load[i]
        #    - ent_loss  : 负熵（值越小熵越高；在总损失里系数取正可鼓励更高熵）
        # ---------------------------------------------------------
        # importance[i] = mean_{batch, seq_len}( probs_full[..., i] )
        importance = probs_full.mean(dim=(0, 1))                # [num_experts]，sum=1

        # 计算 load（两种方式任选其一）
        if load_use_weight:
            # 概率加权：基于 gating_probs 的概率质量统计
            # gating_probs 在每个 token 上对所有专家之和为 1
            load_raw = gating_probs.sum(dim=(0, 1))             # [num_experts]，和为 batch*seq_len
            load = load_raw / (batch * seq_len)                 # [num_experts]，sum=1
        else:
            # one-hot 计数：对 top_k 专家计数（每 token 恰好计数 top_k 次）
            dispatch_one_hot = torch.zeros_like(scores)         # [batch, seq_len, num_experts]
            dispatch_one_hot.scatter_(dim=-1, index=topk_indices, src=torch.ones_like(topk_probs))
            load = dispatch_one_hot.sum(dim=(0, 1)) / (batch * seq_len * self.top_k)  # [num_experts]，sum=1

        # 负载均衡损失（值越小表示越均衡；训练时通常以小系数直接加到总损失中）
        lb_loss = self.num_experts * torch.sum(importance * load)  # 标量

        # 熵正则：ent_loss = E_{batch,seq_len}[ sum_e p_e * log(p_e) ]  （= 负熵）
        ent_loss = (probs_full * (probs_full.clamp_min(eps)).log()).sum(dim=-1).mean()  # 标量

        aux_total = lb_coef * lb_loss + ent_coef * ent_loss
        aux: Dict[str, torch.Tensor] = {
            "lb_loss": lb_loss,
            "ent_loss": ent_loss,
            "total": aux_total,
        }
        return y, aux


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class MoEFeedForwardWithoutComputingAllExperts(nn.Module):
    """
    MoE FFN（不展平 batch/seq_len 轴）：
      - 仅计算被路由到的专家（避免对未命中的专家做前向）
      - 直接用二维索引 (b_idx, t_idx) 回填到 y（index_put_ with accumulate=True）

    维度说明（仅用于注释）：
      batch       : B
      seq_len     : T
      embed_dim   : D
      num_experts : E
      top_k       : K
      hidden_dim  : H
    """
    def __init__(self, cfg: Dict):
        super().__init__()
        self.top_k       = cfg["num_experts_per_tok"]        # K
        self.num_experts = cfg["num_experts"]                # E
        self.embed_dim   = cfg["emb_dim"]                    # D
        self.hidden_dim  = cfg["moe_intermediate_size"]      # H
        self.dtype       = cfg["dtype"]

        # Router: [B, T, D] -> [B, T, E]
        self.gate = nn.Linear(self.embed_dim, self.num_experts, bias=False, dtype=self.dtype)

        # 用 "meta" 设备占位；第一次前向时再 materialize 到真实 device
        meta = torch.device("meta")
        self.fc1 = nn.ModuleList([
            nn.Linear(self.embed_dim, self.hidden_dim, bias=False, dtype=self.dtype, device=meta)
            for _ in range(self.num_experts)
        ])
        self.fc2 = nn.ModuleList([
            nn.Linear(self.embed_dim, self.hidden_dim, bias=False, dtype=self.dtype, device=meta)
            for _ in range(self.num_experts)
        ])
        self.fc3 = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.embed_dim, bias=False, dtype=self.dtype, device=meta)
            for _ in range(self.num_experts)
        ])

    @torch.no_grad()
    def _materialize_if_needed(self, device: torch.device):
        """把 meta 权重移动到实际 device（仅第一次需要）。"""
        if self.fc1[0].weight.device.type != "meta":
            return
        for m in list(self.fc1) + list(self.fc2) + list(self.fc3):
            m.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, embed_dim]
        return:
          y: [batch, seq_len, embed_dim]
        """
        batch, seq_len, embed_dim = x.shape
        assert embed_dim == self.embed_dim
        device = x.device
        self._materialize_if_needed(device)

        # 1) 路由打分与 top-k 选择（按 token）
        scores = self.gate(x)                                     # [B, T, E]
        topk_scores, topk_indices = torch.topk(
            scores, k=self.top_k, dim=-1
        )                                                         # topk_scores : [B, T, K]
                                                                  # topk_indices: [B, T, K] (expert id)
        topk_probs = F.softmax(topk_scores, dim=-1)               # [B, T, K]   (每 token 的 K 专家权重)

        # 2) 输出初始化
        y = torch.zeros_like(x)                                   # [B, T, D]

        # 3) 找出本 batch 实际被命中的专家集合（仅处理这些专家）
        unique_experts = torch.unique(topk_indices)               # [M], M ≤ E

        # 4) 对每个被命中的 expert e：
        #    - 聚合该 expert 在每个 token 上的权重 w_e(b,t) = sum_k prob[b,t,k] * 1[topk_idx[b,t,k]==e]
        #    - 仅对 w_e>0 的 token 过 expert e 的 FFN
        #    - 用二维坐标 (b_idx, t_idx) 将加权结果回填到 y（累加）
        for e in unique_experts.tolist():
            # 标记 top-k 中哪些位置选中了 expert e
            assigned = (topk_indices == e)                       # [B, T, K] (bool)

            # 聚合成“每个 token 对 expert e 的总权重”
            weight_e = (topk_probs * assigned.float()).sum(-1)   # [B, T]

            # 选出真正命中的 token 坐标
            b_idx, t_idx = torch.nonzero(weight_e > 0, as_tuple=True)  # 各为 [n_sel]
            if b_idx.numel() == 0:
                continue

            # 取这些 token 的输入，过 expert e 的 FFN
            x_sel = x[b_idx, t_idx, :]                           # [n_sel, D]
            h = F.silu(self.fc1[e](x_sel)) * self.fc2[e](x_sel)  # [n_sel, H]
            out_e = self.fc3[e](h)                               # [n_sel, D]

            # 取对应权重并回填：y[b_idx, t_idx, :] += weight * out_e
            w_sel = weight_e[b_idx, t_idx].unsqueeze(-1)         # [n_sel, 1]
            y.index_put_((b_idx, t_idx), w_sel * out_e, accumulate=True)

        return y

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