import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import MultiHeadAttention
from extensions.efficient_attention import MHAFlashPyTorchSDP

class FeedForward(nn.Module):
    def __init__(self, emb_dim, *, dropout, bias):
        super().__init__()
        self.out_proj = nn.Linear(4 * emb_dim, emb_dim, bias=bias)
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim, bias=bias),
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
        x_normalized = self.ln_1(x)
        x = x + self.attn(x_normalized, x_normalized, x_normalized)
        x = x + self.ff(self.ln_2(x))
        return x

# 优点
# 计算效率更高：Layer Norm 需要计算均值和标准差，而 RMS Norm 只需要计算均方根，减少了计算量。例如在 GPT-2 模型中，使用 RMS Norm 相比 Layer Norm 可减少约 18% 的训练时间。
# 数值稳定性更好：RMS Norm 不进行均值归一化，避免了 Layer Norm 中可能出现的均值计算精度问题，以及方差趋近于零导致的数值不稳定问题。其分母始终为正，不会出现 Layer Norm 中方差可能为零的情况，在低精度训练（如 FP16、BF16）场景下优势明显。
# 参数效率高：RMS Norm 仅需学习一个缩放参数 γ，而 Layer Norm 需要学习 γ 和 β 两个参数，参数量减少一半，这使得 RMS Norm 在模型训练和推理过程中更加高效，减少了参数更新和存储的开销。
# 更适合长序列模型：RMS Norm 对批次大小和输入分布的变化更加鲁棒，无需依赖批量统计量，特别适用于长序列模型，如 Transformer 架构。
# 缺点：RMS Norm 不去均值，在某些对均值信息敏感的任务中可能表现不如 Layer Norm，例如图像数据任务中，均值归 0 能稳定训练，而 RMS Norm 不能，可能导致训练不稳定。

class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.emb_dim = emb_dim
        self.weight = nn.Parameter(torch.ones(emb_dim, dtype=torch.float32))

    def forward(self, x):
        means = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(means + self.eps)
        return (x_normed * self.weight).to(dtype=x.dtype)

# torch.manual_seed(123)
# example_batch = torch.randn(2, 3, 4)
# rms_norm = RMSNorm(emb_dim=example_batch.shape[-1])
# rmsnorm_pytorch = torch.nn.RMSNorm(example_batch.shape[-1], eps=1e-5)
# assert torch.allclose(rms_norm(example_batch), rmsnorm_pytorch(example_batch))

class RMSNormQwen3(nn.Module):
    def __init__(self, emb_dim, eps=1e-6, bias=False, qwen3_compatible=True):
        super().__init__()
        self.eps = eps
        self.qwen3_compatible = qwen3_compatible
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        input_dtype = x.dtype

        if self.qwen3_compatible:
            x = x.to(torch.float32)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        norm_x = norm_x * self.scale

        if self.shift is not None:
            norm_x = norm_x + self.shift

        return norm_x.to(input_dtype)

class FeedForwardLlamaQwenMoreEfficient(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.input_proj  = nn.Linear(embed_dim, hidden_dim * 2)
        self.output_proj = nn.Linear(hidden_dim, embed_dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        candidate, gate = self.input_proj(x).chunk(2, dim=-1)
        gated = self.act(gate) * candidate # SwiGLU
        return self.output_proj(gated)

class FeedForwardLlamaQwen(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.candidate_proj = nn.Linear(embed_dim, hidden_dim)  # up_proj
        self.gate_proj = nn.Linear(embed_dim, hidden_dim)       # gate_proj
        self.output_proj = nn.Linear(hidden_dim, embed_dim)     # down_proj
        self.swish = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up = self.candidate_proj(x) # Candidate
        gate = self.swish(self.gate_proj(x))   # ← 激活在 gate 上
        return self.output_proj(gate * up)
