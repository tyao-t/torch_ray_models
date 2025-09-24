import math

class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # similar to standard weight initialization
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x

class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)

def replace_linear_with_lora(model, rank, alpha):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            # Replace the Linear layer with LinearWithLoRA
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            # Recursively apply the same function to child modules
            replace_linear_with_lora(module, rank, alpha)
    
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters before: {total_params:,}")
# Total trainable parameters before: 124,441,346

for param in model.parameters():
    param.requires_grad = False

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters after: {total_params:,}")
# Total trainable parameters after: 0

replace_linear_with_lora(model, rank=16, alpha=16)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable LoRA parameters: {total_params:,}")
# Total trainable LoRA parameters: 2,666,528

# Also, since we initialized matrix with 0's, we expect the initial model performance to be unchanged compared to before

import torch
import torch.nn as nn
import torch.nn.functional as F

class AdapterLayer(nn.Module):
    """
    一个经典的瓶颈结构 Adapter 模块。
    顺序： -> DownProject -> Non-Linearity -> UpProject ->
    """
    def __init__(self, in_features, adapter_size, init_scale=1e-3):
        super().__init__()
        self.adapter_size = adapter_size
        
        # 下投影矩阵 (in_features -> adapter_size)
        self.down_proj = nn.Linear(in_features, adapter_size)
        # 上投影矩阵 (adapter_size -> in_features)，初始化为接近0，保证开始时Adapter是恒等映射
        self.up_proj = nn.Linear(adapter_size, in_features)

        # 非线性激活函数，通常使用GELU或ReLU
        self.activation = nn.GELU()

        # 初始化技巧：将上投影权重初始化为（近似）0，保证训练开始时Adapter的输出接近0，不影响原模型。
        with torch.no_grad():
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.up_proj.bias)
            # 下投影可以使用常规初始化
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))

    def forward(self, x):
        # 原始输入 x: [batch_size, ..., in_features]
        # 注意：这里没有skip connection，它将在外面的包装层实现
        z = self.down_proj(x)
        z = self.activation(z)
        z = self.up_proj(z)
        return z


class LinearWithAdapter(nn.Module):
    """
    用Adapter包装一个Linear层。
    遵循Pfeiffer风格：output = original_layer(x) + adapter(original_layer(x))
    注意：这里Adapter是在原层**之后**添加的，是一种串行结构。
    """
    def __init__(self, linear_layer, adapter_size):
        super().__init__()
        self.linear_layer = linear_layer
        # Adapter的输入和输出维度都与原Linear层的输出维度相同
        self.adapter = AdapterLayer(linear_layer.out_features, adapter_size)

    def forward(self, x):
        # 1. 先经过原来的线性层
        original_output = self.linear_layer(x)
        # 2. 将原线性层的输出作为Adapter的输入
        adapter_output = self.adapter(original_output)
        # 3. 将原输出和Adapter输出相加（这就是Adapter的skip connection）
        return original_output + adapter_output