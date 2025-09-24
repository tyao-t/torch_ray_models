import torch

# 创建两个 (1, 5) 的tensor
tensor1 = torch.tensor([[1, 2, 3, 4, 5]])
tensor2 = torch.tensor([[6, 7, 8, 9, 10]])

print("原始张量:")
print("tensor1:", tensor1, "shape:", tensor1.shape)
print("tensor2:", tensor2, "shape:", tensor2.shape)
print("-" * 40)

# dim=0 堆叠
stack_dim0 = torch.stack([tensor1, tensor2], dim=0)
print("1. torch.stack dim=0:")
print("结果形状:", stack_dim0.shape)
print("结果张量:")
print(stack_dim0)
print("-" * 40)

# dim=1 堆叠
stack_dim1 = torch.stack([tensor1, tensor2], dim=1)
print("2. torch.stack dim=1:")
print("结果形状:", stack_dim1.shape)
print("结果张量:")
print(stack_dim1)
print("-" * 40)

# dim=-1 堆叠 (最后一个维度)
stack_dim_neg1 = torch.stack([tensor1, tensor2], dim=-1)
print("3. torch.stack dim=-1:")
print("结果形状:", stack_dim_neg1.shape)
print("结果张量:")
print(stack_dim_neg1)

stack_dim_2 = torch.stack([tensor1, tensor2], dim=2)
print("3. torch.stack dim=2:")
print("结果形状:", stack_dim_2.shape)
print("结果张量:")
print(stack_dim_2)