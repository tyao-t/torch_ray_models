import torch

# 定义一个 3x4 的输入张量
input = torch.tensor([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 11, 12]])
print("Input tensor:\n", input)

# 定义一个新的索引张量
index_col = torch.tensor([[2, 0],
                          [1, 3],
                          [0, 2]])

# 沿着 dim=1 (列方向) 收集
# 对于输出中的每个位置 [i, j]，它的值 = input[i, index[i, j]]
output_col = torch.gather(input, dim=1, index=index_col)

print("\nIndex tensor:\n", index_col)
print("\nOutput tensor (gathered along dim=1):\n", output_col)
# 定义索引张量 index
# 我们希望输出是一个 2x3 的张量
# index = torch.tensor([[0, 1, 2],
#                       [2, 1, 0]])

# # 沿着 dim=0 (行方向) 收集
# # 对于输出中的每个位置 [i, j]，它的值 = input[index[i, j], j]
# output = torch.gather(input, dim=0, index=index)

# print("\nIndex tensor:\n", index)
# print("\nOutput tensor (gathered along dim=0):\n", output)