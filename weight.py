# import os

# import numpy as np
# import scipy.sparse as sp
# import torch
# import torch.nn as nn
# import process
# from model import CombinedModel
# import torch.optim as optim
# import random
# from utils import load_network_data
# from process import align_weights
# from loss import weighted_info_nce_loss
# from logreg import LogReg
# # 设置随机种子
# seed = 1  # 你可以选择任何整数作为种子

# # 设置 Python 内部的随机种子
# random.seed(seed)

# # 设置 numpy 的随机种子
# np.random.seed(seed)

# # 设置 PyTorch 的随机种子
# torch.manual_seed(seed)

# # 如果你使用 GPU (CUDA)，你也需要设置：
# torch.cuda.manual_seed(seed)

# A , X , Y = load_network_data('cora')
# c = Y.shape[1]
# lab = np.argmax(Y, 1)
# #对特征进行行归一化
# X, _ = process.preprocess_features(X)

# X = torch.FloatTensor(X[np.newaxis])
# X = X.squeeze(0)  # 去掉第一个维度

# device = torch.device(f"cuda:{3}") if torch.cuda.is_available() else torch.device("cpu")

# features = X.to(device)

# edge_index = process.convert_adj_to_edge_index(A).to(device)
# ft_size = features.shape[1]
# nb_nodes = features.shape[0]

# # 将稀疏矩阵转换为密集矩阵
# adj_dense = A.toarray()  # 转换为 NumPy 数组

# # 将密集矩阵转换为 PyTorch tensor
# Q_tensor = torch.tensor(adj_dense, dtype=torch.float32)

# features_tensor = features # 不需要梯度，避免重新创建 tensor

# features = features.to(device)
# edge_index = process.convert_adj_to_edge_index(A).to(device)

# # 权重初始化
# w = {}
# for i in range(Q_tensor.size(0)):
#     neighbors = torch.nonzero(Q_tensor[i]).view(-1)  # 找到节点 i 的邻居
#     # 直接创建 tensor，并且保持梯度
#     w[i] = torch.ones(len(neighbors), dtype=torch.float32, requires_grad=True, device=device)  
#     w[i] = w[i] / len(neighbors)  # 归一化，但不进行原地操作
#     w[i].retain_grad()  # 保持梯度

# # 只优化 w[i]，不优化 features_tensor
# optimizer = torch.optim.Adam([w[i] for i in w], lr=0.001)

# # # 目标函数计算：L = ∑i || xi - Σj∈Q(i) w_ij xj ||_2^2
# # def loss_function(features, Q_tensor, w):
# #     loss = 0
# #     for i in range(Q_tensor.size(0)):  # 遍历每个节点
# #         x_i = features[i]  # 获取节点 i 的特征
# #         neighbors = torch.nonzero(Q_tensor[i]).view(-1)  # 获取节点 i 的邻居节点
# #         weighted_sum = torch.sum(w[i].view(-1, 1) * features[neighbors], dim=0)  # 计算加权邻居特征和
# #         loss += torch.norm(x_i - weighted_sum, p=2) ** 2  # 计算 L2 范数的平方
# #     return loss

# # 目标函数计算：L = ∑i || xi - Σj∈Q(i) w_ij xj ||_2^2 + lambda * ||w||_2^2
# def loss_function(features, Q_tensor, w, lamda=0.001):
#     loss = 0
#     reg_loss = 0  # 初始化正则化损失

#     # 计算主损失项
#     for i in range(Q_tensor.size(0)):  # 遍历每个节点
#         x_i = features[i]  # 获取节点 i 的特征
#         neighbors = torch.nonzero(Q_tensor[i]).view(-1)  # 获取节点 i 的邻居节点
#         weighted_sum = torch.sum(w[i].view(-1, 1) * features[neighbors], dim=0)  # 计算加权邻居特征和
#         loss += torch.norm(x_i - weighted_sum, p=2) ** 2  # 计算 L2 范数的平方

#         # 计算正则化项 ||w[i]||_2^2
#         reg_loss += torch.sum(w[i] ** 2)  # 对每个节点的权重向量计算 L2 范数的平方

#     # 最终损失是主损失加上正则化项
#     total_loss = loss + lamda * reg_loss
#     return total_loss



# # 确保 features_tensor 和 Q_tensor 没有被提前分离计算图
# features_tensor = features_tensor.to(device)  # 确保 features_tensor 需要梯度
# Q_tensor = Q_tensor.to(device)  # 确保 Q_tensor 需要梯度

# # 训练过程
# num_epochs = 100
# for epoch in range(num_epochs):
#     optimizer.zero_grad()  # 清空梯度
#     loss = loss_function(features_tensor, Q_tensor, w)  # 计算损失
#     print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    
#     # 反向传播，确保计算图正确
#     loss.backward(retain_graph=True)  # 反向传播，计算梯度
#     optimizer.step()  # 更新参数

# # 对每个节点的权重进行归一化，并限制范围 [0, 1]
# for i in w:
#     # 归一化：确保每个 w[i] 的和为 1
#     w[i] = w[i] / torch.sum(w[i])  # 归一化权重，使其和为 1
    
#     # 限制每个权重的值在 [0, 1] 范围内
#     w[i] = torch.clamp(w[i], min=0.0, max=1.0)

# # 输出优化后的权重
# for i in w:
#     print(f"Optimized weights for node {i}: {w[i].detach().cpu().numpy()}")

# torch.save(w, 'weights_cora_lamda_0.001.pt')

import torch
import torch.nn.functional as F

def get_neighbors_and_weights(z, edge_index, device, epsilon=1e-5):
    N = z.size(0)
    neighbors = {}  # 存储每个节点的邻居
    weights = {}  # 存储每个节点的邻居权重

    for i in range(N):
        # 获取节点 i 的邻居
        neighbor_indices = edge_index[1][edge_index[0] == i].tolist()

        if not neighbor_indices:  # 如果没有邻居，跳过
            neighbors[i] = torch.tensor([], device=device)
            weights[i] = torch.tensor([], device=device)
            continue

        # 保存邻居索引
        neighbors[i] = torch.tensor(neighbor_indices, device=device)

        # 获取邻居的嵌入
        x_i = z[i]
        x_neighbors = z[neighbor_indices]

        # 使用 Gram 矩阵求解最优权重
        G = x_neighbors @ x_neighbors.T  # Gram 矩阵
        b = x_neighbors @ x_i.unsqueeze(-1)
        G = G.to(device)
        b = b.to(device)

        # 使用 Gram 矩阵求解最优权重
        eye_matrix = torch.eye(G.size(0), device=device)
        w = torch.linalg.solve(G + epsilon * eye_matrix, b)

        # 归一化权重
        w = F.relu(w).squeeze()
        if w.sum() > 0:
            w = w / w.sum()
        else:
            w = torch.ones(len(neighbor_indices), device=device) / len(neighbor_indices)

        weights[i] = w

    return neighbors, weights




def convert_to_weight_matrix(weights_dict, neighbors_dict, N, device):
    """
    将权重和邻居信息转换为加权邻接矩阵形式
    :param weights_dict: 每个节点的权重字典
    :param neighbors_dict: 每个节点的邻居字典
    :param N: 节点总数
    :param device: 设备信息
    :return: 权重矩阵 (N, N)
    """
    weight_matrix = torch.zeros((N, N), device=device)  # 初始化为零矩阵

    for i, neighbors in neighbors_dict.items():
        if len(neighbors) > 0:  # 仅处理有邻居的情况
            weights = weights_dict[i]
            weight_matrix[i, neighbors] = weights  # 将权重分配到对应的位置

    return weight_matrix