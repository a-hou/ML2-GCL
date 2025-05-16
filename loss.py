import torch
import torch.nn.functional as F

from weight import convert_to_weight_matrix

import torch.nn as nn

# class Projection(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(Projection, self).__init__()
#         # 定义线性变换的权重矩阵W和偏置项b
#         self.linear = nn.Linear(input_dim, output_dim)

#     def forward(self, X):
#         """
#         X: 输入特征矩阵，形状为 (N, D)，其中 N 为节点数，D 为特征维度
#         """
#         # 通过线性变换将输入特征矩阵投影到输出维度空间
#         return self.linear(X)


def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())



def contrastive_loss(z, neighbors, weights, device, tau=0.4):
    loss = torch.tensor(0.0, device=device, requires_grad=True)  # 创建具有梯度的张量
    #z = Projection(z)
    N = z.shape[0]

    # 确保所有的张量都在相同的设备上
    z = z.to(device)
    f = lambda x: torch.exp(x / tau)
    refl_sim = f(sim(z, z))

    weight_matrix = convert_to_weight_matrix(weights, neighbors, N, device)

    pos_refl_sim = refl_sim * weight_matrix
    #print(pos_refl_sim.sum(1))
    neg_weight_matrix = (weight_matrix == 0).float()
    # 计算每一行的和
    row_sums = torch.sum(neg_weight_matrix, dim=1)

    # 计算每个元素除以该行的和
    normalized_matrix = neg_weight_matrix / row_sums[:, None]
    neg_refl_sim = refl_sim * normalized_matrix
    
    epsilon = 1e-8
    # 添加epsilon以避免计算log(0)
    pos_refl_sim_sum = pos_refl_sim.sum(1) + epsilon
    neg_refl_sim_sum = neg_refl_sim.sum(1) + epsilon

    #return -torch.log(pos_refl_sim.sum(1) / (pos_refl_sim.sum(1) + neg_refl_sim.sum(1))).mean()
    return -torch.log(pos_refl_sim_sum / (pos_refl_sim_sum + neg_refl_sim_sum)).mean()