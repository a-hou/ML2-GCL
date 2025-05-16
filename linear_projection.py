import torch
import torch.nn as nn

class LinearProjection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearProjection, self).__init__()
        # 定义线性变换的权重矩阵W和偏置项b
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, X):
        """
        X: 输入特征矩阵，形状为 (N, D)，其中 N 为节点数，D 为特征维度
        """
        # 通过线性变换将输入特征矩阵投影到输出维度空间
        return self.linear(X)


