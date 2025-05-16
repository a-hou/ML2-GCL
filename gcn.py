import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch
import scipy.sparse as sp
import numpy as np

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, save_mem=True):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=not save_mem)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=not save_mem)
        self.dropout = dropout  # 将 dropout 参数赋值给实例变量

    def forward(self, x, adj):
        x = self.conv1(x, adj)
        #x = self.act(x)  # 使用 PReLU 激活函数
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, adj)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x 


