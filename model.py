import torch
import torch.nn as nn
from gcn import GCN  


import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv  

class CombinedModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, nonlinearity='prelu',dropout=0.5):
        super(CombinedModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

        if nonlinearity == 'prelu':
            self.activation = nn.PReLU()
        elif nonlinearity == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Identity()  # 默认无激活函数

    def forward(self, x, edge_index):

        h = self.conv1(x, edge_index)
        h = self.activation(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        h = self.activation(h)

        return  h  

# class CombinedModel(nn.Module):
#     def __init__(self, in_channels: int, out_channels: int, base_model=GCNConv, k: int = 2, skip=False):
#         super(CombinedModel, self).__init__()
#         self.base_model = base_model

#         assert k >= 2
#         self.k = k
#         self.skip = skip
#         if not self.skip:
#             self.conv = [base_model(in_channels, 2 * out_channels).jittable()]
#             for _ in range(1, k - 1):
#                 self.conv.append(base_model(2 * out_channels, 2 * out_channels))
#             self.conv.append(base_model(2 * out_channels, out_channels))
#             self.conv = nn.ModuleList(self.conv)

#             self.activation = nn.PReLU()
#         else:
#             self.fc_skip = nn.Linear(in_channels, out_channels)
#             self.conv = [base_model(in_channels, out_channels)]
#             for _ in range(1, k):
#                 self.conv.append(base_model(out_channels, out_channels))
#             self.conv = nn.ModuleList(self.conv)

#             self.activation = nn.PReLU()

#     def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
#         if not self.skip:
#             for i in range(self.k):
#                 x = self.activation(self.conv[i](x, edge_index))
#             return x
#         else:
#             h = self.activation(self.conv[0](x, edge_index))
#             hs = [self.fc_skip(x), h]
#             for i in range(1, self.k):
#                 u = sum(hs)
#                 hs.append(self.activation(self.conv[i](u, edge_index)))
#             return hs[-1]
