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
            self.activation = nn.Identity() 

    def forward(self, x, edge_index):

        h = self.conv1(x, edge_index)
        h = self.activation(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        h = self.activation(h)

        return  h  

