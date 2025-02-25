import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv, global_mean_pool

class HypergraphNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3):
        super(HypergraphNet, self).__init__()

        self.hypergraph_conv1 = HypergraphConv(in_channels, hidden_channels)
        self.hypergraph_conv2 = HypergraphConv(hidden_channels, hidden_channels)
        self.hypergraph_conv3 = HypergraphConv(hidden_channels, hidden_channels)

        self.batch_norm1 = nn.BatchNorm1d(hidden_channels)
        self.batch_norm2 = nn.BatchNorm1d(hidden_channels)
        self.batch_norm3 = nn.BatchNorm1d(hidden_channels)

        self.fc = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        x = self.hypergraph_conv1(x, edge_index)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.hypergraph_conv2(x, edge_index)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.hypergraph_conv3(x, edge_index)
        x = self.batch_norm3(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = global_mean_pool(x, batch)
        out = self.fc(x).squeeze(-1)

        return torch.sigmoid(out)  # Sigmoid per classificazione binaria