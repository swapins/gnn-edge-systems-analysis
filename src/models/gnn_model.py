import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.3):
        super(GNN, self).__init__()

        hidden_dim = int(hidden_dim)  # 🔥 safety

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.lin = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, batch, return_embeddings=False):
        # ---- Layer 1 ----
        x = self.conv1(x, edge_index)
        x = self.norm1(x.float())  # 🔥 FP16 safe
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # ---- Layer 2 ----
        x = self.conv2(x, edge_index)
        x = self.norm2(x.float())
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 🔥 Save embeddings BEFORE pooling
        embeddings = x

        # ---- Pooling ----
        x = global_mean_pool(x, batch)

        # ---- Classifier ----
        out = self.lin(x)

        if return_embeddings:
            return out, embeddings

        return out