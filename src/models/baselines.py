import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    GCNConv,
    GATConv,
    SAGEConv,
    global_mean_pool
)

# =========================================================
# Shared Base
# =========================================================
class BaseGNN(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.dropout = dropout

    def apply_dropout(self, x):
        return F.dropout(x, p=self.dropout, training=self.training)


# =========================================================
# GCN
# =========================================================
class GCN(BaseGNN):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__(dropout)

        hidden_dim = int(hidden_dim)  # 🔥 FIX (avoid float bugs)

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.lin = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch, return_embeddings=False):
        x = self.conv1(x, edge_index)
        x = self.norm1(x.float())  # 🔥 FP16-safe
        x = F.relu(x)
        x = self.apply_dropout(x)

        x = self.conv2(x, edge_index)
        x = self.norm2(x.float())
        x = F.relu(x)
        x = self.apply_dropout(x)

        embeddings = x  # 🔥 Node embeddings BEFORE pooling

        x = global_mean_pool(x, batch)
        out = self.lin(x)

        if return_embeddings:
            return out, embeddings

        return out


# =========================================================
# GraphSAGE
# =========================================================
class GraphSAGE(BaseGNN):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__(dropout)

        hidden_dim = int(hidden_dim)

        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.lin = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch, return_embeddings=False):
        x = self.conv1(x, edge_index)
        x = self.norm1(x.float())
        x = F.relu(x)
        x = self.apply_dropout(x)

        x = self.conv2(x, edge_index)
        x = self.norm2(x.float())
        x = F.relu(x)
        x = self.apply_dropout(x)

        embeddings = x

        x = global_mean_pool(x, batch)
        out = self.lin(x)

        if return_embeddings:
            return out, embeddings

        return out


# =========================================================
# GAT (FIXED)
# =========================================================
class GAT(BaseGNN):
    def __init__(self, input_dim, hidden_dim, num_classes, heads=4, dropout=0.3):
        super().__init__(dropout)

        hidden_dim = int(hidden_dim)   # 🔥 CRITICAL FIX
        heads = int(heads)

        self.conv1 = GATConv(
            input_dim,
            hidden_dim,
            heads=heads,
            concat=True,
            dropout=dropout
        )

        self.conv2 = GATConv(
            hidden_dim * heads,
            hidden_dim,
            heads=1,
            concat=False,
            dropout=dropout
        )

        self.norm1 = nn.LayerNorm(hidden_dim * heads)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.lin = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch, return_embeddings=False):
        x = self.conv1(x, edge_index)
        x = self.norm1(x.float())  # 🔥 FP16-safe
        x = F.elu(x)
        x = self.apply_dropout(x)

        x = self.conv2(x, edge_index)
        x = self.norm2(x.float())
        x = F.elu(x)
        x = self.apply_dropout(x)

        embeddings = x

        x = global_mean_pool(x, batch)
        out = self.lin(x)

        if return_embeddings:
            return out, embeddings

        return out


# =========================================================
# Model Factory
# =========================================================
def get_model(
    model_name,
    input_dim,
    hidden_dim,
    num_classes,
    dropout=0.3
):
    model_name = model_name.lower()

    if model_name == "gcn":
        return GCN(input_dim, hidden_dim, num_classes, dropout)

    elif model_name == "sage":
        return GraphSAGE(input_dim, hidden_dim, num_classes, dropout)

    elif model_name == "gat":
        return GAT(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            heads=4,
            dropout=dropout
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")