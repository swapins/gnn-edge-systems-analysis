import torch
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

def load_ppi_tcga_real(batch_size=4):
    # -------------------------
    # Load PPI edges
    # -------------------------
    edges = pd.read_csv("data/ppi_edges.csv")

    proteins = list(set(edges["protein1"]).union(set(edges["protein2"])))
    protein_to_idx = {p: i for i, p in enumerate(proteins)}

    edge_list = [
        (protein_to_idx[p1], protein_to_idx[p2])
        for p1, p2 in zip(edges["protein1"], edges["protein2"])
    ]

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # -------------------------
    # Load TCGA expression
    # -------------------------
    expr = pd.read_csv("data/tcga_expression.csv", index_col=0)

    dataset = []

    for sample_id in expr.columns:
        # Map genes → nodes
        x = []

        for gene in proteins:
            if gene in expr.index:
                x.append(expr.loc[gene, sample_id])
            else:
                x.append(0.0)

        x = torch.tensor(x, dtype=torch.float).unsqueeze(1)

        # Labels file
        labels = pd.read_csv("data/tcga_labels.csv", index_col=0)
        y = torch.tensor([labels.loc[sample_id, "label"]], dtype=torch.long)

        batch = torch.zeros(len(proteins), dtype=torch.long)

        dataset.append(Data(x=x, edge_index=edge_index, y=y, batch=batch))

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)