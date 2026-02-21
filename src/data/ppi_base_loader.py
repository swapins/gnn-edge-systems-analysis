import torch
import pandas as pd
import random
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

def load_ppi_base(batch_size=4):
    edges = pd.read_csv("data/ppi_edges.csv")

    proteins = list(set(edges["protein1"]).union(set(edges["protein2"])))
    protein_to_idx = {p: i for i, p in enumerate(proteins)}

    edge_list = [
        (protein_to_idx[p1], protein_to_idx[p2])
        for p1, p2 in zip(edges["protein1"], edges["protein2"])
    ]

    num_nodes = len(proteins)
    dataset = []

    for _ in range(20):
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        x = torch.randn((num_nodes, 16))
        y = torch.tensor([random.randint(0, 1)], dtype=torch.long)
        batch = torch.zeros(num_nodes, dtype=torch.long)

        dataset.append(Data(x=x, edge_index=edge_index, y=y, batch=batch))

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)