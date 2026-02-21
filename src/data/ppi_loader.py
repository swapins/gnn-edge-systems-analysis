import torch
import pandas as pd
import random
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

def load_ppi_dataset(batch_size=4):
    # -------------------------
    # Load PPI edges
    # -------------------------
    edges = pd.read_csv("data/ppi_edges.csv")

    # Unique proteins
    proteins = list(set(edges["protein1"]).union(set(edges["protein2"])))
    protein_to_idx = {p: i for i, p in enumerate(proteins)}

    # Full edge list
    full_edge_index = [
        (protein_to_idx[p1], protein_to_idx[p2])
        for p1, p2 in zip(edges["protein1"], edges["protein2"])
    ]

    # -------------------------
    # Create multiple graphs
    # -------------------------
    dataset = []
    num_graphs = 20  # number of samples

    for i in range(num_graphs):
        # Randomly sample edges (simulate variation)
        sampled_edges = random.sample(full_edge_index, max(3, len(full_edge_index)//2))

        edge_index = torch.tensor(sampled_edges, dtype=torch.long).t().contiguous()

        num_nodes = len(proteins)

        # Node features (random for now — later replace with gene expression)
        x = torch.randn((num_nodes, 16))

        # Assign random label (binary classification)
        y = torch.tensor([random.randint(0, 1)], dtype=torch.long)

        # Batch (single graph)
        batch = torch.zeros(num_nodes, dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, y=y, batch=batch)
        dataset.append(data)

    # -------------------------
    # DataLoader
    # -------------------------
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader