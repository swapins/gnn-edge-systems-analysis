import torch
import pandas as pd
import random
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def load_ppi_tcga_sim(batch_size=4):
    edges = pd.read_csv("data/ppi_edges.csv")

    proteins = list(set(edges["protein1"]).union(set(edges["protein2"])))
    protein_to_idx = {p: i for i, p in enumerate(proteins)}

    print("\n🧬 DEBUG LOADER:")
    print("Total proteins:", len(proteins))
    print("First 5 proteins:", proteins[:5])

    edge_list = [
        (protein_to_idx[p1], protein_to_idx[p2])
        for p1, p2 in zip(edges["protein1"], edges["protein2"])
    ]

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    num_nodes = len(proteins)

    dataset = []

    for i in range(200):

        x = np.random.normal(0, 1, (num_nodes, 1))
        label = random.randint(0, 1)

        if label == 1:
            x[:5] += 2.5

        x = torch.tensor(x, dtype=torch.float)

        y = torch.tensor([label], dtype=torch.long)
        batch = torch.zeros(num_nodes, dtype=torch.long)

        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            batch=batch
        )

        # 🔥 CRITICAL
        data.gene_names = proteins

        # DEBUG FIRST SAMPLE ONLY
        if i == 0:
            print("\n🧬 DEBUG SAMPLE:")
            print("Gene names attached:", hasattr(data, "gene_names"))
            print("First 5:", data.gene_names[:5])
            print("Count:", len(data.gene_names))

        dataset.append(data)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)