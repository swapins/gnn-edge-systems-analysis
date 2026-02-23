from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

def load_proteins(batch_size=32):
    dataset = TUDataset(root="data/TUDataset", name="PROTEINS")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader