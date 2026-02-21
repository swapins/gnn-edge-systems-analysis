from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

def load_data():
    dataset = TUDataset(root='data/TUDataset', name='PROTEINS')

    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    return loader