# FOR TESTING ONLY
from torch_geometric.datasets import Planetoid

def load_data():
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    return dataset[0]