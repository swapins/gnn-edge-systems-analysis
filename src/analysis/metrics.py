import torch
import time
from thop import profile

def accuracy(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        return correct.item() / int(data.test_mask.sum())


def flops(model, data):
    model.eval()
    flops, params = profile(model, inputs=(data.x, data.edge_index), verbose=False)
    return flops, params


def latency(model, data, runs=50):
    model.eval()

    for _ in range(10):  # warmup
        _ = model(data.x, data.edge_index)

    start = time.time()
    for _ in range(runs):
        _ = model(data.x, data.edge_index)
    end = time.time()

    return (end - start) / runs * 1000


def memory(model, data):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        _ = model(data.x, data.edge_index)
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)