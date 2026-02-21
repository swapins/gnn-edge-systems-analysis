import torch
import psutil
import os

def get_memory(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated()
    else:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss