import numpy as np
from src.analysis.metrics import accuracy, flops, latency, memory

def run(model, data, repeats=5):
    results = []

    for _ in range(repeats):
        res = {
            "accuracy": accuracy(model, data),
            "flops": flops(model, data)[0],
            "memory": memory(model, data),
            "latency": latency(model, data),
        }
        results.append(res)

    def agg(key):
        vals = [r[key] for r in results]
        return np.mean(vals), np.std(vals)

    return {
        "accuracy": agg("accuracy"),
        "flops": agg("flops"),
        "memory": agg("memory"),
        "latency": agg("latency"),
    }