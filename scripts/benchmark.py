import torch
import json
import numpy as np

from src.models.baselines import get_model
from src.models.gnn_model import EdgeGNN

# LOADERS
from src.data.ppi_proteins_loader import load_proteins
from src.data.ppi_tcga_sim_loader import load_ppi_tcga_sim
from src.data.ppi_tcga_real_loader import load_ppi_tcga_real

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# TRAINING
# =========================================================

def train(model, loader, epochs=10):
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    print(f"\n🚀 Training {model.__class__.__name__}")

    for epoch in range(epochs):
        total_loss = 0
        total_samples = 0

        for data in loader:
            data = data.to(DEVICE)

            optimizer.zero_grad()

            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y.view(-1))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_samples += data.y.size(0)

        avg_loss = total_loss / total_samples
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.6f}")


def get_weight_norm(model):
    return sum(p.norm().item() for p in model.parameters())


# =========================================================
# METRICS
# =========================================================

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(DEVICE)

            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)

            correct += (pred == data.y.view(-1)).sum().item()
            total += data.y.size(0)

    return correct / total


def measure_latency(model, loader):
    import time
    model.eval()

    start = time.time()

    with torch.no_grad():
        for data in loader:
            data = data.to(DEVICE)
            _ = model(data.x, data.edge_index, data.batch)

    end = time.time()
    return (end - start) * 1000


def measure_memory(model, loader):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for data in loader:
            data = data.to(DEVICE)
            _ = model(data.x, data.edge_index, data.batch)

    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)


def estimate_flops(model, loader):
    from thop import profile

    data = next(iter(loader))
    data = data.to(DEVICE)

    flops, _ = profile(
        model,
        inputs=(data.x, data.edge_index, data.batch),
        verbose=False
    )

    return flops


# =========================================================
# RUN EXPERIMENT
# =========================================================

def run(model, loader):
    return {
        "accuracy": evaluate(model, loader),
        "latency": measure_latency(model, loader),
        "memory": measure_memory(model, loader),
        "flops": estimate_flops(model, loader)
    }


# =========================================================
# MAIN BENCHMARK
# =========================================================

def benchmark_dataset(name, loader):

    print(f"\n🚀 Running on {name}")

    sample = next(iter(loader))
    input_dim = sample.x.shape[1]
    # num_classes = int(sample.y.max().item()) + 1
    all_labels = []

    for data in loader:
        all_labels.append(data.y.view(-1))

    all_labels = torch.cat(all_labels)

    num_classes = int(all_labels.max().item()) + 1

    baseline = get_model("gcn", input_dim, 128, num_classes).to(DEVICE)
    edgegnn = EdgeGNN(input_dim, num_classes).to(DEVICE)

    # =========================
    # BEFORE TRAINING
    # =========================
    print("\n📊 Before Training:")
    print("Baseline Acc:", evaluate(baseline, loader))
    print("EdgeGNN Acc:", evaluate(edgegnn, loader))

    print("\n🔍 Weight Norm BEFORE:", get_weight_norm(baseline))

    # =========================
    # TRAIN
    # =========================
    print("\n🔧 Training Baseline...")
    train(baseline, loader, epochs=10)

    print("\n🔧 Training Edge-GNN...")
    train(edgegnn, loader, epochs=10)

    # =========================
    # AFTER TRAINING
    # =========================
    print("\n📊 After Training:")
    print("Baseline Acc:", evaluate(baseline, loader))
    print("EdgeGNN Acc:", evaluate(edgegnn, loader))

    print("\n🔍 Weight Norm AFTER:", get_weight_norm(baseline))

    # =========================
    # FINAL METRICS
    # =========================
    base_res = run(baseline, loader)
    edge_res = run(edgegnn, loader)

    acc_drop = ((base_res["accuracy"] - edge_res["accuracy"]) / base_res["accuracy"]) * 100
    flop_red = ((base_res["flops"] - edge_res["flops"]) / base_res["flops"]) * 100
    mem_red = ((base_res["memory"] - edge_res["memory"]) / base_res["memory"]) * 100

    print("\n📈 FINAL RESULTS")
    print(f"Accuracy Drop: {acc_drop:.2f}%")
    print(f"FLOPs Reduction: {flop_red:.2f}%")
    print(f"Memory Reduction: {mem_red:.2f}%")

    return {
        "baseline": base_res,
        "edgegnn": edge_res,
        "improvements": {
            "accuracy_drop_percent": acc_drop,
            "flops_reduction_percent": flop_red,
            "memory_reduction_percent": mem_red
        }
    }


# =========================================================
# ENTRY POINT
# =========================================================

def main():

    results = {}

    datasets = {
        "PROTEINS": load_proteins(),
        "TCGA_SIM": load_ppi_tcga_sim(),
        "TCGA_REAL": load_ppi_tcga_real()
    }

    for name, loader in datasets.items():
        try:
            results[name] = benchmark_dataset(name, loader)
        except Exception as e:
            print(f"❌ Failed on {name}: {e}")

    with open("results/benchmark_multi.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\n✅ All results saved to results/benchmark_multi.json")


if __name__ == "__main__":
    main()