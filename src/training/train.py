import argparse
import yaml
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score
import time
import os

from src.models.baselines import get_model
from src.utils.seed import set_seed
from src.utils.logger import save_log
from src.profiling.memory import get_memory
from src.analysis.gene_importance import extract_gene_importance

# -------------------------
# CLI Arguments
# -------------------------
parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="proteins",
                    choices=["base", "tcga_sim", "tcga_real", "proteins"])
parser.add_argument("--hidden_dim", type=int, default=32)
parser.add_argument("--config", type=str, default="configs/v1/desktop_fp32.yaml")
parser.add_argument("--model", type=str, default="gcn",
                    choices=["gcn", "sage", "gat"])
parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()

# -------------------------
# Load Config
# -------------------------
with open(args.config, "r") as f:
    config = yaml.safe_load(f) or {}

training_cfg = config.get("training", {})
constraints_cfg = config.get("constraints", {})

lambda_memory = constraints_cfg.get("lambda_memory", 0.0)
lambda_time = constraints_cfg.get("lambda_time", 0.0)

config_name = os.path.basename(args.config).replace(".yaml", "")

print(f"⚙️ Config → {config_name}")
print(f"⚙️ Constraint Weights → λ_mem={lambda_memory}, λ_time={lambda_time}")

# -------------------------
# Seed
# -------------------------
set_seed(args.seed)

# -------------------------
# Device
# -------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available() and config.get("device") == "cuda"
    else "cpu"
)

# -------------------------
# AMP
# -------------------------
use_fp16 = config.get("precision", "fp32") == "fp16"
use_amp = use_fp16 and device.type == "cuda"

if use_amp:
    scaler = torch.cuda.amp.GradScaler()
    autocast = torch.cuda.amp.autocast
else:
    from contextlib import nullcontext
    scaler = None
    autocast = nullcontext

# -------------------------
# Dataset Loader
# -------------------------
if args.dataset == "proteins":
    from src.data.ppi_proteins_loader import load_proteins
    data_loader = load_proteins(batch_size=training_cfg.get("batch_size", 4))

elif args.dataset == "tcga_sim":
    from src.data.ppi_tcga_sim_loader import load_ppi_tcga_sim
    data_loader = load_ppi_tcga_sim(batch_size=training_cfg.get("batch_size", 4))

elif args.dataset == "tcga_real":
    from src.data.ppi_tcga_real_loader import load_ppi_tcga_real
    data_loader = load_ppi_tcga_real(batch_size=training_cfg.get("batch_size", 4))

dataset_list = list(data_loader)

print(f"\n📊 Dataset Loaded: {args.dataset}")
print(f"Total samples: {len(dataset_list)}")

# -------------------------
# Train / Val Split
# -------------------------
split_idx = int(0.8 * len(dataset_list))
train_data = dataset_list[:split_idx]
val_data = dataset_list[split_idx:]

# -------------------------
# Model
# -------------------------
sample_data = train_data[0]

model = get_model(
    args.model,
    input_dim=sample_data.num_node_features,
    hidden_dim=args.hidden_dim,
    num_classes=2
).to(device)

optimizer = Adam(model.parameters(), lr=training_cfg.get("lr", 0.001))

# -------------------------
# Constraint Proxies
# -------------------------
def compute_complexity_proxy(model):
    return sum(p.abs().mean() for p in model.parameters())

def compute_latency_proxy(model, data):
    num_nodes = data.x.size(0)
    hidden = sum(p.shape[0] for p in model.parameters() if len(p.shape) > 1)
    return torch.tensor((num_nodes * hidden) / 1e6, device=data.x.device)

# -------------------------
# Training
# -------------------------
log_data = []

for epoch in range(training_cfg.get("epochs", 10)):
    model.train()

    total_loss = 0
    start_epoch = time.time()

    for data in train_data:
        data = data.to(device)
        optimizer.zero_grad()

        with autocast():
            out = model(data.x, data.edge_index, data.batch)
            task_loss = F.cross_entropy(out, data.y)

            complexity = compute_complexity_proxy(model)
            latency = compute_latency_proxy(model, data)

            loss = task_loss + \
                   (lambda_memory * complexity) + \
                   (lambda_time * latency)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    # -------- VALIDATION --------
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for data in val_data:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            probs = torch.softmax(out, dim=1)[:, 1]

            all_preds.extend(probs.cpu().tolist())
            all_labels.extend(data.y.cpu().tolist())

    auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5

    epoch_time = time.time() - start_epoch
    memory = get_memory(device)

    print(f"[{args.model.upper()}] Epoch {epoch:02d} | AUC={auc:.4f} | Loss={total_loss:.4f}")

    log_data.append({
        "epoch": epoch,
        "model": args.model,
        "roc_auc": float(auc),
        "loss": float(total_loss),
        "time": float(epoch_time),
        "memory": int(memory),
        "config": args.config,
        "lambda_memory": lambda_memory,
        "lambda_time": lambda_time,
        "complexity": float(complexity.item()),
        "latency_proxy": float(latency.item()),
        "dataset": args.dataset,
        "hidden_dim": args.hidden_dim,
        "seed": args.seed,
        "status": "success"
    })

# -------------------------
# Gene Importance (FINAL CLEAN)
# -------------------------
importance_path = os.path.join(
    "experiments",
    "analysis",
    f"{args.model}_{config_name}_seed{args.seed}_importance.csv"
)

# extract SINGLE graph from batch
sample_batch = train_data[0]
sample = sample_batch.to_data_list()[0]

gene_names = getattr(sample, "gene_names", None)

sample = sample.to(device)
sample.gene_names = gene_names

extract_gene_importance(
    model=model,
    data=sample,
    gene_names=sample.gene_names,
    save_path=importance_path
)

# -------------------------
# Save Logs
# -------------------------
filename = f"{args.model}_{config_name}_{args.dataset}_hd{args.hidden_dim}_seed{args.seed}.json"

os.makedirs("experiments/device_baseline/results", exist_ok=True)

save_log(
    log_data,
    os.path.join("experiments/device_baseline/results", filename)
)

print("✅ Experiment complete")