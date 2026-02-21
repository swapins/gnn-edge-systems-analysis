import argparse
import yaml
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score
import time
import os
import platform

from src.models.gnn_model import GNN
from src.utils.seed import set_seed
from src.utils.logger import save_log
from src.profiling.memory import get_memory

# -------------------------
# CLI Arguments
# -------------------------
parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="tcga_sim",
                    choices=["base", "tcga_sim", "tcga_real"])

parser.add_argument("--hidden_dim", type=int, default=32)

parser.add_argument("--config", type=str, default="configs/jetson.yaml")

args = parser.parse_args()

# -------------------------
# Hardware Detection
# -------------------------
def detect_hardware():
    system = platform.system().lower()

    if "linux" in system or "windows" in system:
        if torch.cuda.is_available():
            return "desktop_gpu"
        else:
            return "desktop_cpu"

    return "unknown"


def get_config_type(config_path):
    name = config_path.lower()

    if "jetson" in name:
        return "jetson"
    elif "pi" in name:
        return "raspberry_pi"
    elif "desktop" in name:
        return "desktop"
    else:
        return "unknown"


hardware = detect_hardware()
config_type = get_config_type(args.config)

print("=" * 50)
print(f"Detected HW  : {hardware}")
print(f"Config Type  : {config_type}")
print("=" * 50)

# -------------------------
# Load Config (SAFE)
# -------------------------
with open(args.config, "r") as f:
    config = yaml.safe_load(f)

if not config:
    print("⚠️ Empty config → using defaults")
    config = {
        "device": "cpu",
        "training": {
            "epochs": 10,
            "lr": 0.001,
            "seed": 42,
            "batch_size": 4
        }
    }

training_cfg = config.get("training", {})

# -------------------------
# Seed
# -------------------------
set_seed(training_cfg.get("seed", 42))

# -------------------------
# Device Handling
# -------------------------
requested_device = config.get("device", "auto")

if requested_device == "cuda":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("⚠️ CUDA not available → fallback CPU")
        device = torch.device("cpu")
elif requested_device == "cpu":
    device = torch.device("cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Skip Invalid Hardware
# -------------------------
def save_skip(reason):
    filename = f"skipped_{args.dataset}_hd{args.hidden_dim}.json"
    save_log([{
        "status": "skipped",
        "reason": reason,
        "config": args.config,
        "dataset": args.dataset,
        "hidden_dim": args.hidden_dim
    }], os.path.join("logs", filename))


if config_type == "jetson" and "jetson" not in hardware:
    print("⛔ Skipping: Not Jetson hardware")
    save_skip("not_jetson_hardware")
    exit()

if config_type == "raspberry_pi" and "pi" not in hardware:
    print("⛔ Skipping: Not Raspberry Pi")
    save_skip("not_pi_hardware")
    exit()

# -------------------------
# Print Run Info
# -------------------------
print("=" * 50)
print(f"Device       : {device}")
print(f"Config       : {args.config}")
print(f"Dataset      : {args.dataset}")
print(f"Hidden Dim   : {args.hidden_dim}")
print("=" * 50)

# -------------------------
# Load Dataset
# -------------------------
batch_size = training_cfg.get("batch_size", 4)

if args.dataset == "base":
    from src.data.ppi_base_loader import load_ppi_base
    data_loader = load_ppi_base(batch_size=batch_size)

elif args.dataset == "tcga_sim":
    from src.data.ppi_tcga_sim_loader import load_ppi_tcga_sim
    data_loader = load_ppi_tcga_sim(batch_size=batch_size)

elif args.dataset == "tcga_real":
    from src.data.ppi_tcga_real_loader import load_ppi_tcga_real
    data_loader = load_ppi_tcga_real(batch_size=batch_size)

# -------------------------
# Model Setup
# -------------------------
sample_data = next(iter(data_loader))

model = GNN(
    input_dim=sample_data.num_node_features,
    hidden_dim=args.hidden_dim,
    num_classes=2
).to(device)

optimizer = Adam(model.parameters(), lr=training_cfg.get("lr", 0.001))

# -------------------------
# Training
# -------------------------
log_data = []

try:
    for epoch in range(training_cfg.get("epochs", 10)):
        model.train()

        total_loss = 0
        all_preds = []
        all_labels = []

        start_time = time.time()

        for data in data_loader:
            data = data.to(device)

            optimizer.zero_grad()

            out = model(data.x, data.edge_index, data.batch)

            loss = F.cross_entropy(out, data.y)
            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            preds = torch.argmax(out, dim=1).detach().cpu()
            labels = data.y.detach().cpu()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

        try:
            auc = roc_auc_score(all_labels, all_preds)
        except:
            auc = 0.5

        epoch_time = time.time() - start_time
        memory = get_memory(device)

        print(f"Epoch {epoch:02d} | Loss={total_loss:.4f} | AUC={auc:.4f} | Time={epoch_time:.2f}s")

        log_data.append({
            "epoch": epoch,
            "loss": float(total_loss),
            "roc_auc": float(auc),
            "time": float(epoch_time),
            "memory": int(memory),
            "dataset": args.dataset,
            "hidden_dim": args.hidden_dim,
            "device": str(device),
            "config": args.config,
            "status": "success"
        })

except Exception as e:
    print(f"❌ Experiment failed: {e}")

    log_data = [{
        "epoch": -1,
        "status": "failed",
        "error": str(e),
        "dataset": args.dataset,
        "hidden_dim": args.hidden_dim,
        "config": args.config
    }]

# -------------------------
# SAVE LOGS (MASTER + ORGANIZED)
# -------------------------
config_name = os.path.basename(args.config).replace(".yaml", "")
filename = f"{config_name}_{args.dataset}_hd{args.hidden_dim}.json"

# MASTER
os.makedirs("logs", exist_ok=True)
master_path = os.path.join("logs", filename)
save_log(log_data, master_path)

# ORGANIZED
if "desktop" in config_name or "jetson" in config_name or "pi" in config_name:
    exp_folder = "device_baseline"
elif "fp16" in config_name:
    exp_folder = "precision_study"
else:
    exp_folder = "scaling_study"

exp_dir = os.path.join("experiments", exp_folder, "results")
os.makedirs(exp_dir, exist_ok=True)

exp_path = os.path.join(exp_dir, filename)

# overwrite old
if os.path.exists(exp_path):
    os.remove(exp_path)

save_log(log_data, exp_path)

print("=" * 50)
print(f"✅ Saved → {master_path}")
print(f"📁 Organized → {exp_path}")
print("=" * 50)
print("Please run plots_results and compare_devices")