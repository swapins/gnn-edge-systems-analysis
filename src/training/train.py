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
from src.orchestration.device_manager import DeviceManager
from src.orchestration.experiment_registry import create_experiment_metadata
from src.profiling.constraints import enforce_constraints

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
    if torch.cuda.is_available():
        return "desktop_gpu"
    return "desktop_cpu"


def get_config_type(config_path):
    name = config_path.lower()

    if "jetson" in name:
        return "jetson"
    elif "pi" in name:
        return "raspberry_pi"
    elif "desktop" in name:
        return "desktop"
    return "unknown"


hardware = detect_hardware()
config_type = get_config_type(args.config)

print("=" * 50)
print(f"Detected HW  : {hardware}")
print(f"Config Type  : {config_type}")
print("=" * 50)

# -------------------------
# Device Manager + Metadata
# -------------------------
device_manager = DeviceManager()
hardware_info = device_manager.summary()

experiment_meta = create_experiment_metadata(
    args.config,
    args.dataset,
    args.hidden_dim,
    hardware_info
)

print(f"🧪 Experiment ID: {experiment_meta['experiment_id']}")

# -------------------------
# Load Config
# -------------------------
with open(args.config, "r") as f:
    config = yaml.safe_load(f) or {}

training_cfg = config.get("training", {})

# -------------------------
# Seed
# -------------------------
set_seed(training_cfg.get("seed", 42))

# -------------------------
# Device Handling
# -------------------------
requested_device = config.get("device", "auto")

if requested_device == "cuda" and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    if requested_device == "cuda":
        print("⚠️ CUDA not available → fallback CPU")
    device = torch.device("cpu")

# -------------------------
# Ensure logs dir exists early
# -------------------------
os.makedirs("logs", exist_ok=True)

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
# Hardware-aware scaling
# -------------------------
def get_available_memory(device):
    try:
        if device.type == "cuda":
            return torch.cuda.get_device_properties(0).total_memory
        else:
            import psutil
            return psutil.virtual_memory().available
    except:
        return 512 * 1024 * 1024


def adapt_hidden_dim(requested_hd, device):
    memory = get_available_memory(device)

    if memory < 512 * 1024 * 1024:
        return min(requested_hd, 16)
    elif memory < 1 * 1024 * 1024 * 1024:
        return min(requested_hd, 32)
    elif memory < 2 * 1024 * 1024 * 1024:
        return min(requested_hd, 64)
    return requested_hd


original_hd = args.hidden_dim
args.hidden_dim = adapt_hidden_dim(original_hd, device)

if args.hidden_dim != original_hd:
    print(f"⚙️ Adjusted hidden_dim: {original_hd} → {args.hidden_dim}")

# -------------------------
# Model
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
        all_preds, all_labels = [], []

        start_time = time.time()

        for data in data_loader:
            data = data.to(device)

            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)

            loss = F.cross_entropy(out, data.y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            all_preds.extend(torch.argmax(out, dim=1).cpu().tolist())
            all_labels.extend(data.y.cpu().tolist())

        auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5

        epoch_time = time.time() - start_time
        memory = get_memory(device)

        # enforce constraints
        enforce_constraints(memory, config.get("max_memory"))

        print(f"Epoch {epoch:02d} | Loss={total_loss:.4f} | AUC={auc:.4f} | Time={epoch_time:.2f}s")

        log_data.append({
            "epoch": epoch,
            "loss": float(total_loss),
            "roc_auc": float(auc),
            "time": float(epoch_time),
            "memory": int(memory),
            "dataset": args.dataset,
            "hidden_dim": args.hidden_dim,
            "original_hidden_dim": original_hd,
            "adapted_hidden_dim": args.hidden_dim,
            "device": str(device),
            "config": args.config,
            "experiment_id": experiment_meta["experiment_id"],
            "hardware": hardware_info,
            "config_version": config.get("version", "v1"),
            "status": "success"
        })

except Exception as e:
    print(f"❌ Experiment failed: {e}")

    log_data = [{
        "status": "failed",
        "error": str(e),
        "dataset": args.dataset,
        "hidden_dim": args.hidden_dim,
        "experiment_id": experiment_meta["experiment_id"]
    }]

# -------------------------
# SAVE LOGS
# -------------------------
config_name = os.path.basename(args.config).replace(".yaml", "")
filename = f"{config_name}_{args.dataset}_hd{args.hidden_dim}.json"

master_path = os.path.join("logs", filename)
save_log(log_data, master_path)

# organized
exp_folder = "device_baseline" if "desktop" in config_name else "scaling_study"
exp_dir = os.path.join("experiments", exp_folder, "results")
os.makedirs(exp_dir, exist_ok=True)

exp_path = os.path.join(exp_dir, filename)

if os.path.exists(exp_path):
    os.remove(exp_path)

save_log(log_data, exp_path)

print("=" * 50)
print(f"✅ Saved → {master_path}")
print(f"📁 Organized → {exp_path}")
print("=" * 50)

# -------------------------
# Gene Importance
# -------------------------
try:
    from src.analysis.gene_importance import extract_gene_importance

    os.makedirs("experiments/analysis", exist_ok=True)

    extract_gene_importance(
        model,
        save_path=f"experiments/analysis/gene_importance_{config_name}.csv"
    )
except Exception as e:
    print(f"⚠️ Gene importance failed: {e}")