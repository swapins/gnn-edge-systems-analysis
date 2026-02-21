import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict

# -------------------------
# PATHS (UPDATED)
# -------------------------
BASE_EXP_DIR = "experiments"
SCALING_RESULTS_DIR = os.path.join(BASE_EXP_DIR, "scaling_study", "results")
PLOT_DIR = os.path.join(BASE_EXP_DIR, "scaling_study", "plots")

os.makedirs(PLOT_DIR, exist_ok=True)

# -------------------------
# Style (BMC / Bioinformatics)
# -------------------------
plt.rcParams.update({
    "font.size": 12,
    "figure.figsize": (6, 4),
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10
})

# -------------------------
# Load logs (FROM SCALING STUDY)
# -------------------------
results = defaultdict(dict)

if not os.path.exists(SCALING_RESULTS_DIR):
    print("❌ No scaling_study/results found")
    exit()

for file in os.listdir(SCALING_RESULTS_DIR):
    if not file.endswith(".json"):
        continue

    path = os.path.join(SCALING_RESULTS_DIR, file)

    try:
        with open(path, "r") as f:
            data = json.load(f)

        if not data:
            continue

        entry = data[-1]

        # -------------------------
        # Skip invalid logs
        # -------------------------
        if entry.get("status") in ["failed", "skipped"]:
            print(f"⛔ Skipping: {file} ({entry.get('status')})")
            continue

        dataset = entry.get("dataset")
        hidden_dim = entry.get("hidden_dim")

        if dataset is None or hidden_dim is None:
            print(f"⚠️ Missing fields in {file}")
            continue

        results[dataset][hidden_dim] = {
            "auc": entry.get("roc_auc", 0),
            "time": entry.get("time", 0),
            "memory": entry.get("memory", 0) / (1024 * 1024)
        }

    except Exception as e:
        print(f"❌ Error reading {file}: {e}")
        continue

# -------------------------
# Plot function
# -------------------------
def plot_metric(metric, ylabel, filename):
    plt.figure()

    for dataset, values in results.items():
        if not values:
            continue

        x = sorted(values.keys())
        y = [values[h][metric] for h in x]

        if len(x) == 0:
            continue

        plt.plot(x, y, marker='o', linewidth=2, label=dataset)

    if len(results) == 0:
        print(f"⚠️ No valid data for {metric}")
        return

    plt.xlabel("Hidden Dimension")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs Model Size")

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=False)

    plt.tight_layout()
    save_path = os.path.join(PLOT_DIR, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"📊 Saved: {save_path}")

# -------------------------
# Generate plots
# -------------------------
plot_metric("auc", "ROC-AUC", "fig_auc_vs_model.png")
plot_metric("time", "Training Time (s)", "fig_time_vs_model.png")
plot_metric("memory", "Memory Usage (MB)", "fig_memory_vs_model.png")

print("\n✅ Plots saved in experiments/scaling_study/plots/")