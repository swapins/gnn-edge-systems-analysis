import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict
from statistics import mean, stdev

# -------------------------
# PATHS
# -------------------------
BASE_DIR = "experiments/device_baseline/results"
PLOT_DIR = "experiments/scaling_study/plots"

os.makedirs(PLOT_DIR, exist_ok=True)

# -------------------------
# STYLE (Publication Ready)
# -------------------------
plt.rcParams.update({
    "font.size": 12,
    "figure.figsize": (6, 4),
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10
})

# -------------------------
# STORAGE
# -------------------------
data_grouped = defaultdict(lambda: defaultdict(list))
# structure: model → hidden_dim → list(values)

# -------------------------
# LOAD DATA
# -------------------------
for file in sorted(os.listdir(BASE_DIR)):
    if not file.endswith(".json"):
        continue

    path = os.path.join(BASE_DIR, file)

    try:
        with open(path) as f:
            data = json.load(f)

        if not isinstance(data, list):
            continue

        valid = [e for e in data if e.get("status") == "success"]
        if not valid:
            continue

        # 🔥 BEST EPOCH
        best = max(valid, key=lambda x: x.get("roc_auc", 0))

        if best.get("dataset") != "proteins":
            continue

        model = best["model"]
        hd = best["hidden_dim"]

        data_grouped[model][hd].append({
            "auc": best["roc_auc"],
            "time": best["time"],
            "memory": best["memory"] / (1024 * 1024)
        })

    except Exception as e:
        print(f"Error: {file} → {e}")

# -------------------------
# AGGREGATE
# -------------------------
aggregated = {}

for model, hd_dict in data_grouped.items():
    aggregated[model] = {}

    for hd, values in hd_dict.items():
        aucs = [v["auc"] for v in values]
        times = [v["time"] for v in values]
        mems = [v["memory"] for v in values]

        aggregated[model][hd] = {
            "auc_mean": mean(aucs),
            "auc_std": stdev(aucs) if len(aucs) > 1 else 0,
            "time_mean": mean(times),
            "memory_mean": mean(mems)
        }

# -------------------------
# PLOT FUNCTION
# -------------------------
def plot_metric(metric, ylabel, filename):
    plt.figure()

    for model, hd_dict in aggregated.items():
        if not hd_dict:
            continue

        x = sorted(hd_dict.keys())
        y = [hd_dict[h][f"{metric}_mean"] for h in x]
        yerr = [hd_dict[h].get(f"{metric}_std", 0) for h in x]

        plt.errorbar(x, y, yerr=yerr, marker='o', linewidth=2, capsize=3, label=model.upper())

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
# GENERATE FIGURES
# -------------------------
plot_metric("auc", "ROC-AUC", "fig_auc_vs_hidden_dim.png")
plot_metric("time", "Training Time (s)", "fig_time_vs_hidden_dim.png")
plot_metric("memory", "Memory Usage (MB)", "fig_memory_vs_hidden_dim.png")

print("\n✅ Publication-ready plots saved!")