import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict
from statistics import mean, stdev

BASE_DIR = "experiments/device_baseline/results"
PLOT_DIR = "experiments/scaling_study/plots"

os.makedirs(PLOT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 12,
    "figure.figsize": (6,4),
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10
})

model_colors = {
    "gcn": "blue",
    "sage": "green",
    "gat": "red"
}

data_grouped = defaultdict(lambda: defaultdict(list))

# ---------------------------------------------------------
# LOAD RESULTS
# ---------------------------------------------------------
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

        best = max(valid, key=lambda x: x.get("roc_auc", 0))

        dataset = best.get("dataset", "")

        if dataset not in ["tcga_sim", "tcga_real"]:
            continue

        model = best.get("model")
        hd = best.get("hidden_dim")

        total_time = sum(e.get("time",0) for e in valid)
        max_mem = max(e.get("memory",0) for e in valid) / (1024*1024)

        data_grouped[model][hd].append({
            "auc": best.get("roc_auc",0),
            "time": total_time,
            "memory": max_mem
        })

    except Exception as e:
        print(f"Error processing {file}: {e}")

# ---------------------------------------------------------
# AGGREGATE
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# PLOT FUNCTION
# ---------------------------------------------------------
def plot_metric(metric, ylabel, filename):

    plt.figure()

    for model in sorted(aggregated.keys()):

        hd_dict = aggregated[model]

        if not hd_dict:
            continue

        x = sorted(hd_dict.keys())
        y = [hd_dict[h][f"{metric}_mean"] for h in x]
        yerr = [hd_dict[h].get(f"{metric}_std",0) for h in x]

        plt.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            linewidth=2,
            capsize=3,
            label=model.upper(),
            color=model_colors.get(model,"black")
        )

    plt.xlabel("Hidden Dimension")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs Model Size")

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=False)

    plt.tight_layout()

    save_path = os.path.join(PLOT_DIR, filename)

    plt.savefig(save_path, dpi=300)

    plt.close()

    print(f"Saved: {save_path}")

# ---------------------------------------------------------
# GENERATE FIGURES
# ---------------------------------------------------------
plot_metric("auc","ROC-AUC","fig_auc_vs_hidden_dim.png")
plot_metric("time","Training Time (s)","fig_time_vs_hidden_dim.png")
plot_metric("memory","Memory Usage (MB)","fig_memory_vs_hidden_dim.png")

print("\nScaling plots saved.")