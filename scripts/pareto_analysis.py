import os
import json
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = "experiments/device_baseline/results"
OUTPUT_DIR = "experiments/analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

records = []

# ---------------------------------------------------------
# LOAD EXPERIMENT RESULTS
# ---------------------------------------------------------
for file in sorted(os.listdir(BASE_DIR)):

    if not file.endswith(".json"):
        continue

    try:
        with open(os.path.join(BASE_DIR, file)) as f:
            data = json.load(f)

        if not isinstance(data, list):
            continue

        valid = [e for e in data if e.get("status") == "success"]
        if not valid:
            continue

        best = max(valid, key=lambda x: x.get("roc_auc", 0))

        config_path = best.get("config", "unknown")
        config_name = os.path.basename(config_path).replace(".yaml", "")

        records.append({
            "model": best.get("model", "unknown"),
            "hidden_dim": best.get("hidden_dim", 0),
            "config": config_name,
            "dataset": best.get("dataset", "unknown"),
            "auc": best.get("roc_auc", 0),
            "time": sum(e.get("time", 0) for e in valid),
            "memory": max(e.get("memory", 0) for e in valid) / (1024 * 1024),
            "lambda_memory": best.get("lambda_memory", 0.0),
            "lambda_time": best.get("lambda_time", 0.0),
        })

    except Exception as e:
        print(f"Error processing {file}: {e}")


if not records:
    print("No valid experiment results found")
    exit()

df = pd.DataFrame(records)

# ---------------------------------------------------------
# OPTIONAL DATASET FILTER
# ---------------------------------------------------------
if "dataset" in df.columns:
    df = df[df["dataset"].isin(["tcga_real", "tcga_sim"])]

# ---------------------------------------------------------
# REMOVE DUPLICATES
# ---------------------------------------------------------
df = df.drop_duplicates(
    subset=["model", "hidden_dim", "config", "lambda_memory", "lambda_time"]
)

# ---------------------------------------------------------
# NORMALIZATION
# ---------------------------------------------------------
df["time_norm"] = df["time"] / df["time"].max()
df["memory_norm"] = df["memory"] / df["memory"].max()

# ---------------------------------------------------------
# CONSTRAINT SCORE
# ---------------------------------------------------------
df["constraint_score"] = (
    df["auc"]
    - df["lambda_memory"] * df["memory_norm"]
    - df["lambda_time"] * df["time_norm"]
)

# ---------------------------------------------------------
# PARETO FRONTIER DETECTION
# ---------------------------------------------------------
pareto_flags = []

for i, row in df.iterrows():

    dominated = (
        (df["auc"] >= row["auc"]) &
        (df["time_norm"] <= row["time_norm"]) &
        (df["memory_norm"] <= row["memory_norm"]) &
        (
            (df["auc"] > row["auc"]) |
            (df["time_norm"] < row["time_norm"]) |
            (df["memory_norm"] < row["memory_norm"])
        )
    ).any()

    pareto_flags.append(not dominated)

df["pareto"] = pareto_flags

pareto_df = df[df["pareto"]]

# ---------------------------------------------------------
# BEST CONSTRAINT MODEL
# ---------------------------------------------------------
best_constraint = df.loc[df["constraint_score"].idxmax()]

print("\nBEST CONSTRAINT-AWARE MODEL\n")
print(best_constraint)

# ---------------------------------------------------------
# EXPORT CSV FILES
# ---------------------------------------------------------
columns = [
    "model",
    "hidden_dim",
    "config",
    "auc",
    "time",
    "memory",
    "lambda_memory",
    "lambda_time",
    "constraint_score",
    "pareto"
]

df[columns].to_csv(
    os.path.join(OUTPUT_DIR, "pareto_all_points.csv"),
    index=False
)

pareto_df[columns].to_csv(
    os.path.join(OUTPUT_DIR, "pareto_frontier.csv"),
    index=False
)

# ---------------------------------------------------------
# MODEL COLORS
# ---------------------------------------------------------
colors = {
    "gcn": "blue",
    "sage": "green",
    "gat": "red"
}

# ---------------------------------------------------------
# PLOT 1: TIME vs AUC
# ---------------------------------------------------------
plt.figure(figsize=(8,6))

for model in df["model"].unique():

    subset = df[df["model"] == model]

    plt.scatter(
        subset["time"],
        subset["auc"],
        label=model.upper(),
        color=colors.get(model, "black"),
        alpha=0.7
    )

plt.scatter(
    pareto_df["time"],
    pareto_df["auc"],
    s=120,
    marker="x",
    color="black",
    label="Pareto Frontier"
)

pareto_sorted = pareto_df.sort_values("time")

plt.plot(
    pareto_sorted["time"],
    pareto_sorted["auc"],
    linestyle="--",
    color="black"
)

plt.xlabel("Training Time (s)")
plt.ylabel("ROC-AUC")
plt.title("Pareto Frontier (Accuracy vs Time)")
plt.legend()
plt.grid(True)

plt.savefig(
    os.path.join(OUTPUT_DIR, "pareto_time_vs_auc.png"),
    dpi=300
)

plt.close()

# ---------------------------------------------------------
# PLOT 2: MEMORY vs AUC
# ---------------------------------------------------------
plt.figure(figsize=(8,6))

for model in df["model"].unique():

    subset = df[df["model"] == model]

    plt.scatter(
        subset["memory"],
        subset["auc"],
        label=model.upper(),
        color=colors.get(model, "black"),
        alpha=0.7
    )

plt.scatter(
    pareto_df["memory"],
    pareto_df["auc"],
    s=120,
    marker="x",
    color="black",
    label="Pareto Frontier"
)

pareto_sorted = pareto_df.sort_values("memory")

plt.plot(
    pareto_sorted["memory"],
    pareto_sorted["auc"],
    linestyle="--",
    color="black"
)

plt.xlabel("Memory (MB)")
plt.ylabel("ROC-AUC")
plt.title("Pareto Frontier (Accuracy vs Memory)")
plt.legend()
plt.grid(True)

plt.savefig(
    os.path.join(OUTPUT_DIR, "pareto_memory_vs_auc.png"),
    dpi=300
)

plt.close()

# ---------------------------------------------------------
# PLOT 3: CONSTRAINT SCORE
# ---------------------------------------------------------
plt.figure(figsize=(8,6))

for model in df["model"].unique():

    subset = df[df["model"] == model]

    plt.scatter(
        subset["memory"],
        subset["constraint_score"],
        label=model.upper(),
        color=colors.get(model, "black"),
        alpha=0.7
    )

plt.xlabel("Memory (MB)")
plt.ylabel("Constraint Score")
plt.title("Constraint-Aware Model Ranking")
plt.legend()
plt.grid(True)

plt.savefig(
    os.path.join(OUTPUT_DIR, "constraint_score_plot.png"),
    dpi=300
)

plt.close()

print("\nPareto analysis saved in experiments/analysis/")