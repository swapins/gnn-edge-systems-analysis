import os
import json
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = "experiments/device_baseline/results"
OUTPUT_DIR = "experiments/analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

records = []

# -------------------------
# LOAD RESULTS
# -------------------------
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
            "auc": best.get("roc_auc", 0),
            "time": sum(e.get("time", 0) for e in valid),
            "memory": max(e.get("memory", 0) for e in valid) / (1024 * 1024),
            "lambda_memory": best.get("lambda_memory", 0.0),
            "lambda_time": best.get("lambda_time", 0.0),
        })

    except Exception as e:
        print(f"❌ Error processing {file}: {e}")

# -------------------------
# SAFETY CHECK
# -------------------------
if not records:
    print("❌ No valid data found")
    exit()

df = pd.DataFrame(records)

# -------------------------
# REMOVE DUPLICATES (CRITICAL)
# -------------------------
df = df.drop_duplicates(
    subset=["model", "hidden_dim", "config", "lambda_memory", "lambda_time"]
)

# -------------------------
# NORMALIZATION (IMPORTANT)
# -------------------------
df["time_norm"] = df["time"] / df["time"].max()
df["memory_norm"] = df["memory"] / df["memory"].max()

# -------------------------
# CONSTRAINT-AWARE SCORE (CLAIM 3 CORE)
# -------------------------
df["constraint_score"] = (
    df["auc"]
    - df["lambda_memory"] * df["memory_norm"]
    - df["lambda_time"] * df["time_norm"]
)

# -------------------------
# PARETO FUNCTION (NORMALIZED)
# -------------------------
def is_dominated(row, df):
    for _, other in df.iterrows():
        if (
            other["auc"] >= row["auc"] and
            other["time_norm"] <= row["time_norm"] and
            other["memory_norm"] <= row["memory_norm"] and
            (
                other["auc"] > row["auc"] or
                other["time_norm"] < row["time_norm"] or
                other["memory_norm"] < row["memory_norm"]
            )
        ):
            return True
    return False


df["pareto"] = df.apply(lambda row: not is_dominated(row, df), axis=1)
pareto_df = df[df["pareto"] == True]

# -------------------------
# BEST CONSTRAINT MODEL
# -------------------------
best_constraint = df.loc[df["constraint_score"].idxmax()]

print("\n🏆 BEST CONSTRAINT-AWARE MODEL:")
print(best_constraint)

# -------------------------
# SAVE CSV
# -------------------------
columns = [
    "model", "hidden_dim", "config",
    "auc", "time", "memory",
    "lambda_memory", "lambda_time",
    "constraint_score", "pareto"
]

df[columns].to_csv(os.path.join(OUTPUT_DIR, "pareto_all_points.csv"), index=False)
pareto_df[columns].to_csv(os.path.join(OUTPUT_DIR, "pareto_frontier.csv"), index=False)

print("\n📊 Pareto Frontier Points:")
print(pareto_df)

# -------------------------
# PLOT 1: TIME vs AUC
# -------------------------
plt.figure()

for model in df["model"].unique():
    subset = df[df["model"] == model]
    plt.scatter(subset["time"], subset["auc"], label=model)

plt.scatter(
    pareto_df["time"],
    pareto_df["auc"],
    s=100,
    marker="x",
    label="Pareto Frontier"
)

plt.xlabel("Training Time (s)")
plt.ylabel("AUC")
plt.title("Pareto Frontier (Accuracy vs Time)")
plt.legend()
plt.grid(True)

plt.savefig(os.path.join(OUTPUT_DIR, "pareto_time_vs_auc.png"), dpi=300)
plt.close()

# -------------------------
# PLOT 2: MEMORY vs AUC
# -------------------------
plt.figure()

for model in df["model"].unique():
    subset = df[df["model"] == model]
    plt.scatter(subset["memory"], subset["auc"], label=model)

plt.scatter(
    pareto_df["memory"],
    pareto_df["auc"],
    s=100,
    marker="x",
    label="Pareto Frontier"
)

plt.xlabel("Memory (MB)")
plt.ylabel("AUC")
plt.title("Pareto Frontier (Accuracy vs Memory)")
plt.legend()
plt.grid(True)

plt.savefig(os.path.join(OUTPUT_DIR, "pareto_memory_vs_auc.png"), dpi=300)
plt.close()

print("\n✅ Pareto analysis saved in experiments/analysis/")