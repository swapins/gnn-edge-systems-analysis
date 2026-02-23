import os
import json
from collections import defaultdict
import csv
from statistics import mean, stdev

# -------------------------
# PATHS
# -------------------------
BASE_DIR = "experiments/device_baseline/results"
os.makedirs(BASE_DIR, exist_ok=True)

grouped = defaultdict(list)

# -------------------------
# LOAD DATA
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

        if best.get("dataset") != "proteins":
            continue

        model = best["model"]
        config = os.path.basename(best["config"]).replace(".yaml", "")
        hd = best["hidden_dim"]

        grouped[(model, config, hd)].append({
            "auc": best["roc_auc"],
            "time": best["time"],
            "memory": best["memory"] / (1024 * 1024)
        })

    except Exception as e:
        print(f"Error: {file} → {e}")

# -------------------------
# AGGREGATE
# -------------------------
results = []

for (model, config, hd), vals in grouped.items():
    aucs = [v["auc"] for v in vals]
    times = [v["time"] for v in vals]
    mems = [v["memory"] for v in vals]

    results.append({
        "model": model,
        "config": config,
        "hidden_dim": hd,
        "auc_mean": round(mean(aucs), 4),
        "auc_std": round(stdev(aucs), 4) if len(aucs) > 1 else 0,
        "time_mean": round(mean(times), 4),
        "memory_mean": round(mean(mems), 2),
        "runs": len(vals)
    })

# -------------------------
# SORT
# -------------------------
results = sorted(results, key=lambda x: (x["model"], x["config"], x["hidden_dim"]))

# =========================================================
# TABLE 1: MAIN BENCHMARK
# =========================================================
print("\n" + "="*100)
print("TABLE 1: MAIN BENCHMARK (MODEL COMPARISON)")
print("="*100)

for r in results:
    print(f"{r['model']:<6} | {r['config']:<15} | HD={r['hidden_dim']:<3} | "
          f"AUC={r['auc_mean']}±{r['auc_std']}")

# =========================================================
# TABLE 2: PRECISION (FP16 vs FP32)
# =========================================================
print("\n" + "="*100)
print("TABLE 2: PRECISION COMPARISON")
print("="*100)

precision_table = defaultdict(list)

for r in results:
    precision = "fp16" if "fp16" in r["config"] else "fp32"
    precision_table[(r["model"], precision)].append(r["auc_mean"])

for (model, prec), vals in precision_table.items():
    print(f"{model:<6} | {prec:<4} | AUC={round(mean(vals),4)}")

# =========================================================
# TABLE 3: SCALING (HIDDEN DIM)
# =========================================================
print("\n" + "="*100)
print("TABLE 3: SCALING ANALYSIS")
print("="*100)

for r in results:
    print(f"{r['model']:<6} | HD={r['hidden_dim']:<3} | AUC={r['auc_mean']}")

# =========================================================
# TABLE 4: EFFICIENCY
# =========================================================
print("\n" + "="*100)
print("TABLE 4: EFFICIENCY (AUC vs TIME vs MEMORY)")
print("="*100)

for r in results:
    print(f"{r['model']:<6} | HD={r['hidden_dim']:<3} | "
          f"AUC={r['auc_mean']} | Time={r['time_mean']} | Mem={r['memory_mean']}")

# =========================================================
# SAVE CSV
# =========================================================
csv_path = os.path.join(BASE_DIR, "final_all_tables.csv")

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"\n Saved → {csv_path}")