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

        # ✅ Only successful runs
        valid = [e for e in data if e.get("status") == "success"]
        if not valid:
            continue

        # ✅ Best epoch based on AUC
        best = max(valid, key=lambda x: x.get("roc_auc", 0))

        # ✅ Only proteins dataset
        if best.get("dataset") != "proteins":
            continue

        model = best["model"]
        config = os.path.basename(best["config"]).replace(".yaml", "")
        hd = best["hidden_dim"]

        grouped[(model, config, hd)].append({
            "auc": best["roc_auc"],
            "time": sum(e.get("time", 0) for e in valid),  # total training time
            "memory": max(e.get("memory", 0) for e in valid) / (1024 * 1024)  # peak memory MB
        })

    except Exception as e:
        print(f"❌ Error: {file} → {e}")

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
        "auc_std": round(stdev(aucs), 4) if len(aucs) > 1 else 0.0,
        "time_mean": round(mean(times), 4),
        "memory_mean": round(mean(mems), 2),
        "runs": len(vals)
    })

# -------------------------
# SAFETY CHECK
# -------------------------
if not results:
    print("❌ No valid results found")
    exit()

# -------------------------
# SORT
# -------------------------
results = sorted(results, key=lambda x: (x["model"], x["hidden_dim"], x["config"]))

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
# TABLE 2: PRECISION
# =========================================================
print("\n" + "="*100)
print("TABLE 2: PRECISION COMPARISON")
print("="*100)

precision_table = defaultdict(list)

for r in results:
    precision = "fp16" if "fp16" in r["config"].lower() else "fp32"
    precision_table[(r["model"], precision)].append(r["auc_mean"])

for (model, prec), vals in precision_table.items():
    print(f"{model:<6} | {prec:<4} | AUC={round(mean(vals),4)}")

# =========================================================
# TABLE 3: SCALING
# =========================================================
print("\n" + "="*100)
print("TABLE 3: SCALING ANALYSIS")
print("="*100)

for r in results:
    print(f"{r['model']:<6} | {r['config']:<15} | HD={r['hidden_dim']:<3} | "
          f"AUC={r['auc_mean']}")

# =========================================================
# TABLE 4: EFFICIENCY
# =========================================================
print("\n" + "="*100)
print("TABLE 4: EFFICIENCY (AUC vs TIME vs MEMORY)")
print("="*100)

for r in results:
    print(f"{r['model']:<6} | {r['config']:<15} | HD={r['hidden_dim']:<3} | "
          f"AUC={r['auc_mean']} | Time={r['time_mean']} | Mem={r['memory_mean']}")

# =========================================================
# TABLE 5: BEST MODEL PER CONFIG
# =========================================================
print("\n" + "="*100)
print("TABLE 5: BEST MODEL PER CONFIG")
print("="*100)

best_by_config = {}

for r in results:
    key = (r["config"], r["hidden_dim"])
    if key not in best_by_config or r["auc_mean"] > best_by_config[key]["auc_mean"]:
        best_by_config[key] = r

for (_, _), v in best_by_config.items():
    print(f"{v['config']:<15} | HD={v['hidden_dim']} | BEST={v['model']} | AUC={v['auc_mean']}")

# =========================================================
# EXPORT CSVs
# =========================================================

# ---- Table 1
with open(os.path.join(BASE_DIR, "table_main.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

# ---- Table 2
precision_rows = [
    {"model": m, "precision": p, "auc_mean": round(mean(v), 4)}
    for (m, p), v in precision_table.items()
]

with open(os.path.join(BASE_DIR, "table_precision.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["model", "precision", "auc_mean"])
    writer.writeheader()
    writer.writerows(precision_rows)

# ---- Table 3
scaling_rows = [
    {"model": r["model"], "config": r["config"], "hidden_dim": r["hidden_dim"], "auc": r["auc_mean"]}
    for r in results
]

with open(os.path.join(BASE_DIR, "table_scaling.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["model", "config", "hidden_dim", "auc"])
    writer.writeheader()
    writer.writerows(scaling_rows)

# ---- Table 4
efficiency_rows = [
    {
        "model": r["model"],
        "config": r["config"],
        "hidden_dim": r["hidden_dim"],
        "auc": r["auc_mean"],
        "time": r["time_mean"],
        "memory": r["memory_mean"]
    }
    for r in results
]

with open(os.path.join(BASE_DIR, "table_efficiency.csv"), "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["model", "config", "hidden_dim", "auc", "time", "memory"]
    )
    writer.writeheader()
    writer.writerows(efficiency_rows)

# ---- Table 5
best_rows = [
    {
        "config": v["config"],
        "hidden_dim": v["hidden_dim"],
        "best_model": v["model"],
        "auc": v["auc_mean"]
    }
    for v in best_by_config.values()
]

with open(os.path.join(BASE_DIR, "table_best.csv"), "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["config", "hidden_dim", "best_model", "auc"]
    )
    writer.writeheader()
    writer.writerows(best_rows)

# ---- Master CSV
with open(os.path.join(BASE_DIR, "final_all_tables.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("\n📁 All tables exported successfully")