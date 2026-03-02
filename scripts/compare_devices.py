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

        # -------------------------
        # BEST EPOCH (Constraint-aware)
        # -------------------------
        def score(e):
            return e.get("roc_auc", 0) - 0.0001 * e.get("memory", 0)

        best = max(valid, key=score)

        # -------------------------
        # ONLY PROTEINS
        # -------------------------
        if best.get("dataset") != "proteins":
            continue

        # -------------------------
        # SAFE FIELD EXTRACTION
        # -------------------------
        model = best.get("model", "unknown")

        config_path = best.get("config", "unknown")
        if config_path != "unknown":
            config = os.path.basename(config_path).replace(".yaml", "")
        else:
            config = "unknown"

        # ✅ FIXED: always defined
        hd = best.get("hidden_dim", 0)

        # Claim 3 params
        lm = best.get("lambda_memory", 0.0)
        lt = best.get("lambda_time", 0.0)

        grouped[(model, config, hd, lm, lt)].append({
            "auc": best.get("roc_auc", 0),
            "time": sum(e.get("time", 0) for e in valid),
            "memory": max(e.get("memory", 0) for e in valid) / (1024 * 1024)
        })

    except Exception as e:
        print(f"❌ Error: {file} → {e}")

# -------------------------
# AGGREGATE
# -------------------------
results = []

for (model, config, hd, lm, lt), vals in grouped.items():
    aucs = [v["auc"] for v in vals]
    times = [v["time"] for v in vals]
    mems = [v["memory"] for v in vals]

    results.append({
        "model": model,
        "config": config,
        "hidden_dim": hd,
        "lambda_memory": lm,
        "lambda_time": lt,
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
results = sorted(
    results,
    key=lambda x: (x["model"], x["hidden_dim"], x["lambda_memory"], x["config"])
)

# =========================================================
# TABLE 1: MAIN BENCHMARK
# =========================================================
print("\n" + "="*110)
print("TABLE 1: MAIN BENCHMARK (MODEL COMPARISON)")
print("="*110)

for r in results:
    print(f"{r['model']:<6} | {r['config']:<15} | HD={r['hidden_dim']:<3} | "
          f"AUC={r['auc_mean']}±{r['auc_std']}")

# =========================================================
# TABLE 2: PRECISION
# =========================================================
print("\n" + "="*110)
print("TABLE 2: PRECISION COMPARISON")
print("="*110)

precision_table = defaultdict(list)

for r in results:
    precision = "fp16" if "fp16" in r["config"].lower() else "fp32"
    precision_table[(r["model"], precision)].append(r["auc_mean"])

for (model, prec), vals in precision_table.items():
    print(f"{model:<6} | {prec:<4} | AUC={round(mean(vals),4)}")

# =========================================================
# TABLE 3: SCALING
# =========================================================
print("\n" + "="*110)
print("TABLE 3: SCALING ANALYSIS")
print("="*110)

for r in results:
    print(f"{r['model']:<6} | {r['config']:<15} | HD={r['hidden_dim']:<3} | "
          f"AUC={r['auc_mean']}")

# =========================================================
# TABLE 4: EFFICIENCY
# =========================================================
print("\n" + "="*110)
print("TABLE 4: EFFICIENCY (AUC vs TIME vs MEMORY)")
print("="*110)

for r in results:
    print(f"{r['model']:<6} | {r['config']:<15} | HD={r['hidden_dim']:<3} | "
          f"AUC={r['auc_mean']} | Time={r['time_mean']} | Mem={r['memory_mean']}")

# =========================================================
# TABLE 5: BEST MODEL PER CONFIG
# =========================================================
print("\n" + "="*110)
print("TABLE 5: BEST MODEL PER CONFIG")
print("="*110)

best_by_config = {}

for r in results:
    key = (r["config"], r["hidden_dim"], r["lambda_memory"])
    if key not in best_by_config or r["auc_mean"] > best_by_config[key]["auc_mean"]:
        best_by_config[key] = r

for v in best_by_config.values():
    print(f"{v['config']:<15} | HD={v['hidden_dim']} | λ={v['lambda_memory']} | "
          f"BEST={v['model']} | AUC={v['auc_mean']}")

# =========================================================
# TABLE 6: CONSTRAINT-AWARE ANALYSIS (FINAL ONLY)
# =========================================================
print("\n" + "="*100)
print("TABLE 6: CONSTRAINT-AWARE ANALYSIS")
print("="*100)

constraint_rows = []

for r in results:
    row = {
        "model": r["model"],
        "hidden_dim": r["hidden_dim"],
        "lambda_memory": r["lambda_memory"],
        "lambda_time": r["lambda_time"],
        "auc_mean": r["auc_mean"],
        "memory_mean": r["memory_mean"]
    }

    constraint_rows.append(row)

    print(f"{row['model']:<6} | HD={row['hidden_dim']:<3} | "
          f"λ_mem={row['lambda_memory']} | λ_time={row['lambda_time']} | "
          f"AUC={row['auc_mean']} | Mem={row['memory_mean']}")

# =========================================================
# EXPORT CSVs
# =========================================================

# ---- Table 1 (master)
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
with open(os.path.join(BASE_DIR, "table_scaling.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["model", "config", "hidden_dim", "auc"])
    writer.writeheader()
    writer.writerows([
        {
            "model": r["model"],
            "config": r["config"],
            "hidden_dim": r["hidden_dim"],
            "auc": r["auc_mean"]
        }
        for r in results
    ])

# ---- Table 4
with open(os.path.join(BASE_DIR, "table_efficiency.csv"), "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["model", "config", "hidden_dim", "auc", "time", "memory"]
    )
    writer.writeheader()
    writer.writerows([
        {
            "model": r["model"],
            "config": r["config"],
            "hidden_dim": r["hidden_dim"],
            "auc": r["auc_mean"],
            "time": r["time_mean"],
            "memory": r["memory_mean"]
        }
        for r in results
    ])

# ---- Table 5
with open(os.path.join(BASE_DIR, "table_best.csv"), "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["config", "hidden_dim", "lambda_memory", "best_model", "auc"]
    )
    writer.writeheader()
    writer.writerows([
        {
            "config": v["config"],
            "hidden_dim": v["hidden_dim"],
            "lambda_memory": v["lambda_memory"],
            "best_model": v["model"],
            "auc": v["auc_mean"]
        }
        for v in best_by_config.values()
    ])

# ---- Table 6
with open(os.path.join(BASE_DIR, "table_constraints.csv"), "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "model",
            "hidden_dim",
            "lambda_memory",
            "lambda_time",
            "auc_mean",
            "memory_mean"
        ]
    )
    writer.writeheader()
    writer.writerows(constraint_rows)

# ---- MASTER CSV
with open(os.path.join(BASE_DIR, "final_all_tables.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("\n📁 All tables exported successfully (Claim 3 ready)")