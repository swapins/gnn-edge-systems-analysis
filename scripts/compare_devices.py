import os
import json
import csv
from collections import defaultdict
from statistics import mean, stdev

BASE_DIR = "experiments/device_baseline/results"
os.makedirs(BASE_DIR, exist_ok=True)

grouped = defaultdict(list)

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

        def score(e):
            return e.get("roc_auc", 0) - 0.0001 * e.get("memory", 0)

        best = max(valid, key=score)

        model = best.get("model", "unknown")

        config_path = best.get("config", "unknown")
        config = os.path.basename(config_path).replace(".yaml", "") if config_path != "unknown" else "unknown"

        hd = best.get("hidden_dim", 0)
        lm = best.get("lambda_memory", 0.0)
        lt = best.get("lambda_time", 0.0)

        grouped[(model, config, hd, lm, lt)].append({
            "auc": best.get("roc_auc", 0),
            "time": sum(e.get("time", 0) for e in valid),
            "memory": max(e.get("memory", 0) for e in valid) / (1024 * 1024)
        })

    except Exception as e:
        print(f"Error processing {file}: {e}")


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

if not results:
    print("No valid experiment results found")
    exit()

results = sorted(
    results,
    key=lambda x: (x["model"], x["hidden_dim"], x["lambda_memory"], x["config"])
)

precision_table = defaultdict(list)

for r in results:
    precision = "fp16" if "fp16" in r["config"].lower() else "fp32"
    precision_table[(r["model"], precision)].append(r["auc_mean"])


best_by_config = {}

for r in results:
    key = (r["config"], r["hidden_dim"], r["lambda_memory"])
    if key not in best_by_config or r["auc_mean"] > best_by_config[key]["auc_mean"]:
        best_by_config[key] = r


constraint_rows = [
    {
        "model": r["model"],
        "hidden_dim": r["hidden_dim"],
        "lambda_memory": r["lambda_memory"],
        "lambda_time": r["lambda_time"],
        "auc_mean": r["auc_mean"],
        "memory_mean": r["memory_mean"]
    }
    for r in results
]


with open(os.path.join(BASE_DIR, "table_main.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)


precision_rows = [
    {"model": m, "precision": p, "auc_mean": round(mean(v), 4)}
    for (m, p), v in precision_table.items()
]

with open(os.path.join(BASE_DIR, "table_precision.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["model", "precision", "auc_mean"])
    writer.writeheader()
    writer.writerows(precision_rows)


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


with open(os.path.join(BASE_DIR, "final_all_tables.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)


print("\nTables exported successfully")