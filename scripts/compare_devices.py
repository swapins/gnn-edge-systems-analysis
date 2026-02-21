import os
import json
from collections import defaultdict
import csv

# -------------------------
# PATHS (UPDATED)
# -------------------------
BASE_EXP_DIR = "experiments"
DEVICE_RESULTS_DIR = os.path.join(BASE_EXP_DIR, "device_baseline", "results")
OUTPUT_DIR = os.path.join(BASE_EXP_DIR, "device_baseline", "results")

os.makedirs(OUTPUT_DIR, exist_ok=True)

results = []

# -------------------------
# LOAD LOGS (DEVICE BASELINE)
# -------------------------
if not os.path.exists(DEVICE_RESULTS_DIR):
    print("❌ No device_baseline/results found")
    exit()

for file in os.listdir(DEVICE_RESULTS_DIR):
    if not file.endswith(".json"):
        continue

    path = os.path.join(DEVICE_RESULTS_DIR, file)

    try:
        with open(path) as f:
            data = json.load(f)

        if not data:
            continue

        entry = data[-1]

        # -------------------------
        # Skip invalid logs
        # -------------------------
        if entry.get("status") == "failed":
            print(f"{file} → ❌ FAILED")
            continue

        if entry.get("status") == "skipped":
            print(f"{file} → ⛔ SKIPPED")
            continue

        config = os.path.basename(entry["config"]).replace(".yaml", "")
        dataset = entry["dataset"]
        hd = entry["hidden_dim"]

        results.append({
            "config": config,
            "dataset": dataset,
            "hidden_dim": hd,
            "auc": round(entry["roc_auc"], 4),
            "time_sec": round(entry["time"], 4),
            "memory_mb": round(entry["memory"] / (1024 * 1024), 2)
        })

    except Exception as e:
        print(f"❌ Error reading {file}: {e}")
        continue

# -------------------------
# SORT RESULTS
# -------------------------
results = sorted(results, key=lambda x: (x["config"], x["hidden_dim"]))

# -------------------------
# PRINT TABLE (CLEAN)
# -------------------------
print("\n" + "="*80)
print("📊 DEVICE BASELINE COMPARISON")
print("="*80)

header = f"{'Config':<18} {'Dataset':<10} {'HD':<5} {'AUC':<8} {'Time(s)':<10} {'Mem(MB)':<10}"
print(header)
print("-"*80)

for r in results:
    print(f"{r['config']:<18} "
          f"{r['dataset']:<10} "
          f"{r['hidden_dim']:<5} "
          f"{r['auc']:<8} "
          f"{r['time_sec']:<10} "
          f"{r['memory_mb']:<10}")

print("="*80)

# -------------------------
# SAVE CSV (IN EXPERIMENT FOLDER)
# -------------------------
csv_path = os.path.join(OUTPUT_DIR, "device_comparison.csv")

with open(csv_path, "w", newline="") as csvfile:
    fieldnames = ["config", "dataset", "hidden_dim", "auc", "time_sec", "memory_mb"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"\n✅ CSV saved → {csv_path}")