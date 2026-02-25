import os
import pandas as pd
import itertools
from scipy.stats import spearmanr, pearsonr

# -------------------------
# PATH
# -------------------------
BASE_DIR = "experiments/analysis"

files = [f for f in os.listdir(BASE_DIR) if f.endswith(".csv")]

if len(files) < 2:
    print("❌ Need at least 2 importance files")
    exit()

# -------------------------
# LOAD ALL FILES
# -------------------------
data = {}

for file in files:
    path = os.path.join(BASE_DIR, file)

    df = pd.read_csv(path)

    if "gene" not in df.columns or "importance" not in df.columns:
        print(f"⚠️ Skipping invalid file: {file}")
        continue

    df = df.sort_values("gene")  # ensure alignment
    data[file] = df

# -------------------------
# PAIRWISE COMPARISON
# -------------------------
results = []

pairs = list(itertools.combinations(data.keys(), 2))

for f1, f2 in pairs:
    df1 = data[f1]
    df2 = data[f2]

    imp1 = df1["importance"].values
    imp2 = df2["importance"].values

    # Spearman (ranking consistency)
    sp_corr, _ = spearmanr(imp1, imp2)

    # Pearson (magnitude consistency)
    pr_corr, _ = pearsonr(imp1, imp2)

    results.append({
        "run_1": f1,
        "run_2": f2,
        "spearman": round(sp_corr, 4),
        "pearson": round(pr_corr, 4)
    })

# -------------------------
# RESULTS TABLE
# -------------------------
results_df = pd.DataFrame(results)

print("\n" + "="*100)
print("🧬 GENE IMPORTANCE STABILITY")
print("="*100)
print(results_df)

# -------------------------
# SUMMARY
# -------------------------
mean_spearman = results_df["spearman"].mean()
mean_pearson = results_df["pearson"].mean()

print("\n" + "="*100)
print("📊 SUMMARY")
print("="*100)
print(f"Mean Spearman (ranking stability): {round(mean_spearman, 4)}")
print(f"Mean Pearson  (value similarity): {round(mean_pearson, 4)}")

# -------------------------
# SAVE
# -------------------------
save_path = os.path.join(BASE_DIR, "gene_stability_results.csv")
results_df.to_csv(save_path, index=False)

print(f"\n✅ Saved → {save_path}")