import os
import pandas as pd
from itertools import combinations
from scipy.stats import spearmanr, pearsonr

BASE_DIR = "experiments/analysis"

files = [f for f in os.listdir(BASE_DIR) if f.endswith("_importance.csv")]

records = []

def load_df(path):
    df = pd.read_csv(path)

    # safety
    if "gene" not in df.columns or "importance" not in df.columns:
        return None

    return df

# -------------------------
# COMPARE ALL PAIRS (RELAXED)
# -------------------------
for f1, f2 in combinations(files, 2):

    try:
        df1 = load_df(os.path.join(BASE_DIR, f1))
        df2 = load_df(os.path.join(BASE_DIR, f2))

        if df1 is None or df2 is None:
            continue

        # merge on gene
        merged = df1.merge(df2, on="gene", suffixes=("_1", "_2"))

        if len(merged) < 2:
            continue

        s_corr = spearmanr(merged["importance_1"], merged["importance_2"]).correlation
        p_corr = pearsonr(merged["importance_1"], merged["importance_2"])[0]

        records.append({
            "file_1": f1,
            "file_2": f2,
            "spearman": round(s_corr, 4),
            "pearson": round(p_corr, 4),
            "overlap_genes": len(merged)
        })

    except Exception as e:
        print(f"⚠️ Skipped {f1} vs {f2}: {e}")

# -------------------------
# RESULTS
# -------------------------
if not records:
    print("❌ Still no valid comparisons")
    print("👉 Likely: gene mismatch OR empty files")
    exit()

df_out = pd.DataFrame(records)

print("\n🧬 BIOLOGICAL CONSISTENCY RESULTS")
print(df_out)

print("\n📊 SUMMARY:")
print(f"Mean Spearman: {df_out['spearman'].mean():.4f}")
print(f"Mean Pearson : {df_out['pearson'].mean():.4f}")

df_out.to_csv(os.path.join(BASE_DIR, "biological_consistency.csv"), index=False)

print("\n✅ Saved → biological_consistency.csv")