import pandas as pd
import numpy as np

genes = ["TP53", "BRCA1", "BRCA2", "EGFR", "PIK3CA", "AKT1", "MTOR", "MDM2"]

num_samples = 50

# Create expression data
data = {"gene": genes}

for i in range(num_samples):
    sample_name = f"sample_{i}"

    # base expression
    values = np.random.normal(1.5, 0.5, len(genes))

    # simulate cancer
    if i % 2 == 0:
        values[:3] += 1.5  # oncogenes activated

    data[sample_name] = values

df = pd.DataFrame(data)
df.to_csv("data/tcga_expression.csv", index=False)

# Labels
labels = []

for i in range(num_samples):
    labels.append({
        "sample": f"sample_{i}",
        "label": 1 if i % 2 == 0 else 0
    })

labels_df = pd.DataFrame(labels)
labels_df.to_csv("data/tcga_labels.csv", index=False)

print("TCGA-like dataset generated")