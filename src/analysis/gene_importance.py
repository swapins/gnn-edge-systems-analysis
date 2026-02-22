import torch
import pandas as pd
import os

def extract_gene_importance(model, gene_names=None, save_path="results/gene_importance.csv"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with torch.no_grad():
        weights = model.conv1.lin.weight.abs().mean(dim=0).cpu().numpy()

    if gene_names is None:
        gene_names = [f"Gene_{i}" for i in range(len(weights))]

    df = pd.DataFrame({
        "gene": gene_names,
        "importance": weights
    })

    df = df.sort_values(by="importance", ascending=False)

    df.to_csv(save_path, index=False)

    print(f"🧬 Gene importance saved → {save_path}")

    return df