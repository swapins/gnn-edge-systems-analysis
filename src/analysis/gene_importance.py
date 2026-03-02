import torch
import pandas as pd
import os


def extract_gene_importance(
    model,
    data,
    gene_names=None,
    save_path="experiments/analysis/gene_importance.csv"
):
    """
    Gradient-based gene importance (biologically meaningful)

    ✔ Uses input gradients
    ✔ Supports real gene names (Claim 4)
    ✔ Fully robust
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    model.eval()

    # -------------------------
    # Enable gradients on input
    # -------------------------
    x = data.x.clone().detach().requires_grad_(True)

    out = model(x, data.edge_index, data.batch)

    probs = torch.softmax(out, dim=1)[:, 1]
    score = probs.mean()

    score.backward()

    # importance = x.grad.abs().mean(dim=0).cpu().numpy()
    # -------------------------
    # Importance = gradient magnitude PER GENE
    # -------------------------
    importance = x.grad.abs().mean(dim=1).cpu().numpy()


    # =========================================================
    # 🔥 CRITICAL FIX: USE REAL GENE NAMES
    # =========================================================
    if gene_names is None:
        if hasattr(data, "gene_names"):
            gene_names = data.gene_names
        else:
            gene_names = [f"Gene_{i}" for i in range(len(importance))]

    # Safety check
    if len(gene_names) != len(importance):
        print("⚠️ Gene mismatch → fallback to default names")
        gene_names = [f"Gene_{i}" for i in range(len(importance))]

    # -------------------------
    # Save
    # -------------------------
    df = pd.DataFrame({
        "gene": gene_names,
        "importance": importance
    })

    df = df.sort_values(by="importance", ascending=False)

    df.to_csv(save_path, index=False)

    print(f"🧬 Gene importance saved → {save_path}")

    return df