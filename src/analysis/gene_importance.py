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

    - Uses input sensitivity (∂output / ∂input)
    - Data-dependent
    - Supports Claim 2 (biological consistency)
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    model.eval()

    # -------------------------
    # Enable gradients on input
    # -------------------------
    x = data.x.clone().detach().requires_grad_(True)

    out = model(x, data.edge_index, data.batch)

    # Focus on positive class (oncology signal)
    probs = torch.softmax(out, dim=1)[:, 1]

    # Aggregate signal
    score = probs.mean()

    # Backprop
    score.backward()

    # -------------------------
    # Importance = gradient magnitude
    # -------------------------
    importance = x.grad.abs().mean(dim=0).cpu().numpy()

    # -------------------------
    # Gene names
    # -------------------------
    if gene_names is None:
        gene_names = [f"Gene_{i}" for i in range(len(importance))]

    df = pd.DataFrame({
        "gene": gene_names,
        "importance": importance
    })

    df = df.sort_values(by="importance", ascending=False)

    df.to_csv(save_path, index=False)

    print(f"🧬 Gene importance saved → {save_path}")

    return df