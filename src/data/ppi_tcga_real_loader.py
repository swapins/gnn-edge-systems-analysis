import os
import gzip
import requests
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

STRING_URL = "https://stringdb-static.org/download/protein.links.v11.5/9606.protein.links.v11.5.txt.gz"
STRING_PATH = os.path.join(DATA_DIR, "string_ppi.txt.gz")


# =========================================================
# DOWNLOAD STRING (ROBUST)
# =========================================================
def download_string():
    if os.path.exists(STRING_PATH):
        print("✅ STRING already exists")
        return

    print("⬇️ Downloading STRING PPI...")
    r = requests.get(STRING_URL, stream=True)

    with open(STRING_PATH, "wb") as f:
        for chunk in tqdm(r.iter_content(1024 * 1024)):
            if chunk:
                f.write(chunk)

    print("✅ STRING downloaded")


# =========================================================
# LOAD STRING PPI
# =========================================================
def load_string_edges(score_threshold=700):
    print("🧬 Loading STRING PPI...")

    edges = pd.read_csv(STRING_PATH, sep=" ")

    # Filter high-confidence edges
    edges = edges[edges["combined_score"] >= score_threshold]

    # Convert STRING IDs → gene symbols
    edges["protein1"] = edges["protein1"].str.split(".").str[1]
    edges["protein2"] = edges["protein2"].str.split(".").str[1]

    print(f"🔗 STRING edges after filter: {len(edges)}")

    return edges


# =========================================================
# MAIN LOADER
# =========================================================
def load_ppi_tcga_real(batch_size=1, max_nodes=500):
    print("\n🚀 Loading TCGA REAL dataset (STRING-powered)...")

    # -------------------------
    # Ensure STRING
    # -------------------------
    download_string()
    edges = load_string_edges()

    # -------------------------
    # Load TCGA
    # -------------------------
    expr = pd.read_csv("data/tcga_expression.csv", index_col=0)
    labels = pd.read_csv("data/tcga_labels.csv", index_col=0)

    print(f"📊 Expression: {expr.shape}")
    print(f"🧬 Labels: {labels.shape}")

    # -------------------------
    # Normalize
    # -------------------------
    expr = expr.apply(pd.to_numeric, errors="coerce").fillna(0)

    mean = expr.mean(axis=1)
    std = expr.std(axis=1).replace(0, 1)
    expr = expr.sub(mean, axis=0).div(std, axis=0)

    # -------------------------
    # Gene alignment
    # -------------------------
    tcga_genes = set(expr.index)
    ppi_genes = set(edges["protein1"]).union(set(edges["protein2"]))

    overlap = list(tcga_genes.intersection(ppi_genes))

    print(f"🧬 TCGA genes: {len(tcga_genes)}")
    print(f"🔗 STRING genes: {len(ppi_genes)}")
    print(f"✅ Overlap: {len(overlap)}")

    # =========================================================
    # GRAPH BUILDING
    # =========================================================
    if len(overlap) < 100:
        print("⚠️ Low overlap → fallback to kNN graph")

        proteins = list(expr.index[:max_nodes])
        gene_matrix = torch.tensor(expr.loc[proteins].values, dtype=torch.float)

        gene_matrix = F.normalize(gene_matrix, dim=1)
        similarity = torch.mm(gene_matrix, gene_matrix.t())

        k = 10
        edge_list = []

        for i in range(similarity.shape[0]):
            _, idx = torch.topk(similarity[i], k + 1)
            for j in idx[1:]:
                edge_list.append((i, j.item()))

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    else:
        print("✅ Using STRING biological graph")

        proteins = overlap[:max_nodes]
        protein_to_idx = {p: i for i, p in enumerate(proteins)}

        edge_list = []
        edge_weight = []

        for _, row in edges.iterrows():
            p1, p2 = row["protein1"], row["protein2"]

            if p1 in protein_to_idx and p2 in protein_to_idx:
                edge_list.append((protein_to_idx[p1], protein_to_idx[p2]))
                edge_weight.append(row["combined_score"] / 1000.0)

        if len(edge_list) == 0:
            raise ValueError("❌ No valid STRING edges after filtering")

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    print(f"🔗 Edges: {edge_index.shape[1]}")
    print(f"🧬 Nodes: {len(proteins)}")

    # =========================================================
    # DATASET BUILD
    # =========================================================
    dataset = []

    for sample_id in expr.columns:
        if sample_id not in labels.index:
            continue

        x = torch.tensor(
            [expr.loc[gene, sample_id] for gene in proteins],
            dtype=torch.float
        ).unsqueeze(1)

        y = torch.tensor([labels.loc[sample_id, "label"]], dtype=torch.long)

        data = Data(
            x=x,
            edge_index=edge_index,
            y=y
        )

        # Add weights if exist
        if 'edge_weight' in locals():
            data.edge_attr = edge_weight

        data.gene_names = proteins

        dataset.append(data)

    print("\n🧬 FINAL DATASET")
    print(f"Samples: {len(dataset)}")
    print(f"Nodes: {len(proteins)}")

    if len(dataset) == 0:
        raise ValueError("❌ Dataset is empty")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("✅ DataLoader ready")

    return loader