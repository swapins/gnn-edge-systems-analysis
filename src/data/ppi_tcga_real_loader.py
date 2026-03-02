import os
import requests
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

STRING_LINKS_URL = "https://stringdb-static.org/download/protein.links.v11.5/9606.protein.links.v11.5.txt.gz"
STRING_INFO_URL = "https://stringdb-static.org/download/protein.info.v11.5/9606.protein.info.v11.5.txt.gz"

STRING_LINKS_PATH = os.path.join(DATA_DIR, "string_links.txt.gz")
STRING_INFO_PATH = os.path.join(DATA_DIR, "string_info.txt.gz")


# =========================================================
# DOWNLOAD
# =========================================================
def download(url, path):
    if os.path.exists(path):
        print(f"✅ Exists: {os.path.basename(path)}")
        return

    print(f"⬇️ Downloading {os.path.basename(path)}")
    r = requests.get(url, stream=True)

    with open(path, "wb") as f:
        for chunk in tqdm(r.iter_content(1024 * 1024)):
            if chunk:
                f.write(chunk)


# =========================================================
# LOAD STRING (FIXED FOR YOUR FORMAT)
# =========================================================
def load_string_edges(score_threshold=700):
    print("🧬 Loading STRING PPI...")

    download(STRING_LINKS_URL, STRING_LINKS_PATH)
    download(STRING_INFO_URL, STRING_INFO_PATH)

    edges = pd.read_csv(STRING_LINKS_PATH, sep=" ")
    info = pd.read_csv(STRING_INFO_PATH, sep="\t")

    print(f"🔍 STRING info columns: {list(info.columns)}")

    # ✅ HANDLE YOUR CASE
    if "#string_protein_id" in info.columns:
        id_col = "#string_protein_id"
    elif "protein_id" in info.columns:
        id_col = "protein_id"
    else:
        raise ValueError(f"❌ Unknown STRING format: {info.columns}")

    # -------------------------
    # CLEAN IDS (REMOVE 9606.)
    # -------------------------
    info[id_col] = info[id_col].astype(str).str.replace("9606.", "", regex=False)

    edges["protein1"] = edges["protein1"].astype(str).str.replace("9606.", "", regex=False)
    edges["protein2"] = edges["protein2"].astype(str).str.replace("9606.", "", regex=False)

    # -------------------------
    # MAP TO GENE SYMBOLS
    # -------------------------
    id_to_gene = dict(zip(info[id_col], info["preferred_name"]))

    edges = edges[edges["combined_score"] >= score_threshold]

    edges["gene1"] = edges["protein1"].map(id_to_gene)
    edges["gene2"] = edges["protein2"].map(id_to_gene)

    edges = edges.dropna(subset=["gene1", "gene2"])

    print(f"🔗 STRING edges (filtered): {len(edges)}")

    return edges[["gene1", "gene2", "combined_score"]]


# =========================================================
# MAIN LOADER
# =========================================================
def load_ppi_tcga_real(batch_size=1, max_nodes=300):
    print("\n🚀 Loading TCGA REAL dataset (FINAL FIXED)...")

    expr = pd.read_csv("data/tcga_expression.csv", index_col=0)
    labels = pd.read_csv("data/tcga_labels.csv", index_col=0)
    edges = load_string_edges()

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
    string_genes = set(edges["gene1"]).union(set(edges["gene2"]))

    overlap = list(tcga_genes.intersection(string_genes))

    print(f"🧬 TCGA genes: {len(tcga_genes)}")
    print(f"🔗 STRING genes: {len(string_genes)}")
    print(f"✅ Overlap: {len(overlap)}")

    # =========================================================
    # GRAPH BUILDING
    # =========================================================
    if len(overlap) >= 100:
        print("✅ Using STRING biological graph")

        proteins = overlap[:max_nodes]
        protein_to_idx = {p: i for i, p in enumerate(proteins)}

        edge_list = []
        edge_weight = []

        for _, row in edges.iterrows():
            p1, p2 = row["gene1"], row["gene2"]

            if p1 in protein_to_idx and p2 in protein_to_idx:
                edge_list.append((protein_to_idx[p1], protein_to_idx[p2]))
                edge_weight.append(row["combined_score"] / 1000.0)

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    else:
        print("⚠️ Low overlap → using kNN graph")

        proteins = list(expr.index[:max_nodes])
        gene_matrix = torch.tensor(expr.loc[proteins].values, dtype=torch.float)

        gene_matrix = F.normalize(gene_matrix, dim=1)
        similarity = torch.mm(gene_matrix, gene_matrix.t())

        k = 10
        edge_list = []

        for i in range(len(proteins)):
            _, idx = torch.topk(similarity[i], k + 1)
            for j in idx[1:]:
                edge_list.append((i, j.item()))

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

        edge_weight = None

    print(f"🔗 Edges: {edge_index.shape[1]}")
    print(f"🧬 Nodes: {len(proteins)}")

    # =========================================================
    # DATASET
    # =========================================================
    dataset = []

    for sample_id in expr.columns:
        if sample_id not in labels.index:
            continue

        x = torch.tensor(expr.loc[proteins, sample_id].values, dtype=torch.float).unsqueeze(1)
        y = torch.tensor([labels.loc[sample_id, "label"]], dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, y=y)

        if edge_weight is not None:
            data.edge_attr = edge_weight

        data.gene_names = proteins
        dataset.append(data)

    if len(dataset) == 0:
        raise ValueError("❌ Dataset empty")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("\n🧬 FINAL DATASET")
    print(f"Samples: {len(dataset)}")
    print(f"Nodes: {len(proteins)}")
    print("✅ DataLoader ready")

    return loader