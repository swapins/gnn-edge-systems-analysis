import os
import pandas as pd
import numpy as np
import requests
import gzip
import shutil
from tqdm import tqdm

# =========================================================
# CONFIG
# =========================================================
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

TCGA_EXPR_URL = "https://toil.xenahubs.net/download/TcgaTargetGtex_rsem_gene_tpm.gz"
TCGA_PHENO_URL = "https://pancanatlas.xenahubs.net/download/TCGA_phenotype_denseDataOnlyDownload.tsv.gz"

GENCODE_GTF_URL = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.annotation.gtf.gz"

# =========================================================
# PATHS
# =========================================================
expr_gz_path = os.path.join(DATA_DIR, "tcga_expression_raw.gz")
expr_path = os.path.join(DATA_DIR, "tcga_expression_raw.tsv")

pheno_gz_path = os.path.join(DATA_DIR, "tcga_phenotype.tsv.gz")
pheno_path = os.path.join(DATA_DIR, "tcga_phenotype.tsv")

gtf_gz_path = os.path.join(DATA_DIR, "gencode.v44.annotation.gtf.gz")

# =========================================================
# DOWNLOAD FUNCTION (ROBUST)
# =========================================================
def download(url, path, retries=3):
    if os.path.exists(path):
        print(f"✅ Exists: {path}")
        return

    for attempt in range(retries):
        try:
            print(f"⬇️ Downloading: {url}")
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            total = int(response.headers.get('content-length', 0))

            with open(path, 'wb') as f, tqdm(
                desc=os.path.basename(path),
                total=total,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))

            print(f"✅ Saved: {path}")
            return

        except Exception as e:
            print(f"⚠️ Attempt {attempt+1} failed: {e}")
            if attempt == retries - 1:
                raise

# =========================================================
# GTF PARSER (HIGHLY OPTIMIZED)
# =========================================================
def parse_gtf_mapping(gtf_path):
    print("🧬 Parsing GENCODE GTF for gene mapping...")

    mapping = {}
    seen = set()

    with gzip.open(gtf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue

            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue

            feature_type = parts[2]
            if feature_type != "gene":
                continue

            attributes = parts[8]

            try:
                gene_id = attributes.split('gene_id "')[1].split('"')[0]
                gene_name = attributes.split('gene_name "')[1].split('"')[0]

                gene_id = gene_id.split(".")[0]

                # Avoid duplicates
                if gene_id not in seen:
                    mapping[gene_id] = gene_name
                    seen.add(gene_id)

            except Exception:
                continue

    print(f"✅ Parsed {len(mapping)} gene mappings")
    return mapping

# =========================================================
# STEP 0: DOWNLOAD FILES
# =========================================================
download(TCGA_EXPR_URL, expr_gz_path)
download(TCGA_PHENO_URL, pheno_gz_path)
download(GENCODE_GTF_URL, gtf_gz_path)

# =========================================================
# STEP 1: EXTRACT
# =========================================================
if not os.path.exists(expr_path):
    print("📦 Extracting expression...")
    with gzip.open(expr_gz_path, "rb") as f_in:
        with open(expr_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

if not os.path.exists(pheno_path):
    print("📦 Extracting phenotype...")
    with gzip.open(pheno_gz_path, "rb") as f_in:
        with open(pheno_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

# =========================================================
# STEP 2: LOAD EXPRESSION
# =========================================================
print("\n📊 Loading expression...")

expr = pd.read_csv(expr_path, sep="\t", index_col=0)

# Keep TCGA samples only
expr = expr.loc[:, expr.columns.str.contains("TCGA")]

# Convert safely
expr = expr.apply(pd.to_numeric, errors="coerce").fillna(0).astype("float32")

print(f"Initial shape: {expr.shape}")

# =========================================================
# STEP 3: GENE MAPPING (GENCODE)
# =========================================================
expr.index = expr.index.astype(str).str.split(".").str[0]

ensembl_to_symbol = parse_gtf_mapping(gtf_gz_path)

print("🔗 Applying gene mapping...")

expr["gene_symbol"] = expr.index.map(ensembl_to_symbol)

# Drop unmapped
expr = expr.dropna(subset=["gene_symbol"])

# Aggregate duplicates
expr = expr.groupby("gene_symbol").mean()

print(f"After mapping: {expr.shape}")

# =========================================================
# STEP 4: LOG TRANSFORM
# =========================================================
expr[expr < 0] = 0
expr = np.log2(expr + 1)

# =========================================================
# STEP 5: VARIANCE FILTER
# =========================================================
print("🔬 Selecting top variable genes...")

variance = expr.var(axis=1)
top_genes = variance.nlargest(2000).index

expr = expr.loc[top_genes]

print(f"After variance filter: {expr.shape}")

# =========================================================
# STEP 6: NORMALIZATION
# =========================================================
mean = expr.mean(axis=1)
std = expr.std(axis=1).replace(0, 1)

expr = expr.sub(mean, axis=0).div(std, axis=0)

# =========================================================
# STEP 7: SAVE EXPRESSION
# =========================================================
expr_out = os.path.join(DATA_DIR, "tcga_expression.csv")
expr.to_csv(expr_out)

print(f"✅ Saved expression: {expr_out}")

# =========================================================
# STEP 8: LABEL PROCESSING
# =========================================================
print("🏷️ Processing labels...")

pheno = pd.read_csv(pheno_path, sep="\t")

# Detect columns
sample_col = next((c for c in ["sample", "_sample", "Sample", "sampleID", "submitter_id"] if c in pheno.columns), None)
type_col = next((c for c in ["_sample_type", "sample_type", "SampleType"] if c in pheno.columns), None)

if sample_col is None or type_col is None:
    raise ValueError("❌ Could not detect required phenotype columns")

# Filter TCGA
pheno = pheno[pheno[sample_col].astype(str).str.contains("TCGA")]

# Binary labels
pheno["label"] = (pheno[type_col] == "Primary Tumor").astype(int)

labels = pheno.set_index(sample_col)[["label"]]

# Align
common = expr.columns.intersection(labels.index)

expr = expr[common]
labels = labels.loc[common]

print(f"Final samples: {len(common)}")

# =========================================================
# SAVE LABELS
# =========================================================
labels_out = os.path.join(DATA_DIR, "tcga_labels.csv")
labels.to_csv(labels_out)

print(f"✅ Saved labels: {labels_out}")

# =========================================================
# FINAL SUMMARY
# =========================================================
print("\n🔥 DATASET READY")
print(f"Samples: {expr.shape[1]}")
print(f"Genes: {expr.shape[0]}")