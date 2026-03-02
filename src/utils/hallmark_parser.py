def load_hallmark_genes(gmt_path="data/hallmark.gmt"):
    """
    Parse GMT file → returns set of genes
    """
    genes = set()

    with open(gmt_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")

            # format: pathway_name, description, genes...
            pathway = parts[0]
            gene_list = parts[2:]

            # 🔥 Keep only cancer-relevant pathways
            if any(k in pathway for k in [
                "P53", "KRAS", "MYC", "MTOR",
                "PI3K", "APOPTOSIS", "DNA_REPAIR"
            ]):
                genes.update(gene_list)

    return genes