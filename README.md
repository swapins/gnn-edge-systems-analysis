# Edge-GNN: Systems-Level Analysis of Protein-Protein Interaction in Oncology
![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Status](https://img.shields.io/badge/status-stable-green.svg)
![Platform](https://img.shields.io/badge/platform-Edge_AI-orange.svg)
[![Python package](https://github.com/swapins/gnn-edge-systems-analysis/actions/workflows/python-package.yml/badge.svg)](https://github.com/swapins/gnn-edge-systems-analysis/actions/workflows/python-package.yml)

**Author:** Swapin Vidya

**Role:** Senior Systems Architect • Lead Edge AI Researcher

**Domain:** Computational Oncology • Graph Representation Learning • Distributed Edge Systems

---

## Executive Summary

This repository introduces a **systems-driven framework** for the deployment of Graph Neural Networks (GNNs) on Protein-Protein Interaction (PPI) networks within resource-constrained environments.

While contemporary bioinformatics often relies on monolithic high-performance computing (HPC) clusters, this research demonstrates that **clinically relevant graph learning** can be democratized. By optimizing the interplay between biological graph complexity and hardware limitations, this system enables high-fidelity inference on NVIDIA Jetson, Raspberry Pi, and CPU-bound edge nodes without sacrificing predictive accuracy.

---

## Architectural Framework

### 1. Bio-Compute Alignment Layer

Biological networks present a unique computational challenge: they are inherently high-dimensional and non-Euclidean. My architecture resolves the "Memory-Throughput Gap" via:

* **Sparse Graph Representation:** Minimizing the memory footprint of adjacency matrices.
* **Hardware-Aware Scaling:** Dynamic adjustment of `hidden_dim` and `layer_depth` based on real-time telemetry.
* **Precision Switching:** Seamless transitions between **FP32** (for training stability) and **FP16/INT8** (for edge inference).

### 2. Model & Task Topology

* **Backbone:** GCN/GraphSAGE/GAT (Configurable).
* **Pooling:** Global Mean/Max pooling for graph-level representation.
* **Objective:** Binary classification (Malignant vs. Benign phenotypes).
* **Dataset Support:** Synthetic PPI, Injected TCGA (The Cancer Genome Atlas), and Real-world Patient Genomics.

---

```mermaid
flowchart TD

A[Protein Interaction Data (PPI / Gene Expression)] --> B[Preprocessing Layer]
B --> C[Feature Engineering (Normalization, Encoding)]

C --> D[Graph Construction (PPI Network)]

D --> E[GNN Core Engine]
E --> E1[GCN / GraphSAGE / GAT Layers]
E1 --> E2[Node Embeddings]
E2 --> E3[Protein Function Prediction]

E3 --> F[Edge Inference Runtime]

F --> F1[CPU Execution (SBC)]
F --> F2[Adaptive Compute (GPU → CPU fallback)]

F --> G[Output (Oncology Insight / Classification)]

subgraph Edge Constraints
H1[Memory Limits]
H2[Latency Budget]
H3[Energy Constraints]
end

E --> H1
F --> H2
F --> H3
```


## Engineering & Installation

### Environment Setup

```bash
git clone https://github.com/swapins/gnn-edge-systems-analysis.git
cd gnn-edge-systems-analysis

# Initialize isolated environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Production-grade install (links the gnn-edge CLI command)
pip install -r requirements.txt
pip install -e .

```

---

## Experiment Orchestration

The system utilizes a modular CLI located in `src/cli` for reproducible research.

### 🔹 Full Automated Pipeline

The `gnn-edge` runner (mapped to `src/cli/main.py`) manages the entire lifecycle: hardware detection, config injection, and artifact logging.

```bash
gnn-edge run

```

* **State Management:** Automatically cleans legacy logs and prevents data corruption.
* **Matrix Execution:** Iterates across `configs/v1` × `datasets` × `hardware_profiles`.

### 🔹 Granular Execution

For specific hyperparameter tuning or architectural debugging:

```bash
python -m src.training.train \
    --config configs/v1/desktop_fp32.yaml \
    --dataset tcga_real \
    --hidden_dim 128

```

---

## Architect-Level Experiment Design

### Systems Features

* **Adaptive Fallback:** Intelligent CUDA → CPU switching with detailed warning intercepts.
* **Experiment Registry:** Unique UUID-based tracking for every run to ensure 100% reproducibility.
* **Constraint-Aware Scaling:** Automatic model pruning or batch-size reduction upon OOM (Out-of-Memory) detection.

### Data Hierarchy

| Scenario | Data Type | Purpose |
| --- | --- | --- |
| **Base** | Synthetic PPI | Sanity testing & pipeline validation. |
| **TCGA Simulated** | Injected Expression | Testing model robustness against biological noise. |
| **TCGA Real** | Patient Genomics | Validating clinical relevance and AUC benchmarks. |

---

## Benchmarks & Insights

Research indicates that optimized GNNs on edge hardware can maintain a high Area Under the Curve (AUC) while operating within strict thermal and power envelopes.

| Hardware Profile | Target AUC | Latency | Memory Peak |
| --- | --- | --- | --- |
| **Desktop (RTX 4090)** | 0.92 | < 5ms | ~400MB |
| **NVIDIA Jetson** | 0.89 | ~15ms | ~450MB |
| **Raspberry Pi 4/5** | 0.82 | ~80ms | ~320MB |

---

## Repository Structure

```text
gnn-edge-systems-analysis/
├── src/
│   ├── cli/            # Central entry point (main.py, __init__.py)
│   ├── orchestration/  # Experiment lifecycle & hardware detection
│   ├── profiling/      # Resource telemetry (CPU/GPU/RAM)
│   ├── models/         # GNN Architectures (GCN, GAT, SAGE)
│   ├── training/       # Training loops & validation logic
│   ├── data/           # PPI graph processing & loaders
│   └── analysis/       # Post-experiment result processing
├── configs/            # Versioned YAML hardware/model configs
├── experiments/        # Structured experiment output & artifacts
├── logs/               # Telemetry and failure-safe logging
├── results/            # Aggregated CSVs and performance metrics
├── scripts/            # Visualization and utility scripts
├── requirements.txt    # Project dependencies
└── setup.py            # Package distribution & CLI registration

```

---

## Future Roadmap

1. **Federated Edge Learning:** Enabling multi-institution training without data exfiltration.
2. **Quantization-Aware Training (QAT):** Pushing models to 4-bit/8-bit for microcontroller deployment.
3. **Explainable AI (XAI):** Integrating GNNExplainer to identify critical protein sub-graphs for clinicians.


## Citation

If this research or the underlying systems framework assists in your work, please cite it using the following formats:

### BibTeX

```bibtex
@software{Vidya_Edge-GNN_Systems-Level_Analysis_2026,
  author = {Vidya, Swapin},
  title = {{Edge-GNN: Systems-Level Analysis of Protein-Protein Interaction in Oncology}},
  url = {https://github.com/swapins/gnn-edge-systems-analysis},
  version = {1.0.0},
  year = {2026},
  month = {2}
}

```

### APA

Vidya, S. (2026). *Edge-GNN: Systems-Level Analysis of Protein-Protein Interaction in Oncology* (Version 1.0.0) [Computer software]. [https://github.com/swapins/gnn-edge-systems-analysis](https://www.google.com/url?sa=E&source=gmail&q=https://github.com/swapins/gnn-edge-systems-analysis)

---

## Contact

**Vidya, S. (2026).** *Edge-Based Execution of Graph Neural Networks for Protein Interaction Network Analysis in Clinical Oncology.*

**Portfolio:** [Peachbot AI](https://peachbot.in)

**Email:** [swapin@peachbot.in](mailto:swapin@peachbot.in)

**LinkedIn:** [linkedin.com/in/swapin-vidya/](https://www.linkedin.com/in/swapin-vidya/)

---