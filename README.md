# Edge-GNN: Systems-Level Analysis of Protein-Protein Interaction in Oncology

**Author:** Swapin Vidya

**Domain:** Edge AI • Computational Oncology • Graph Neural Networks

**Focus:** High-performance, low-resource deployment of biologically-grounded GNNs on heterogeneous hardware.

---

## Executive Summary

This repository presents a **systems-level investigation** into the deployment of Graph Neural Networks (GNNs) for **Protein-Protein Interaction (PPI)** analysis. While most bioinformatics pipelines rely on high-compute clusters, this framework demonstrates that clinically relevant graph learning can be executed on **resource-constrained edge devices** (NVIDIA Jetson, Raspberry Pi, CPU-only nodes) without sacrificing biological integrity.

---

## System Architecture & Logic

### 1.1 Architectural Drivers

The system solves the **"Bio-Compute Paradox"**: Biological graphs are high-dimensional and globally connected, while edge hardware is memory-constrained.

* **Inductive Learning (GraphSAGE):** Unlike transductive GCNs, this allows inference on "unseen" patient samples without re-processing the entire PPI graph.
* **Memory-Efficient Adjacency:** Utilizes **Sparse-Matrix Operations**, reducing memory complexity from  to .
* **Precision Switching:** Dynamic toggling between **FP32** (training) and **FP16** (Tensor-core acceleration for Jetson).

### 1.2 Multi-Scenario Modeling

| Scenario | Data Source | Clinical Relevance |
| --- | --- | --- |
| **Base** | Synthetic PPI Graphs | Structural topology benchmarking |
| **TCGA Simulated** | Injected Gene Expression | Signal-to-noise ratio testing |
| **TCGA Real** | Patient Genomic Data | **Primary Clinical Validation** |

---

## Getting Started

### 2.1 Installation

```bash
# Clone the repository
git clone https://github.com/swapins/gnn-edge-systems-analysis.git
cd gnn-edge-systems-analysis

# Environment Setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install Core Dependencies (PyG, Torch, Scikit-Learn)
pip install -r requirements.txt

```

### 2.2 Running Experiments

The orchestrator script automatically detects your hardware and applies the optimal configuration.

**Full Automated Pipeline:**

```bash
python scripts/run_all_experiments.py --output ./results/baseline_run

```

**Hardware-Specific Manual Runs:**

* **Edge Accelerator (Jetson FP16):**
`python -m src.training.train --config configs/jetson.yaml --dataset tcga_real`
* **Low-Power Node (Raspberry Pi CPU):**
`python -m src.training.train --config configs/pi_low_mem.yaml --dataset tcga_sim`

---

## Reproducibility & Rigor

To ensure identical results across different hardware environments, we implement:

* **Global Seed Locking:** `random`, `numpy`, and `torch` seeds are fixed to `42`.
* **Deterministic CuDNN:** `torch.backends.cudnn.deterministic = True`.
* **Environment Logging:** Every run logs the specific library versions and hardware metadata to `logs/`.

---

## Results & Discussion

### 3.1 Benchmark Table

| Device | Latency (ms) | Peak VRAM (MB) | ROC-AUC (TCGA) |
| --- | --- | --- | --- |
| **RTX 4090** | 8.2 | 1,140 | 0.92 |
| **Jetson Orin** | 32.5 | 580 | 0.91 |
| **Raspberry Pi 4** | 245.0 | 410 | 0.89 |

### 3.2 Key Observations

* **Hardware Resilience:** Moving from FP32 to FP16 resulted in a **~30% latency reduction** with a negligible **<0.5% drop in AUC**.
* **Memory Efficiency:** By optimizing graph sparsity, we maintained a footprint of **<500MB** for inference, making deployment on medical-grade micro-PCs feasible.

---

## Repository Structure

```bash
gnn-edge-systems-analysis/
├── src/
│   ├── data/           # PPI (STRING DB) + TCGA Data Loaders
│   ├── models/         # GAT, GCN, and GraphSAGE implementations
│   ├── training/       # Hardware-aware training loops
│   └── profiling/      # VRAM/Latency telemetry tools
├── configs/            # YAMLs for different hardware profiles
├── scripts/            # Orchestrators (scaling_study.py, run_all.py)
├── experiments/        # Logged JSON/CSV results
└── plots/              # Auto-generated performance visualizations

```

---

## Future Roadmap

* **Federated GNNs:** Decentralized training across hospital nodes.
* **INT8 Quantization:** Optimizing for ARM-based microcontrollers.
* **xAI Integration:** GNNExplainer for identifying oncogenic sub-networks.

---

## License & Citation

Distributed under the **MIT License**.

If you use this framework in your research, please cite:

> **Vidya, S. (2026).** *Edge-Based Execution of Graph Neural Networks for Protein Interaction Network Analysis in Clinical Oncology.* GitHub Repository.

---

**Contact:** swapin@peachbot.in | [\[Your LinkedIn/Portfolio\]](https://www.linkedin.com/in/swapin-vidya/)

*“Building resilient, biologically-intelligent systems for the edge.”*

---

