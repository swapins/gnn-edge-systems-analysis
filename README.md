# GNN Edge Systems Analysis

This repository investigates the computational behavior of Graph Neural Networks (GNNs) for protein interaction network analysis across heterogeneous hardware platforms, including:

- Raspberry Pi (CPU-only)
- NVIDIA Jetson Nano (edge GPU)
- Desktop GPU systems

## Objective

To characterize:
- Numerical stability
- Memory constraints
- Latency behavior
- Cross-device convergence dynamics

## Relation to Base Work

This project extends:
https://github.com/swapins/oncology-gnn-edge

and focuses on systems-level experimental analysis.

## Structure

- configs/ → device configurations
- experiments/ → experiment definitions
- src/ → core implementation
- logs/ → raw experiment logs
- results/ → processed results
- analysis/ → notebooks and plots