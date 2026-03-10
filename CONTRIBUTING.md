# Contributing to Edge-GNN

Thank you for your interest in contributing to **Edge-GNN**.

Edge-GNN is a research-oriented framework for **constraint-aware graph neural networks applied to biological interaction modeling**. Contributions from researchers, engineers, and students are welcome.

This project focuses on:

- Graph Neural Networks (GNNs)
- Computational Biology
- Protein–Protein Interaction Networks
- Computational Oncology
- Systems-aware machine learning


# Ways to Contribute

We welcome several types of contributions.

## Research Contributions

- New graph neural network architectures
- Improvements to constraint-aware optimization
- Biological graph construction methods
- New biological datasets or preprocessing pipelines
- Improved efficiency-performance tradeoff analysis

## Systems Contributions

- Hardware profiling improvements
- Edge deployment optimizations
- Experiment orchestration enhancements
- Performance benchmarking tools

## Documentation

- Improving README clarity
- Adding tutorials or walkthroughs
- Expanding dataset documentation
- Improving reproducibility instructions

## Bug Fixes

If you find bugs in the training pipeline, profiling tools, or experiment orchestration, please submit an issue or pull request.


# Development Setup

Clone the repository:

```bash
git clone https://github.com/swapins/gnn-edge-systems-analysis.git
cd gnn-edge-systems-analysis
````

Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```


# Running Experiments

Run the full experimental pipeline:

```bash
gnn-edge run
```

Generate experiment plots:

```bash
gnn-edge plot
```

Run benchmarking experiments:

```bash
gnn-edge benchmark
```

Perform Pareto analysis:

```bash
gnn-edge pareto
```


# Code Style Guidelines

Please follow these guidelines when contributing:

* Use Python 3.9+
* Follow PEP8 style guidelines
* Write clear and modular code
* Document functions and classes when possible
* Avoid unnecessary external dependencies

---

# Pull Request Process

1. Fork the repository
2. Create a feature branch

```bash
git checkout -b feature/my-improvement
```

3. Commit your changes

```bash
git commit -m "Add new feature or improvement"
```

4. Push the branch

```bash
git push origin feature/my-improvement
```

5. Open a Pull Request

Please include:

* Description of the change
* Motivation for the change
* Expected impact on experiments or results


# Reporting Issues

If you encounter a problem, please create an issue including:

* Operating system
* Python version
* Hardware environment (CPU/GPU)
* Configuration file used
* Error messages or logs

This information helps reproduce and fix the issue quickly.



# Research Collaboration

Researchers interested in extending Edge-GNN to new biological domains or datasets are encouraged to open discussions or submit proposals.

Potential collaboration areas include:

* biological network analysis
* computational oncology
* scalable graph learning
* edge AI for biomedical applications


# Code of Conduct

Please maintain a respectful and collaborative environment for all contributors.

Constructive feedback and open discussion are encouraged.

---

Thank you for contributing to **Edge-GNN**.

