from setuptools import setup, find_packages

setup(
    name="gnn-edge",
    version="1.0.0",
    description="Systems-level GNN analysis for PPI in Oncology",
    author="Swapin Vidya",
    packages=find_packages(),
    # Critical: This ensures dependencies install automatically
    install_requires=[
        "torch",
        "torch-geometric",
        "pyyaml",
        "pandas",
        "numpy",
        "matplotlib"
    ],
    entry_points={
        "console_scripts": [
            "gnn-edge=src.cli.main:main"
        ]
    },
    python_requires=">=3.8",
)