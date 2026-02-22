from setuptools import setup, find_packages

setup(
    name="gnn-edge",
    version="1.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "gnn-edge=src.cli.main:main"
        ]
    },
)