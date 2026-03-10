import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(
        prog="gnn-edge",
        description="Edge-GNN: A framework for benchmarking Graph Neural Networks on edge hardware."
    )

    # The 'dest' argument is what we check in the if/else logic
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Adding help descriptions to each subparser
    subparsers.add_parser("run", help="Execute the full training and evaluation pipeline")
    subparsers.add_parser("plot", help="Generate research figures and result distributions")
    subparsers.add_parser("compare", help="Compare performance across different hardware profiles")
    subparsers.add_parser("benchmark", help="Measure hardware-specific latency and throughput")
    subparsers.add_parser("pareto", help="Analyze Accuracy vs. Efficiency trade-offs (Pareto Frontier)")

    # If no arguments are provided, print help and exit
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    if args.command == "run":
        os.system("python scripts/run_all_experiments.py")

    elif args.command == "plot":
        os.system("python scripts/plot_results.py")

    elif args.command == "compare":
        os.system("python scripts/compare_devices.py")
    
    elif args.command == "benchmark":
        os.system("python scripts/benchmark.py")

    elif args.command == "pareto":
        os.system("python scripts/pareto_analysis.py")

if __name__ == "__main__":
    main()