import argparse
import os

def main():
    parser = argparse.ArgumentParser(prog="gnn-edge")

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("run")
    subparsers.add_parser("plot")
    subparsers.add_parser("compare")

    args = parser.parse_args()

    if args.command == "run":
        os.system("python scripts/run_all_experiments.py")

    elif args.command == "plot":
        os.system("python scripts/plot_results.py")

    elif args.command == "compare":
        os.system("python scripts/compare_devices.py")

    else:
        print("Usage: gnn-edge [run|plot|compare]")