import subprocess
import sys
import shutil
import os
from tqdm import tqdm

# =========================================================
# CLEAN ENVIRONMENT
# =========================================================
def clean_previous_runs():
    print("🧹 Cleaning previous experiment data...")

    if os.path.exists("logs"):
        shutil.rmtree("logs")

    exp_paths = [
        "experiments/device_baseline/results",
        "experiments/scaling_study/results",
        "experiments/precision_study/results",
        "experiments/analysis"
    ]

    for path in exp_paths:
        if os.path.exists(path):
            shutil.rmtree(path)

    os.makedirs("logs", exist_ok=True)

    for path in exp_paths:
        os.makedirs(path, exist_ok=True)

    print("✅ Clean environment ready\n")


# =========================================================
# CONFIG
# =========================================================
python_exe = sys.executable
CONFIG_VERSION = "v1"

seeds = [42, 123, 999]

configs = [
    f"configs/{CONFIG_VERSION}/desktop_fp32.yaml",
    f"configs/{CONFIG_VERSION}/desktop_fp16.yaml",
    f"configs/{CONFIG_VERSION}/jetson.yaml",
    f"configs/{CONFIG_VERSION}/pi.yaml"
]

datasets = [
    "tcga_sim",
    "tcga_real"
]

hidden_dims = [16, 32, 64, 128]

models = ["gcn", "sage", "gat"]


# =========================================================
# HARDWARE FILTER
# =========================================================
def is_valid_run(config):
    config = config.lower()

    if "jetson" in config or "pi" in config:
        return False

    return True


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    clean_previous_runs()

    valid_configs = [c for c in configs if is_valid_run(c)]

    total_runs = (
        len(models)
        * len(valid_configs)
        * len(datasets)
        * len(hidden_dims)
        * len(seeds)
    )

    success_runs = 0

    print(f"🚀 Total experiment runs: {total_runs}\n")

    pbar = tqdm(total=total_runs, desc="Running Experiments", ncols=100)

    for model in models:
        for config in valid_configs:
            for dataset in datasets:
                for hd in hidden_dims:
                    for seed in seeds:

                        print("=" * 90)
                        print(f"🚀 {model.upper()} | {dataset} | hd={hd} | seed={seed} | {config}")
                        print("=" * 90)

                        cmd = [
                            python_exe,
                            "-m",
                            "src.training.train",
                            "--model", model,
                            "--dataset", dataset,
                            "--hidden_dim", str(hd),
                            "--config", config,
                            "--seed", str(seed)
                        ]

                        try:
                            subprocess.run(cmd, check=True)
                            success_runs += 1

                        except subprocess.CalledProcessError:
                            print(f"❌ FAILED: {model} | {dataset} | hd={hd} | seed={seed}")

                        pbar.update(1)

    pbar.close()

    print("\n" + "=" * 90)
    print(f"✅ Completed Runs: {success_runs}/{total_runs}")
    print("=" * 90)