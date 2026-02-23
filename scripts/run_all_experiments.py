import subprocess
import sys
import shutil
import os

# -------------------------
# CLEAN ENVIRONMENT
# -------------------------
def clean_previous_runs():
    print("🧹 Cleaning previous experiment data...")

    if os.path.exists("logs"):
        shutil.rmtree("logs")

    exp_paths = [
        "experiments/device_baseline/results",
        "experiments/scaling_study/results",
        "experiments/precision_study/results"
    ]

    for path in exp_paths:
        if os.path.exists(path):
            shutil.rmtree(path)

    os.makedirs("logs", exist_ok=True)

    for path in exp_paths:
        os.makedirs(path, exist_ok=True)

    print("✅ Clean environment ready\n")


# -------------------------
# CONFIG
# -------------------------
python_exe = sys.executable
CONFIG_VERSION = "v1"
seeds = [42, 123, 999]

configs = [
    f"configs/{CONFIG_VERSION}/desktop_fp32.yaml",
    f"configs/{CONFIG_VERSION}/desktop_fp16.yaml",
    f"configs/{CONFIG_VERSION}/jetson.yaml",
    f"configs/{CONFIG_VERSION}/pi.yaml"
]

datasets = ["proteins"]
hidden_dims = [16, 32, 64, 128]

# 🔥 NEW: MODEL COMPARISON
models = ["gcn", "sage", "gat"]

# -------------------------
# HARDWARE FILTER
# -------------------------
def is_valid_run(config):
    config = config.lower()

    # Skip Jetson / Pi on desktop
    if "jetson" in config or "pi" in config:
        print(f"⛔ Skipping incompatible config: {config}")
        return False

    return True


# -------------------------
# RUN EXPERIMENTS
# -------------------------
if __name__ == "__main__":
    clean_previous_runs()

    total_runs = 0
    success_runs = 0




if __name__ == "__main__":
    clean_previous_runs()

    total_runs = 0
    success_runs = 0

    for model in models:
        for config in configs:

            if not is_valid_run(config):
                continue

            for dataset in datasets:
                for hd in hidden_dims:
                    for seed in seeds:  

                        print("=" * 90)
                        print(f"🚀 Running: {model.upper()} | {config} | {dataset} | hd={hd} | seed={seed}")
                        print("=" * 90)

                        cmd = [
                            python_exe,
                            "-m",
                            "src.training.train",
                            "--model", model,
                            "--dataset", dataset,
                            "--hidden_dim", str(hd),
                            "--config", config,
                            "--seed", str(seed)   # 🔥 PASS SEED
                        ]

                        try:
                            subprocess.run(cmd, check=True)
                            success_runs += 1
                        except subprocess.CalledProcessError:
                            print(f"❌ FAILED: {model} | {dataset} | hd={hd} | seed={seed}")
                            continue

                        total_runs += 1

    print("\n" + "=" * 90)
    print(f"✅ Completed Runs: {success_runs}/{total_runs}")
    print("=" * 90)