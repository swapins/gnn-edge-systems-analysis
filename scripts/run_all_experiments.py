import subprocess
import sys
import shutil
import os

def clean_previous_runs():
    print("Cleaning previous experiment data...")

    # Remove logs
    if os.path.exists("logs"):
        shutil.rmtree("logs")

    # Remove experiment results
    exp_paths = [
        "experiments/device_baseline/results",
        "experiments/scaling_study/results",
        "experiments/precision_study/results"
    ]

    for path in exp_paths:
        if os.path.exists(path):
            shutil.rmtree(path)

    # Recreate folders
    os.makedirs("logs", exist_ok=True)

    for path in exp_paths:
        os.makedirs(path, exist_ok=True)

    print("Clean environment ready\n")

python_exe = sys.executable

CONFIG_VERSION = "v1"

configs = [
    f"configs/{CONFIG_VERSION}/desktop_fp32.yaml",
    f"configs/{CONFIG_VERSION}/desktop_fp16.yaml",
    f"configs/{CONFIG_VERSION}/jetson.yaml",
    f"configs/{CONFIG_VERSION}/pi.yaml"
]

datasets = ["base", "tcga_sim", "tcga_real"]
hidden_dims = [16, 32, 64, 128]

if __name__ == "__main__":
    clean_previous_runs()

    for config in configs:
        for dataset in datasets:
            for hd in hidden_dims:
                print(f"\nRunning: {config} | {dataset} | hd={hd}")

                cmd = [
                    python_exe,
                    "-m",
                    "src.training.train",
                    "--dataset", dataset,
                    "--hidden_dim", str(hd),
                    "--config", config
                ]

                subprocess.run(cmd)