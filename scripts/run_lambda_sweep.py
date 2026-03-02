import os
import subprocess
import yaml
import itertools
from copy import deepcopy

# =========================================================
# BASE CONFIGS
# =========================================================
BASE_CONFIGS = [
    "configs/v1/desktop_fp32.yaml",
    "configs/v1/desktop_fp16.yaml"
]

MODELS = ["gcn", "sage", "gat"]
HIDDEN_DIMS = [16, 32, 64, 128]
SEEDS = [42, 123, 999]

# 🔥 CLAIM 3 CORE (λ sweep)
LAMBDA_MEMORY = [0.0, 0.001, 0.01, 0.05]
LAMBDA_TIME = [0.0, 0.001, 0.01, 0.05]

TEMP_CONFIG_DIR = "configs/generated"
os.makedirs(TEMP_CONFIG_DIR, exist_ok=True)

# =========================================================
# HELPER: CREATE TEMP CONFIG
# =========================================================
def create_temp_config(base_config_path, lambda_mem, lambda_time):
    with open(base_config_path, "r") as f:
        config = yaml.safe_load(f)

    config = config or {}

    # Inject constraints safely
    config["constraints"] = {
        "lambda_memory": lambda_mem,
        "lambda_time": lambda_time
    }

    base_name = os.path.basename(base_config_path).replace(".yaml", "")
    new_name = f"{base_name}_lm{lambda_mem}_lt{lambda_time}.yaml"

    new_path = os.path.join(TEMP_CONFIG_DIR, new_name)

    with open(new_path, "w") as f:
        yaml.dump(config, f)

    return new_path

# =========================================================
# RUN EXPERIMENT
# =========================================================
def run_experiment(model, config, hidden_dim, seed):
    cmd = [
        "python", "-m", "src.training.train",
        "--model", model,
        "--config", config,
        "--hidden_dim", str(hidden_dim),
        "--seed", str(seed),
        "--dataset", "proteins"
    ]

    print("=" * 100)
    print(f"🚀 {model.upper()} | {os.path.basename(config)} | HD={hidden_dim} | seed={seed}")
    print("=" * 100)

    subprocess.run(cmd)

# =========================================================
# MAIN SWEEP
# =========================================================
def main():
    total_runs = 0

    for base_config in BASE_CONFIGS:
        for lm, lt in itertools.product(LAMBDA_MEMORY, LAMBDA_TIME):

            # Avoid redundant (0,0) duplicates across configs
            if lm == 0.0 and lt == 0.0:
                config_path = base_config
            else:
                config_path = create_temp_config(base_config, lm, lt)

            for model, hd, seed in itertools.product(MODELS, HIDDEN_DIMS, SEEDS):

                run_experiment(model, config_path, hd, seed)
                total_runs += 1

    print("\n" + "=" * 100)
    print(f"✅ TOTAL RUNS COMPLETED: {total_runs}")
    print("=" * 100)


if __name__ == "__main__":
    main()