import subprocess
import sys

python_exe = sys.executable

cmd = [python_exe, "-m", "src.training.train"]

print(f"Running: {cmd}")

subprocess.run(cmd)