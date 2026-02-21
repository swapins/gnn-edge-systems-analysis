import os
import json

def save_log(data, filepath):
    # Ensure directory exists
    dirpath = os.path.dirname(filepath)

    if dirpath != "":
        os.makedirs(dirpath, exist_ok=True)

    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)