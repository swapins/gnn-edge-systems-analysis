import os
import requests

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Public mirror (no login required)
HALLMARK_URL = "https://raw.githubusercontent.com/igordot/msigdbr/master/inst/extdata/h.all.v7.5.1.symbols.gmt"

SAVE_PATH = os.path.join(DATA_DIR, "hallmark.gmt")

def download():
    if os.path.exists(SAVE_PATH):
        print("✅ Hallmark file already exists")
        return

    print("⬇️ Downloading Hallmark gene sets...")
    r = requests.get(HALLMARK_URL)

    with open(SAVE_PATH, "wb") as f:
        f.write(r.content)

    print(f"✅ Saved → {SAVE_PATH}")


if __name__ == "__main__":
    download()