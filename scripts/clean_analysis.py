import os
import shutil

# =========================================================
# TARGET DIRECTORIES (SAFE CLEAN)
# =========================================================
TARGET_DIRS = [
    "experiments/analysis/gene_stability",
]

# =========================================================
# OPTIONAL: ALSO CLEAN OLD IMPORTANCE FILES
# (UNCOMMENT ONLY IF YOU WANT FULL RESET)
# =========================================================
# TARGET_DIRS.append("experiments/analysis/gene_importance")

# =========================================================
# CLEAN FUNCTION
# =========================================================
def clean_dirs(dirs):
    print("\n🧹 Cleaning analysis folders...\n")

    for path in dirs:
        if os.path.exists(path):
            shutil.rmtree(path)
            print(f"❌ Removed: {path}")
        else:
            print(f"⚠️ Not found (skipped): {path}")

    print("\n📁 Recreating structure...\n")

    # Recreate structure
    os.makedirs("experiments/analysis/gene_stability/tables", exist_ok=True)
    os.makedirs("experiments/analysis/gene_stability/latex", exist_ok=True)

    print("✅ Clean structure ready\n")


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    clean_dirs(TARGET_DIRS)