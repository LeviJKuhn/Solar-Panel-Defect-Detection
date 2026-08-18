"""Download the dataset into data/.

Run: python scripts/download_data.py
"""
import os
import shutil
import sys
from pathlib import Path

# Make the project root importable regardless of where this script is run from
sys.path.append(str(Path(__file__).resolve().parent.parent))

import kagglehub

from src.config import load_config


def main():
    cfg = load_config()

    cache_path = kagglehub.dataset_download(cfg["data"]["kaggle_dataset"])
    print("Downloaded to kagglehub cache:", cache_path)

    for root, dirs, files_list in os.walk(cache_path):
        level = root.replace(cache_path, "").count(os.sep)
        indent = "  " * level
        print(f"{indent}{os.path.basename(root)}/")

    # Copy from kagglehub's cache into the repo's data/ folder so
    # everything downstream reads from cfg["data"]["data_dir"].
    dest = cfg["data"]["data_dir"]
    src = os.path.join(cache_path, "Faulty_solar_panel")
    if not os.path.exists(dest):
        shutil.copytree(src, dest)
        print(f"Copied dataset to {dest}")
    else:
        print(f"{dest} already exists, skipping copy")


if __name__ == "__main__":
    main()
