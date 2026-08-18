"""Train the model end to end. Reads config/config.yaml (or --config), saves
model to models/ and metrics/figures to outputs/.

Run: python scripts/train.py
     python scripts/train.py --config config/config_stratified.yaml
"""
import argparse
import os
import sys
from pathlib import Path

# Make the project root importable regardless of where this script is run from
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim

from src.config import load_config
from src.data import get_dataloaders
from src.model import build_model
from src.train import get_device, train_model
from src.evaluate import evaluate_model, plot_confusion_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None,
                        help="Path to a config YAML (defaults to config/config.yaml)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    split_type = "stratified" if cfg["data"].get("stratified", False) else "random"
    print(f"Split strategy: {split_type}")

    os.makedirs(cfg["output"]["figures_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(cfg["output"]["model_path"]), exist_ok=True)

    train_loader, test_loader, dataset = get_dataloaders(cfg)
    class_names = dataset.classes  # sanity source of truth vs. config list

    device = get_device()
    model = build_model(cfg["model"]["num_classes"]).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["train"]["learning_rate"])

    epoch_accuracies = train_model(
        model, criterion, optimizer, train_loader, device, epochs=cfg["train"]["epochs"]
    )

    _, test_accuracy, class_accuracy, all_labels, all_preds = evaluate_model(
        model, criterion, test_loader, device, class_names
    )

    plot_confusion_matrix(
        all_labels, all_preds, class_names,
        save_path=os.path.join(cfg["output"]["figures_dir"], "confusion_matrix.png"),
    )

    torch.save(model.state_dict(), cfg["output"]["model_path"])
    print(f"Model saved to {cfg['output']['model_path']}")
    print(f"Test accuracy: {test_accuracy:.2f}")


if __name__ == "__main__":
    main()
