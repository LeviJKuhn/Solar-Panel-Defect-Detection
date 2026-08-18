"""Run inference on a new image or folder of images.

Run: python scripts/predict.py --input path/to/image.jpg
     python scripts/predict.py --input path/to/folder/
"""
import argparse
import glob
import os
import sys
from pathlib import Path

# Make the project root importable regardless of where this script is run from
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image

from src.config import load_config
from src.data import get_transform
from src.model import build_model


def load_model(cfg, device):
    model = build_model(cfg["model"]["num_classes"], pretrained=False)
    model.load_state_dict(torch.load(cfg["output"]["model_path"], map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_image(model, transform, class_names, image_path, device):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred_idx = probs.argmax().item()

    return class_names[pred_idx], probs[pred_idx].item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Image file or folder of images")
    args = parser.parse_args()

    cfg = load_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, device)
    transform = get_transform(
        cfg["data"]["image_size"], cfg["data"]["normalize_mean"], cfg["data"]["normalize_std"]
    )
    class_names = cfg["data"]["class_names"]

    if os.path.isdir(args.input):
        paths = glob.glob(os.path.join(args.input, "*.jpg")) + \
                glob.glob(os.path.join(args.input, "*.jpeg")) + \
                glob.glob(os.path.join(args.input, "*.png"))
    else:
        paths = [args.input]

    for path in paths:
        label, confidence = predict_image(model, transform, class_names, path, device)
        print(f"{path}: {label} ({confidence * 100:.2f}%)")


if __name__ == "__main__":
    main()
