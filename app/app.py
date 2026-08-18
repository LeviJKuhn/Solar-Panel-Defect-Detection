"""Streamlit demo app.

Run: streamlit run app/app.py
"""
import sys
from pathlib import Path

# Make the project root importable regardless of where this app is run from
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import streamlit as st
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image

from src.config import load_config

cfg = load_config()

class_names = cfg["data"]["class_names"]


@st.cache_resource
def load_model():
    model = resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, cfg["model"]["num_classes"])
    model.load_state_dict(torch.load(cfg["output"]["model_path"], map_location="cpu"))
    model.eval()
    return model


model = load_model()

transform = transforms.Compose([
    transforms.Resize((cfg["data"]["image_size"], cfg["data"]["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize(cfg["data"]["normalize_mean"], cfg["data"]["normalize_std"]),
])

st.title("Solar Panel Fault Classifier")
st.write("Upload a solar panel image to classify its condition.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred_idx = probs.argmax().item()

    st.subheader(f"Prediction: {class_names[pred_idx]}")
    st.write("Confidence scores:")
    for i, name in enumerate(class_names):
        st.write(f"{name}: {probs[i].item() * 100:.2f}%")
