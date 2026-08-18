"""Model definition."""
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def build_model(num_classes, pretrained=True):
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model
