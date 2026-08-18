"""Evaluation metrics and confusion matrix plotting."""
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix


def evaluate_model(model, criterion, test_loader, device, class_names=None):
    model.eval()

    running_loss = 0.0
    running_corrects = 0.0
    total = 0

    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            predicted = outputs.argmax(dim=1)
            running_corrects += (predicted == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

            for label, prediction in zip(labels, predicted):
                label = label.item()
                class_total[label] += 1
                if label == prediction.item():
                    class_correct[label] += 1

    avg_loss = running_loss / total
    overall_accuracy = 100 * running_corrects / total

    print(f"Loss: {avg_loss:.4f} Overall Accuracy: {overall_accuracy:.2f}")
    print("Per-class accuracy:")
    per_class_accuracy = {}
    for class_idx in sorted(class_total.keys()):
        name = class_names[class_idx] if class_names else f"Class {class_idx}"
        acc = 100 * class_correct[class_idx] / class_total[class_idx]
        per_class_accuracy[name] = acc
        print(f"  {name}: {acc:.2f}%  ({class_correct[class_idx]}/{class_total[class_idx]})")

    return avg_loss, overall_accuracy, per_class_accuracy, all_labels, all_preds


def plot_confusion_matrix(all_labels, all_preds, class_names, save_path=None):
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()
