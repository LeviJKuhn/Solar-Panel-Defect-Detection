"""Training loop."""
import torch


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model, criterion, optimizer, train_loader, device, epochs):
    epoch_accuracies = []
    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        running_corrects = 0.0
        total = 0

        for images, labels in train_loader:
            optimizer.zero_grad()

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            running_corrects += (predicted == labels).sum().item()
            total += labels.size(0)

        accuracy = 100 * running_corrects / total
        epoch_accuracies.append(accuracy)
        print(f"Epochs: {epoch + 1} Loss: {loss.item():.4f} Accuracy: {accuracy:.2f}")

    return epoch_accuracies
