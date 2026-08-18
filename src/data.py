"""Dataset loading and transforms.

The main model uses stratified sampling (`get_dataloaders`), which preserves
each class's proportion across the train/test split. Simple random sampling
(`get_random_split_dataloaders`) is kept as a comparison baseline.
"""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split


def get_transform(image_size, mean, std):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def get_dataset(cfg):
    """Load the raw ImageFolder dataset with transforms applied."""
    transform = get_transform(
        cfg["data"]["image_size"],
        cfg["data"]["normalize_mean"],
        cfg["data"]["normalize_std"],
    )
    return datasets.ImageFolder(cfg["data"]["data_dir"], transform=transform)


def get_dataloaders(cfg):
    """Stratified train/test split - the project default.

    Each class keeps the same proportion in train and test, which matters here
    because the class counts are uneven.
    """
    dataset = get_dataset(cfg)

    # dataset.targets gives labels without loading/transforming every image
    labels = dataset.targets
    indices = list(range(len(dataset)))

    train_indices, test_indices = train_test_split(
        indices,
        test_size=1 - cfg["data"]["train_split"],
        stratify=labels,
        random_state=cfg["data"]["seed"],
    )

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    batch_size = cfg["data"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, dataset


def get_random_split_dataloaders(cfg):
    """Simple random train/test split, used as a comparison baseline."""
    dataset = get_dataset(cfg)

    train_size = int(cfg["data"]["train_split"] * len(dataset))
    test_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(cfg["data"]["seed"])
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size], generator=generator
    )

    batch_size = cfg["data"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, dataset
