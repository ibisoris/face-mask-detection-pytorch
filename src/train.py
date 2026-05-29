from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights
from tqdm import tqdm


# Project paths. These are relative to the project root.
DATA_DIR = Path("data")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "mask_detector.pth"

# Training settings. Start small; you can increase epochs later.
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001
IMAGE_SIZE = 224
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42


class RGBImageFolder(datasets.ImageFolder):
    """ImageFolder that always converts images to RGB before transforms."""

    def __getitem__(self, index):
        path, label = self.samples[index]

        with Image.open(path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_device():
    """Use a GPU if one is available, otherwise use the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_dataloaders():
    """Create train and validation DataLoaders from data/with_mask and data/without_mask."""
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    dataset = RGBImageFolder(DATA_DIR, transform=transform)

    if len(dataset.classes) != 2:
        raise ValueError(f"Expected 2 classes, but found: {dataset.classes}")

    validation_size = int(len(dataset) * VALIDATION_SPLIT)
    train_size = len(dataset) - validation_size

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_dataset, validation_dataset = random_split(
        dataset,
        [train_size, validation_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    return dataset, train_loader, validation_loader


def create_model(num_classes):
    """Load a pretrained ResNet18 and replace its final layer for 2 classes."""
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    # Freeze the feature extractor so training is faster and easier.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # ResNet18's last layer is named fc. Replace it with a new classifier.
    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, num_classes)

    return model


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train the model for one epoch and return loss/accuracy."""
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predictions = torch.max(outputs, 1)
        correct_predictions += (predictions == labels).sum().item()
        total_predictions += labels.size(0)

    epoch_loss = running_loss / total_predictions
    epoch_accuracy = correct_predictions / total_predictions
    return epoch_loss, epoch_accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model on validation data."""
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predictions = torch.max(outputs, 1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

    epoch_loss = running_loss / total_predictions
    epoch_accuracy = correct_predictions / total_predictions
    return epoch_loss, epoch_accuracy


def main():
    torch.manual_seed(RANDOM_SEED)
    device = get_device()

    print(f"Using device: {device}")
    dataset, train_loader, validation_loader = create_dataloaders()
    print(f"Classes: {dataset.classes}")
    print(f"Class mapping: {dataset.class_to_idx}")
    print(f"Training images: {len(train_loader.dataset)}")
    print(f"Validation images: {len(validation_loader.dataset)}")

    model = create_model(num_classes=len(dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()

    # Only the new final layer has requires_grad=True, so only it is optimized.
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    best_validation_accuracy = 0.0
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )
        validation_loss, validation_accuracy = evaluate(
            model,
            validation_loader,
            criterion,
            device,
        )

        print(f"Train loss: {train_loss:.4f} | Train accuracy: {train_accuracy:.4f}")
        print(
            f"Validation loss: {validation_loss:.4f} | "
            f"Validation accuracy: {validation_accuracy:.4f}"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "class_to_idx": dataset.class_to_idx,
                "image_size": IMAGE_SIZE,
                "model_name": "resnet18",
            }
            torch.save(checkpoint, MODEL_PATH)
            print(f"Saved best model to {MODEL_PATH}")

    print(f"\nBest validation accuracy: {best_validation_accuracy:.4f}")


if __name__ == "__main__":
    main()
