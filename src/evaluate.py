from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from tqdm import tqdm


# These settings match src/train.py so evaluation uses the same validation split.
DATA_DIR = Path("data")
MODEL_PATH = Path("models/mask_detector.pth")
BATCH_SIZE = 32
IMAGE_SIZE = 224
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42


class RGBImageFolder(datasets.ImageFolder):
    """ImageFolder that converts every image to RGB before applying transforms."""

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


def create_validation_loader():
    """Load the dataset and recreate the same validation split used during training."""
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

    validation_size = int(len(dataset) * VALIDATION_SPLIT)
    train_size = len(dataset) - validation_size

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    _, validation_dataset = random_split(
        dataset,
        [train_size, validation_size],
        generator=generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    return dataset, validation_loader


def create_model(num_classes):
    """Create the same ResNet18 architecture used in training."""
    model = models.resnet18(weights=None)
    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, num_classes)
    return model


def load_model(model_path, device):
    """Load the saved checkpoint without changing or retraining it."""
    checkpoint = torch.load(model_path, map_location=device)
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {index: class_name for class_name, index in class_to_idx.items()}

    model = create_model(num_classes=len(class_to_idx))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, idx_to_class


def evaluate_model(model, validation_loader, device):
    """Run the model on the validation set and collect predictions."""
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in tqdm(validation_loader, desc="Evaluating"):
            images = images.to(device)

            outputs = model(images)
            _, predictions = torch.max(outputs, 1)

            all_labels.extend(labels.tolist())
            all_predictions.extend(predictions.cpu().tolist())

    return all_labels, all_predictions


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    device = get_device()
    print(f"Using device: {device}")

    dataset, validation_loader = create_validation_loader()
    model, idx_to_class = load_model(MODEL_PATH, device)

    labels, predictions = evaluate_model(model, validation_loader, device)
    class_names = [idx_to_class[index] for index in sorted(idx_to_class)]

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )
    matrix = confusion_matrix(labels, predictions)

    print("\nValidation results")
    print(f"Classes: {dataset.classes}")
    print(f"Validation images: {len(validation_loader.dataset)}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1_score:.4f}")

    print("\nConfusion matrix")
    print(f"Rows = true labels, columns = predicted labels: {class_names}")
    print(matrix)


if __name__ == "__main__":
    main()
