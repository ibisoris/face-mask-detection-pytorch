import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


MODEL_PATH = Path("models/mask_detector.pth")


def get_device():
    """Use a GPU if one is available, otherwise use the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_model(num_classes):
    """Create the same ResNet18 shape used during training."""
    model = models.resnet18(weights=None)
    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, num_classes)
    return model


def load_checkpoint(model_path, device):
    """Load the trained model and class names from disk."""
    checkpoint = torch.load(model_path, map_location=device)
    class_to_idx = checkpoint["class_to_idx"]

    idx_to_class = {index: class_name for class_name, index in class_to_idx.items()}
    model = create_model(num_classes=len(class_to_idx))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, idx_to_class


def preprocess_image(image_path, image_size):
    """Open an image, convert it to RGB, resize it, and turn it into a tensor."""
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image_tensor = transform(image)

    # Add a batch dimension: [channels, height, width] -> [1, channels, height, width]
    return image_tensor.unsqueeze(0)


def predict(image_path, model_path):
    device = get_device()
    checkpoint = torch.load(model_path, map_location=device)
    image_size = checkpoint.get("image_size", 224)

    model, idx_to_class = load_checkpoint(model_path, device)
    image_tensor = preprocess_image(image_path, image_size).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_index = torch.max(probabilities, 1)

    predicted_class = idx_to_class[predicted_index.item()]
    return predicted_class, confidence.item()


def main():
    parser = argparse.ArgumentParser(description="Predict whether a face has a mask.")
    parser.add_argument("image", type=Path, help="Path to the image to classify")
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        help="Path to the trained model checkpoint",
    )
    args = parser.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    if not args.model.exists():
        raise FileNotFoundError(
            f"Model not found: {args.model}. Run python src/train.py first."
        )

    predicted_class, confidence = predict(args.image, args.model)
    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()
