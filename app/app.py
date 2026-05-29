from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


MODEL_PATH = Path("models/mask_detector.pth")
IMAGE_SIZE = 224

DISPLAY_LABELS = {
    "with_mask": {
        "text": "✅ Mask Detected",
        "class_name": "mask-result-success",
        "summary": "The model predicts that this person is wearing a face mask.",
    },
    "without_mask": {
        "text": "⚠️ No Mask Detected",
        "class_name": "mask-result-warning",
        "summary": "The model predicts that this person is not wearing a face mask.",
    },
}


def get_device():
    """Use a GPU if one is available, otherwise use the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_model(num_classes):
    """Create the same ResNet18 architecture used during training."""
    model = models.resnet18(weights=None)
    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, num_classes)
    return model


@st.cache_resource
def load_model(model_path):
    """Load the trained model once and reuse it between Streamlit reruns."""
    device = get_device()
    checkpoint = torch.load(model_path, map_location=device)
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {index: class_name for class_name, index in class_to_idx.items()}

    model = create_model(num_classes=len(class_to_idx))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    image_size = checkpoint.get("image_size", IMAGE_SIZE)
    return model, idx_to_class, image_size, device


def preprocess_image(image, image_size):
    """Convert the uploaded image to RGB, resize it, normalize it, and add a batch size."""
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

    image = image.convert("RGB")
    image_tensor = transform(image)
    return image_tensor.unsqueeze(0)


def predict(image, model, idx_to_class, image_size, device):
    """Predict the class name and confidence for one uploaded image."""
    image_tensor = preprocess_image(image, image_size).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_index = torch.max(probabilities, 1)

    predicted_class = idx_to_class[predicted_index.item()]
    return predicted_class, confidence.item()


def add_custom_styles():
    """Add a small amount of CSS to make the app feel cleaner and more modern."""
    st.markdown(
        """
        <style>
        .main .block-container {
            max-width: 900px;
            padding-top: 2.5rem;
        }

        .app-subtitle {
            color: #b9c0cc;
            font-size: 1.05rem;
            line-height: 1.6;
            margin-bottom: 1.5rem;
        }

        .prediction-card {
            border-radius: 8px;
            margin-top: 1.25rem;
            padding: 1.25rem 1.35rem;
            border: 1px solid rgba(255, 255, 255, 0.12);
            background: #111827;
        }

        .mask-result-success {
            border-left: 6px solid #22c55e;
            box-shadow: 0 0 0 1px rgba(34, 197, 94, 0.12);
        }

        .mask-result-warning {
            border-left: 6px solid #ef4444;
            box-shadow: 0 0 0 1px rgba(239, 68, 68, 0.12);
        }

        .prediction-label {
            font-size: 2rem;
            font-weight: 800;
            line-height: 1.2;
            margin-bottom: 0.35rem;
        }

        .prediction-summary {
            color: #d1d5db;
            font-size: 1rem;
            margin-bottom: 0;
        }

        .confidence-label {
            color: #e5e7eb;
            font-size: 1.05rem;
            font-weight: 700;
            margin-top: 1.25rem;
            margin-bottom: 0.35rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_prediction(predicted_class, confidence):
    """Show a friendly prediction label, color styling, and confidence bar."""
    display = DISPLAY_LABELS.get(
        predicted_class,
        {
            "text": predicted_class,
            "class_name": "",
            "summary": "The model returned this prediction.",
        },
    )

    st.markdown(
        f"""
        <div class="prediction-card {display["class_name"]}">
            <div class="prediction-label">{display["text"]}</div>
            <p class="prediction-summary">{display["summary"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    confidence_percent = round(confidence * 100)
    st.markdown(
        f'<div class="confidence-label">Confidence: {confidence:.2%}</div>',
        unsafe_allow_html=True,
    )
    st.progress(confidence_percent)


def main():
    st.set_page_config(page_title="Face Mask Detection")
    add_custom_styles()

    st.title("Face Mask Detection")
    st.markdown(
        "Upload a face image and the trained ResNet18 model will predict whether "
        "the person is wearing a mask.",
        unsafe_allow_html=False,
    )

    if not MODEL_PATH.exists():
        st.error("Model file not found. Please make sure models/mask_detector.pth exists.")
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is None:
        st.info("Choose an image to get a prediction.")
        return

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    model, idx_to_class, image_size, device = load_model(MODEL_PATH)
    predicted_class, confidence = predict(
        image,
        model,
        idx_to_class,
        image_size,
        device,
    )

    show_prediction(predicted_class, confidence)


if __name__ == "__main__":
    main()
