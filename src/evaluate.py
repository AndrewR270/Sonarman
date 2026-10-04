import os
import torch
from sklearn.metrics import classification_report, confusion_matrix
from src.dataset import get_dataloaders
from src.model import SonarClassifier

MODEL_PATH = os.path.join("models", "sonar_cnn.pth")


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, val_loader, classes = get_dataloaders(batch_size=16)

    # Load trained model
    model = SonarClassifier(num_classes=len(classes)).to(device)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Train the model first!"
        )

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for spectrograms, labels in val_loader:
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            outputs = model(spectrograms)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\n--- Classification Report ---")
    print(
        classification_report(
            all_labels, all_preds, target_names=classes, zero_division=0
        )
    )

    print("--- Confusion Matrix ---")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)


if __name__ == "__main__":
    evaluate()
