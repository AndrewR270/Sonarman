import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.dataset import get_dataloaders
from src.model import SonarClassifier

# Hyperparameters
NUM_EPOCHS = 15
LEARNING_RATE = 0.001
BATCH_SIZE = 16
MODEL_SAVE_PATH = os.path.join("models", "sonar_cnn.pth")


def train_model():
    # Setup directories
    os.makedirs("models", exist_ok=True)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Load data
    train_loader, val_loader, classes = get_dataloaders(batch_size=BATCH_SIZE)
    num_classes = len(classes)

    # Initialize network, criterion, and optimizer
    model = SonarClassifier(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0

    print("\n--- Starting Training Loop ---")
    for epoch in range(1, NUM_EPOCHS + 1):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for spectrograms, labels in train_loader:
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(spectrograms)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * spectrograms.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / total_train
        train_acc = (correct_train / total_train) * 100.0

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for spectrograms, labels in val_loader:
                spectrograms = spectrograms.to(device)
                labels = labels.to(device)
                outputs = model(spectrograms)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * spectrograms.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_loss = val_loss / total_val if total_val > 0 else 0.0
        val_acc = (correct_val / total_val) * 100.0 if total_val > 0 else 0.0

        print(
            f"Epoch [{epoch:02d}/{NUM_EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1f}%"
        )

        # Save checkpoint if best validation accuracy achieved
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f"\nTraining completed! Model checkpoint saved: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train_model()
