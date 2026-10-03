import torch
import torch.nn as nn
import torch.nn.functional as F


class SonarClassifier(nn.Module):
    """2D CNN Classifier for Mel-Spectrogram acoustic vessel signals."""

    def __init__(self, num_classes: int = 7):
        super(SonarClassifier, self).__init__()

        # Conv Block 1
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(16)

        # Conv Block 2
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(32)

        # Conv Block 3
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm2d(64)

        # Pooling & Dropout
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.3)

        # Adaptive Pooling guarantees fixed feature size
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [batch_size, 1, 128, time_frames]

        # Conv 1 -> BatchNorm -> ReLU -> Pool
        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        # Conv 2 -> BatchNorm -> ReLU -> Pool
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # Conv 3 -> BatchNorm -> ReLU -> Pool
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Global Adaptive Average Pool -> [batch_size, 64, 4, 4]
        x = self.global_pool(x)

        # Flatten tensor for Linear layers -> [batch_size, 1024]
        x = torch.flatten(x, 1)

        # FC 1 -> Dropout -> ReLU
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # Output logits [batch_size, num_classes]
        logits = self.fc2(x)
        return logits


if __name__ == "__main__":
    # Diagnostic architecture check
    model = SonarClassifier(num_classes=7)
    dummy_input = torch.randn(16, 1, 128, 216)
    output = model(dummy_input)

    print(
        f"Model initialized successfully!\nOutput Shape: {output.shape}"
    )  # Expected: [16, 7]
