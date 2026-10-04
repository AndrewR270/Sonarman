import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class SonarDataset(Dataset):
    """PyTorch Dataset for preprocessed Sonar Mel-Spectrogram tensors."""

    def __init__(self, processed_dir: str = os.path.join("data", "processed")):
        self.processed_dir = processed_dir
        self.filepaths = []
        self.labels = []

        # Index class directories and build numerical label mapping
        self.classes = sorted(
            [
                d
                for d in os.listdir(processed_dir)
                if os.path.isdir(os.path.join(processed_dir, d))
            ]
        )
        self.class_to_idx_dict = {
            cls_name: idx for idx, cls_name in enumerate(self.classes)
        }

        # Populate file paths and corresponding class index labels
        for cls_name in self.classes:
            class_dir = os.path.join(processed_dir, cls_name)
            for fname in os.listdir(class_dir):
                if fname.endswith(".pt"):
                    self.filepaths.append(os.path.join(class_dir, fname))
                    self.labels.append(self.class_to_idx_dict[cls_name])

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int):
        # Load preprocessed tensor [1, N_MELS, TIME_FRAMES]
        tensor_path = self.filepaths[idx]
        spectrogram = torch.load(tensor_path)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return spectrogram, label


def get_dataloaders(
    processed_dir: str = os.path.join("data", "processed"),
    batch_size: int = 16,
    train_split: float = 0.8,
):
    """Creates training and validation DataLoaders."""
    dataset = SonarDataset(processed_dir=processed_dir)

    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size

    train_data, val_data = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, dataset.classes


if __name__ == "__main__":
    # Diagnostic test loop
    train_loader, val_loader, classes = get_dataloaders()
    print(f"Detected Vessel Classes ({len(classes)}): {classes}")
    print(f"Training: {len(train_loader)} | Validation: {len(val_loader)}")

    for x_batch, y_batch in train_loader:
        print(
            f"Batch Spectrogram Shape: {x_batch.shape}"
        )  # Expected: [batch_size, 1, 128, time_frames]
        print(f"Batch Labels Shape: {y_batch.shape}")
        break
