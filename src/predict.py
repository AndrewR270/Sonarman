import os
import torch
import torch.nn.functional as F
from src.preprocess import compute_mel_spectrogram
from src.model import SonarClassifier

MODEL_PATH = os.path.join("models", "sonar_cnn.pth")
CLASSES = [
    "cargo",
    "passenger",
    "ssbn_ohio",
    "ssn_akula",
    "ssn_los_angeles",
    "tanker",
    "tug",
]


def predict_vessel(audio_path: str, model_path: str = MODEL_PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Compute spectrogram tensor
    mel_tensor = compute_mel_spectrogram(audio_path)

    # Add batch dimension [1, 1, 128, TIME]
    mel_batch = mel_tensor.unsqueeze(0).to(device)

    # 2. Load model
    model = SonarClassifier(num_classes=len(CLASSES)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. Inference
    with torch.no_grad():
        logits = model(mel_batch)
        probabilities = F.softmax(logits, dim=1).squeeze(0)

    top_prob, top_idx = torch.max(probabilities, dim=0)
    predicted_class = CLASSES[top_idx.item()]
    confidence = top_prob.item() * 100.0

    print(f"\nAudio File: {audio_path}")
    print(f"Prediction: {predicted_class} ({confidence:.2f}% confidence)\n")
    print("Class Probabilities:")
    for cls, prob in zip(CLASSES, probabilities):
        print(f"  {cls:<15}: {prob.item() * 100:.2f}%")

    return predicted_class, confidence


if __name__ == "__main__":
    import sys

    test_file = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.join("data", "raw", "cargo", "cargo_01.wav")
    )
    predict_vessel(test_file)
