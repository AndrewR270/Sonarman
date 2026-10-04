import os
import tempfile
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile
from src.model import SonarClassifier
from src.preprocess import compute_mel_spectrogram

app = FastAPI(
    title="Sonarman Acoustic Classification API",
    description="Acoustic naval & autonomous vessel detection.",
)

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

# Load model on startup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SonarClassifier(num_classes=len(CLASSES)).to(device)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()


@app.get("/")
def health_check():
    return {"status": "healthy", "model_loaded": os.path.exists(MODEL_PATH)}


@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        mel_tensor = compute_mel_spectrogram(tmp_path)
        mel_batch = mel_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(mel_batch)
            probs = F.softmax(logits, dim=1).squeeze(0)

        top_prob, top_idx = torch.max(probs, dim=0)
        predictions = {
            cls: round(prob.item() * 100, 2)
            for cls, prob in zip(CLASSES, probs)  # noqa: E501
        }

        return {
            "predicted_class": CLASSES[top_idx.item()],
            "confidence": round(top_prob.item() * 100, 2),
            "all_probabilities": predictions,
        }
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
