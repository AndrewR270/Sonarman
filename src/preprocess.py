import os
import soundfile as sf
import torch
import torchaudio.transforms as T

# ==============================================================================
# Global Preprocessing Hyperparameters
# ==============================================================================

SAMPLE_RATE = 22050  # Target Amplitude Sampling Rate /second (Hz)
FFT_WINDOW_LENGTH = 1024  # 1024 samples @ 22,050 Hz = ~46.4 ms per FFT window.
HOP_LENGTH = 512  # Overlap of 50% between FFT windows, ~23.2 ms per frame.
MEL_BANDS = 128  # 128 Mel frequency channels to stretch low-frequencies

RAW_DATA_DIR = os.path.join("data", "raw")
PROCESSED_DATA_DIR = os.path.join("data", "processed")


def compute_mel_spectrogram(
    audio_path: str,
    sample_rate: int = SAMPLE_RATE,
    n_fft: int = FFT_WINDOW_LENGTH,
    hop_length: int = HOP_LENGTH,
    n_mels: int = MEL_BANDS,
) -> torch.Tensor:
    # 1. Load audio data as float32 using soundfile
    data, sr = sf.read(audio_path, dtype="float32")

    # Convert numpy array to tensor with shape [channels, samples]
    if data.ndim == 1:
        waveform = torch.from_numpy(data).unsqueeze(0)
    else:
        waveform = torch.from_numpy(data.T)

    # 2. Resample if original sample rate differs from target
    if sr != sample_rate:
        resampler = T.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)

    # 3. Convert multi-channel audio to mono [1, samples]
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # 4. Compute Mel Spectrogram
    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    mel_spectrogram = mel_transform(waveform)

    # 5. Convert amplitude to dB scale
    log_mel_spectrogram = T.AmplitudeToDB(top_db=80.0)(mel_spectrogram)

    return log_mel_spectrogram


def preprocess_all_vessel_data() -> None:
    """Converts all raw .wav audio files to Mel-spectrogram tensors, and saves
    into `data/processed/` as PyTorch binary files (.pt)."""

    # Ensure raw audio files exist before executing preprocessing
    if not os.path.exists(RAW_DATA_DIR):
        print(f"Directory '{RAW_DATA_DIR}' not found. Generate audio first.")
        return

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    print(f"Preprocessing from '{RAW_DATA_DIR}' -> '{PROCESSED_DATA_DIR}'\n")

    processed_count = 0

    for vessel_class in os.listdir(RAW_DATA_DIR):
        class_raw_dir = os.path.join(RAW_DATA_DIR, vessel_class)

        if not os.path.isdir(class_raw_dir):
            continue

        class_processed_dir = os.path.join(PROCESSED_DATA_DIR, vessel_class)
        os.makedirs(class_processed_dir, exist_ok=True)

        for filename in os.listdir(class_raw_dir):
            if filename.endswith(".wav"):
                audio_path = os.path.join(class_raw_dir, filename)

                # Compute the 2D log-Mel spectrogram tensor
                mel_tensor = compute_mel_spectrogram(audio_path)

                tensor_name = os.path.splitext(filename)[0] + ".pt"
                output_path = os.path.join(class_processed_dir, tensor_name)

                torch.save(mel_tensor, output_path)

                processed_count += 1

        print(f"  [+] Preprocessed class directory: '{vessel_class}'")

    print(f"\nTotal saved tensors: {processed_count}")


# Ensures function runs only when executed directly via CLI
if __name__ == "__main__":
    preprocess_all_vessel_data()
