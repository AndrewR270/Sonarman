import argparse
import os
import sys
import numpy as np
import scipy.io.wavfile as wavfile


def analyze_audio_file(file_path: str, top_n_frequencies: int = 3) -> None:
    """Performs statistical and spectral FFT analysis on a given audio file."""
    if not os.path.exists(file_path):
        print(f"Error: Target file '{file_path}' does not exist.")
        sys.exit(1)

    # Load audio file
    sample_rate, audio_data = wavfile.read(file_path)

    # Handle multi-channel audio by taking the mono channel or average
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    duration_seconds = len(audio_data) / sample_rate

    print("=" * 60)
    print(f"AUDIO DIAGNOSTIC REPORT: {os.path.basename(file_path)}")
    print(f"Path: {file_path}")
    print("=" * 60)

    # 1. Basic Audio Statistics
    print(f"Sample Rate : {sample_rate} Hz")
    print(f"Duration    : {duration_seconds:.2f} seconds")
    print(f"Data Type   : {audio_data.dtype}")
    print(f"Min Value   : {audio_data.min()}")
    print(f"Max Value   : {audio_data.max()}")

    # Check for clipping in 16-bit PCM format
    if audio_data.dtype == np.int16 and (
        audio_data.min() <= -32768 or audio_data.max() >= 32767
    ):
        print("Audio signal appears to be clipping past maximum amplitude.")

    # 2. Spectral Analysis using Fast Fourier Transform (FFT)
    fft_spectrum = np.abs(np.fft.rfft(audio_data))
    freq_axis = np.fft.rfftfreq(len(audio_data), d=1.0 / sample_rate)

    # Retrieve top N dominant frequencies
    top_freq_indices = np.argsort(fft_spectrum)[-top_n_frequencies:][::-1]

    print(f"\nTop {top_n_frequencies} Dominant Frequencies Detected:")
    for rank, idx in enumerate(top_freq_indices, start=1):
        freq_hz = freq_axis[idx]
        magnitude = fft_spectrum[idx]
        print(f"  {rank}. {freq_hz:6.1f} Hz  (Magnitude: {magnitude:,.0f})")

    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Acoustic Sonar Signal Diagnostic Tool"
    )
    parser.add_argument(
        "file_path",
        type=str,
        nargs="?",
        default="data/raw/cargo/cargo_01.wav",
        help="Path to target .wav file (default: data/raw/cargo/cargo_01.wav)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=3,
        help="Number of peak frequencies to display (default: 3)",
    )
    args = parser.parse_args()
    analyze_audio_file(args.file_path, top_n_frequencies=args.top)


if __name__ == "__main__":
    main()
