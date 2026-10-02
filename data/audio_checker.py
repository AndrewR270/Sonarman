import numpy as np
import scipy.io.wavfile as wavfile

# Load a generated file
sample_rate, audio_data = wavfile.read("data/raw/cargo/cargo_01.wav")

# 1. Basic Stats
print(f"Sample Rate: {sample_rate} Hz")
print(f"Duration: {len(audio_data) / sample_rate:.2f} seconds")
print(f"Data Type: {audio_data.dtype}")
print(f"Min Value: {audio_data.min()}, Max Value: {audio_data.max()}")

# 2. Spectral Analysis (Find Peak Frequencies using FFT)
fft_spectrum = np.abs(np.fft.rfft(audio_data))
freq_axis = np.fft.rfftfreq(len(audio_data), d=1.0 / sample_rate)

# Top 3 most intense frequencies
top_freq_indices = np.argsort(fft_spectrum)[-3:][::-1]
print("\nTop 3 Dominant Frequencies Detected:")
for idx in top_freq_indices:
    print(f"  - {freq_axis[idx]:.1f} Hz (Magnitude: {fft_spectrum[idx]:.0f})")
