# Sonarman

An AI/ML project in PyTorch with Numpy and Pandas.

## Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html): Package Manager

## Setup

Make new development environment: `conda create -n sonarman python=3.11 -y`

Activate the environment: `conda activate sonarman`

Check current environment: `conda env list`

Install Dependencies: `pip install -r requirements.txt`

## Commands

Generate synthetic audio (from root): `python -m data.synthetic_gen`

Analyze audio file (from root): `python data.audio_checker [options]` (use --help if needed)

## Preprocessing Pipeline

Preprocessing handles the transformation of 16-bit PCM audio waveforms (.wav) to log-scaled Mel-Spectrogram tensors (.pt) suitable for training PyTorch 2D Convolutional Neural Networks (CNNs).

- Load Audio: Read .wav files into PyTorch tensors using torchaudio.
- Channel Normalization: Ensure mono audio structure.
- Short-Time Fourier Transform (STFT) to Mel Scale: Map raw time-domain audio signals onto a frequency-vs-time grid adjusted for sensory perception (n_mels=128).
- Log-Dynamic Range Compression: Convert linear amplitude to logarithmic decibels so faint blade harmonics aren't swallowed by dominant engine peaks.
- Disk Caching (data/processed/): Save converted spectrograms as PyTorch binary tensors (.pt) for fast loading during training.

A Fourier Transform decomposes a complex sound wave into constituent sine waves, showing the energy (magnitude) present across different pitch frequencies. The Short Time Fourier Transform (STFT) used by Sonarman splits the 1D audio wave into overlapping 46ms segments and computes a Fourier Transform on each segment to make a 2D time-frequency matrix. T.MelSpectrogram projects the linear Hz bins into non-linearly spaced frequency channels.

To allow for more accurate neural net training. AmplitudeToDB converts linear power/amplitude (A) into decibels (dB) and clips signals weaker than -80 dB relative to peak, suppressing background noise floor artifacts.

## Devlog

10/1: Added scaffolding, new miniconda evnrionment, synthetic_gen.py, audio_checker.py, lint checks and formatting.