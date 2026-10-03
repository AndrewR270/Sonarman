# Sonarman

**Sonarman** is an end-to-end Deep Learning acoustic classification pipeline built in PyTorch for autonomous underwater marine systems. It transforms raw hydrophone time-domain acoustic signals into logarithmic Mel-Spectrogram feature maps ($128 \times 216$) to detect and classify vessel signatures across 7 distinct naval classes (`cargo`, `passenger`, `ssbn_ohio`, `ssn_akula`, `ssn_los_angeles`, `tanker`, `tug`).

The repository features an end-to-end Machine Learning pipeline: synthetic audio generation, digital signal processing (DSP), custom PyTorch dataset loading, 2D Convolutional Neural Network (CNN) architecture with batch normalization, metric evaluation, single-file CLI inference, and a low-latency FastAPI microservice.

---

## Technical Architecture & Pipeline Execution Order

The end-to-end workflow follows a strict sequential data pipeline:

1. **Synthetic Signal Generation (`data/synthetic_gen.py`)**: Synthesizes multi-frequency audio waveforms (.wav) representing mechanical blade harmonics and cavitation noise for 7 vessel classes.
2. **Audio Inspection (`data/audio_checker.py`)**: Validates audio sampling rates, channel counts, and duration constraints before processing.
3. **Preprocessing Pipeline (`src/preprocess.py`)**: Converts raw time-domain waveforms into log-scaled Mel-Spectrogram tensors (`.pt`) using Short-Time Fourier Transforms (STFT).
4. **Dataset Management (`src/dataset.py`)**: Discovers cached `.pt` files, maps categorical labels, and creates PyTorch `DataLoader` instances with dynamic train/val splits.
5. **Model Architecture (`src/model.py`)**: Defines a 2D CNN with 3 Convolutional/BatchNorm/ReLU blocks, MaxPool2d, Adaptive AvgPooling, and Linear classification heads.
6. **Model Training (`src/train.py`)**: Trains the network using `nn.CrossEntropyLoss` and `optim.Adam`, saving the highest-accuracy checkpoint to `models/sonar_cnn.pth`.
7. **Model Evaluation (`src/evaluate.py`)**: Evaluates model performance on validation data, outputting per-class precision/recall and a confusion matrix via `scikit-learn`.
8. **CLI Inference (`src/predict.py`)**: Takes any standalone `.wav` audio file, computes its spectral representation on the fly, and outputs class probability rankings.
9. **API Deployment (`app.py`)**: Hosts a FastAPI web service with a graphical UI (`/`) and a POST endpoint (`/predict`) for real-time robotic integration.

---

## File Map & Repository Structure

| File Path | Description & Functionality |
| :--- | :--- |
| `data/synthetic_gen.py` | Generates synthetic `.wav` audio samples with class-specific fundamental frequencies, harmonics, and acoustic noise. |
| `data/audio_checker.py` | Utility script to inspect sample rates, duration, clipping, and spectral sanity of `.wav` files. |
| `src/preprocess.py` | DSP module that loads audio via `soundfile`, normalizes channels, computes STFT Mel-Spectrograms, applies DB scaling, and caches `.pt` tensors. |
| `src/dataset.py` | Custom PyTorch `SonarDataset` class and `get_dataloaders()` factory function for indexing cached tensors and creating batch iterators. |
| `src/model.py` | `SonarClassifier` 2D CNN module with Batch Normalization, Dropout (p=0.3), and Adaptive Average Pooling for spatial feature extraction. |
| `src/train.py` | Execution script for the training loop, loss tracking, validation accuracy monitoring, and state dictionary checkpointing. |
| `src/evaluate.py` | Generates detailed evaluation metrics including precision, recall, F1-scores, and confusion matrices on validation splits. |
| `src/predict.py` | Standalone CLI inference module for predicting single `.wav` audio files with softmax confidence scores. |
| `app.py` | FastAPI application providing a REST API endpoint (`POST /predict`) and an HTML browser frontend (`GET /`) for live predictions. |
| `requirements.txt` | Core python package dependencies (`torch`, `soundfile`, `fastapi`, `uvicorn`, `scikit-learn`, etc.). |

---

## Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html): Package Manager

---

## Setup

Make new development environment: `conda create -n sonarman python=3.11 -y`

Activate the environment: `conda activate sonarman`

Check current environment: `conda env list`

Install Dependencies: `pip install -r requirements.txt`

## Commands

Run these from root.

Generate synthetic audio: `python -m data.synthetic_gen`

Inspect raw audio files: `python data.audio_checker [options]` (use --help if needed)

Preprocess raw audio to tensors: `python -m src.preprocess`

Verify dataset loading: `python -m src.dataset`

Verify model structure: `python -m src.model`

Train model: `python -m src.train`

Evaluate model metrics: `python -m src.evaluate`

Run single-file CLI prediction: `python -m src.predict [file]`

Launch FastAPI web interface & API: `uvicorn app:app --reload`

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

10/2: Added preprocessing piepeline, overhauled audio checker, and adjusted synthetic generation params.

10/03:

Resolved Windows C++ backend linkage issues in torchaudio.load() by building direct NumPy array transformation pipelines via soundfile.

Built custom PyTorch SonarDataset and DataLoader factory (src/dataset.py).

Designed lightweight SonarClassifier 2D CNN architecture (src/model.py) featuring 3 Conv/BatchNorm/ReLU stages and Adaptive Average Pooling.

Implemented model training and checkpoint saving loop (src/train.py), achieving 83.9% training accuracy and ~71.4% validation accuracy.

Developed model evaluation suite (src/evaluate.py) and standalone CLI inference tool (src/predict.py).

Containerized model inference into a REST API and browser application using FastAPI and Uvicorn (app.py).