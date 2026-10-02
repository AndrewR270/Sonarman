import os
import numpy as np
import scipy.io.wavfile as wavfile_writer

# ==============================================================================
# Global Audio & Dataset Configuration Settings
# ==============================================================================
SAMPLING_RATE = 22050  # Num of amplitude measurements / second (hz)
DURATION = 5.0    # Playback time for a synthetic audio clip (s)
SAMPLES_PER_CLASS = 10  # Num of unique .wav files per vessel class
RAW_AUDIO_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "raw")

# ==============================================================================
# Acoustic Profiles for Different Vessel Categories
# Dictionary entries define the acoustic fingerprint / signature of ship class
# "type": {
#     "engine_frequency": Primary engine firing/rotation frequency (hz)
#     "blade_frequencies": Overtones produced by propeller blades
#     "ocean_noise_level": Amplitude multiplier for ambient ocean noise
# },
# ==============================================================================
VESSEL_ACOUSTIC_PROFILES = {
    "cargo": {
        "engine_frequency": 120.0,
        "blade_frequencies": [240.0, 360.0],
        "ocean_noise_level": 0.3,
    },
    "tanker": {
        "engine_frequency": 80.0,
        "blade_frequencies": [160.0, 240.0],
        "ocean_noise_level": 0.4,
    },
    "passenger": {
        "engine_frequency": 200.0,
        "blade_frequencies": [400.0, 600.0],
        "ocean_noise_level": 0.2,
    },
    "tug": {
        "engine_frequency": 150.0,
        "blade_frequencies": [300.0, 450.0],
        "ocean_noise_level": 0.25,
    },
}


def generate_synthetic_vessel_audio(
    engine_frequency: float,
    blade_frequencies: list[float],
    ocean_noise_level: float,
    audio_duration_seconds: float = DURATION,
    sampling_rate_hz: int = SAMPLING_RATE,
) -> np.ndarray:
    """Generates a synthetic hydrophone audio signal containing fundamental
     engine tones, propeller blade harmonics, and ambient ocean noise."""

    total_audio_sample_count = int(sampling_rate_hz * audio_duration_seconds)

    # Evenly spaced array of timepoints over duration
    time_points_array = np.linspace(
        start=0.0,
        stop=audio_duration_seconds,
        num=total_audio_sample_count,
        endpoint=False
    )

    # Float array for the acoustic signal (sine wave) of the engine frequency.
    engine_signal = np.sin(2.0 * np.pi * engine_frequency * time_points_array)

    # Secondary acoustic frequencies (harmonics) caused by propeller blades.
    # Each harmonic is weighted at 50% (0.5 multiplier) of the engine volume.
    for freq in blade_frequencies:
        engine_signal += 0.5 * np.sin(2.0 * np.pi * freq * time_points_array)

    # Gaussian white noise to simulate background ocean noise.
    # 'mean=0' centres the wave, 'scale' sets the volume/standard deviation.
    ambient_ocean_noise = np.random.normal(
        loc=0.0,
        scale=ocean_noise_level,
        size=time_points_array.shape
    )

    # Combine engine_signal and ambient_ocean_noise into a single audio signal.
    raw_audio_signal = engine_signal + ambient_ocean_noise

    # Normalization Step:
    # 1. Find the absolute highest combined_raw_audio_signal amplitude peak.
    # 2. Divide by the amplitude so all values scale between -1.0 and +1.0.
    # 3. Scale to fit 16-bit signed integer PCM wave format (-32768 to 32767).
    max_amplitude = np.max(np.abs(raw_audio_signal))
    normalized_audio = raw_audio_signal / max_amplitude
    normalized_16bit_pcm_audio = np.int16(normalized_audio * 32767)

    return normalized_16bit_pcm_audio


def main():
    """Generates sample audio for each vessel category."""

    # Create the root destination directory ('data/raw') if it does not exist
    os.makedirs(RAW_AUDIO_OUTPUT_DIR, exist_ok=True)
    print(f"Generating synthetic sonar audio in: {RAW_AUDIO_OUTPUT_DIR}\n")

    # Loop over every vessel type and its corresponding dictionary parameters
    for category_name, parameters in VESSEL_ACOUSTIC_PROFILES.items():

        # Build a subfolder path for the current vessel category
        category_path = os.path.join(RAW_AUDIO_OUTPUT_DIR, category_name)
        os.makedirs(category_path, exist_ok=True)

        for file_index in range(SAMPLES_PER_CLASS):

            # Call audio generator function using specific acoustic parameters
            audio_data = generate_synthetic_vessel_audio(
                engine_frequency=parameters["engine_frequency"],
                blade_frequencies=parameters["blade_frequencies"],
                ocean_noise_level=parameters["ocean_noise_level"]
            )

            # Format the output file name with two-digit zero padding
            filename = f"{category_name}_{file_index + 1:02d}.wav"
            output_path = os.path.join(category_path, filename)

            # Save the integer array into a valid 16-bit uncompressed WAV file
            wavfile_writer.write(output_path, SAMPLING_RATE, audio_data)

        print(f"  [+] Created {SAMPLES_PER_CLASS} {category_name} files.")

    print("\nSynthetic sonar dataset successfully generated!")


# Ensures main() runs only when executing this script directly from terminal
if __name__ == "__main__":
    main()
