import os
import librosa
import soundfile as sf
import numpy as np

# =========================
# SETTINGS
# =========================
AUDIO_PATH = r"D:\Music-Genre-Transfer\data\raw\pop\pop.00000.wav"
OUTPUT_PATH = r"D:\Music-Genre-Transfer\outputs\original_pop_reconstructed.wav"

SAMPLE_RATE = 22050
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
DURATION = 30


# =========================
# LOAD ORIGINAL AUDIO
# =========================
print("Loading original audio...")

audio, sr = librosa.load(
    AUDIO_PATH,
    sr=SAMPLE_RATE,
    mono=True,
    duration=DURATION
)

print("Original audio loaded!")
print("Sample rate:", sr)
print("Audio length:", len(audio))


# =========================
# CREATE MEL SPECTROGRAM
# =========================
print("\nCreating mel-spectrogram...")

mel = librosa.feature.melspectrogram(
    y=audio,
    sr=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    power=2.0
)

mel_db = librosa.power_to_db(
    mel,
    ref=np.max
)

# Normalize to 0-1
mel_normalized = (mel_db + 80.0) / 80.0
mel_normalized = np.clip(mel_normalized, 0.0, 1.0)

print("Mel shape:", mel_normalized.shape)


# =========================
# CONVERT BACK TO AUDIO
# =========================
print("\nConverting mel-spectrogram back to audio...")

# Convert normalized values back to dB
mel_db_reconstructed = (
    mel_normalized * 80.0
) - 80.0

# Convert dB to power
mel_power = librosa.db_to_power(
    mel_db_reconstructed
)

# Griffin-Lim reconstruction
reconstructed_audio = librosa.feature.inverse.mel_to_audio(
    mel_power,
    sr=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_iter=64,
    power=2.0
)


# =========================
# NORMALIZE AUDIO
# =========================
max_value = np.max(np.abs(reconstructed_audio))

if max_value > 0:
    reconstructed_audio = (
        reconstructed_audio / max_value
    ) * 0.95


# =========================
# SAVE AUDIO
# =========================
sf.write(
    OUTPUT_PATH,
    reconstructed_audio,
    SAMPLE_RATE
)

print("\nReconstructed audio saved successfully! ✅")
print("File:")
print(OUTPUT_PATH)

print("\nRECONSTRUCTION TEST COMPLETED! ✅")