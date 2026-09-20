
import os
import numpy as np
import librosa
import soundfile as sf


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_RATE = 22050
N_MELS = 128
HOP_LENGTH = 512

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)


SPECTROGRAM_PATH = os.path.join(
    OUTPUT_DIR,
    "pop_to_pop.npy"
)


# Generated audio
AUDIO_OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "pop_to_pop.wav"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("GENERATING AUDIO FROM TRANSFORMED SPECTROGRAM")
    print("=" * 60)

    # Check spectrogram
    if not os.path.exists(SPECTROGRAM_PATH):

        print(
            f"\nERROR: Spectrogram not found:\n"
            f"{SPECTROGRAM_PATH}"
        )

        return

    # Load generated spectrogram
    print("\nLoading generated spectrogram...")

    mel_normalized = np.load(
        SPECTROGRAM_PATH,
        allow_pickle=False
    )

    print(
        f"Spectrogram shape: "
        f"{mel_normalized.shape}"
    )

    print(
        f"Minimum value: "
        f"{mel_normalized.min():.4f}"
    )

    print(
        f"Maximum value: "
        f"{mel_normalized.max():.4f}"
    )

    # --------------------------------------------------------
    # Convert normalized spectrogram back to approximate dB
    # --------------------------------------------------------

    # During preprocessing the spectrogram was normalized
    # between 0 and 1.
    #
    # We approximately map:
    # 0 -> -80 dB
    # 1 -> 0 dB

    mel_db = (
        mel_normalized * 80.0
    ) - 80.0

    # Convert dB spectrogram back to power spectrogram
    mel_power = librosa.db_to_power(
        mel_db
    )

    print("\nConverting mel-spectrogram to audio...")

    # --------------------------------------------------------
    # Reconstruct audio
    # --------------------------------------------------------

    audio = librosa.feature.inverse.mel_to_audio(
        mel_power,
        sr=SAMPLE_RATE,
        n_fft=2048,
        hop_length=HOP_LENGTH,
        power=2.0,
        n_iter=32
    )

    # Normalize audio safely
    max_value = np.max(
        np.abs(audio)
    )

    if max_value > 0:

        audio = (
            audio / max_value
        ) * 0.95

    # --------------------------------------------------------
    # Save WAV
    # --------------------------------------------------------

    sf.write(
        AUDIO_OUTPUT_PATH,
        audio,
        SAMPLE_RATE
    )

    print(
        "\nGenerated audio saved successfully! ✅"
    )

    print(
        f"\nAudio file:\n"
        f"{AUDIO_OUTPUT_PATH}"
    )

    print(
        f"\nAudio duration: "
        f"{len(audio) / SAMPLE_RATE:.2f} seconds"
    )

    print("\n" + "=" * 60)
    print("AUDIO GENERATION COMPLETED! ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()

