import os
import sys
import numpy as np
import tensorflow as tf
import librosa
import librosa.display
import matplotlib.pyplot as plt

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SRC_DIR)

from model import ConditionalAutoencoder


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_RATE = 22050
DURATION = 30
N_MELS = 128
HOP_LENGTH = 512

GENRES = [
    "classical",
    "jazz",
    "metal",
    "pop",
    "rock"
]

GENRE_TO_ID = {
    "classical": 0,
    "jazz": 1,
    "metal": 2,
    "pop": 3,
    "rock": 4
}

BASE_DIR = os.path.dirname(SRC_DIR)

# NEW IMPROVED MODEL WEIGHTS
WEIGHTS_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_improved_music_genre_autoencoder.weights.h5"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD AUDIO
# ============================================================

def audio_to_mel(audio_path):

    print("\nLoading audio:")
    print(audio_path)

    audio, _ = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        duration=DURATION,
        mono=True
    )

    target_length = SAMPLE_RATE * DURATION

    if len(audio) < target_length:

        audio = np.pad(
            audio,
            (0, target_length - len(audio))
        )

    else:

        audio = audio[:target_length]

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    mel_min = mel_db.min()
    mel_max = mel_db.max()

    if mel_max > mel_min:

        mel_normalized = (
            (mel_db - mel_min)
            / (mel_max - mel_min)
        )

    else:

        mel_normalized = np.zeros_like(mel_db)

    return mel_normalized.astype(np.float32)


# ============================================================
# LOAD IMPROVED TRAINED MODEL
# ============================================================

def load_trained_model():

    print("\n" + "=" * 60)
    print("LOADING IMPROVED TRAINED MODEL")
    print("=" * 60)

    print("\nWeights path:")
    print(WEIGHTS_PATH)

    if not os.path.exists(WEIGHTS_PATH):

        raise FileNotFoundError(
            f"Improved model weights not found:\n{WEIGHTS_PATH}"
        )

    model = ConditionalAutoencoder()

    dummy_spectrogram = tf.zeros(
        (1, 128, 1292, 1)
    )

    dummy_genre = tf.zeros(
        (1,),
        dtype=tf.int32
    )

    model(
        [dummy_spectrogram, dummy_genre]
    )

    model.load_weights(WEIGHTS_PATH)

    print("\nImproved trained model loaded successfully! ✅")

    return model


# ============================================================
# GENRE TRANSFER
# ============================================================

def transfer_genre(
    model,
    input_audio,
    source_genre,
    target_genre
):

    print("\n" + "=" * 60)
    print("MUSIC GENRE TRANSFER")
    print("=" * 60)

    print(f"\nSource genre : {source_genre}")
    print(f"Target genre : {target_genre}")

    if source_genre not in GENRE_TO_ID:

        raise ValueError(
            f"Unknown source genre: {source_genre}"
        )

    if target_genre not in GENRE_TO_ID:

        raise ValueError(
            f"Unknown target genre: {target_genre}"
        )

    # Convert audio into mel-spectrogram
    mel = audio_to_mel(input_audio)

    print(
        f"\nInput mel-spectrogram shape: {mel.shape}"
    )

    # Add channel dimension
    spectrogram = np.expand_dims(
        mel,
        axis=-1
    )

    # Add batch dimension
    spectrogram = np.expand_dims(
        spectrogram,
        axis=0
    )

    # Target genre ID
    target_id = np.array(
        [GENRE_TO_ID[target_genre]],
        dtype=np.int32
    )

    print(
        f"Target genre ID: {target_id[0]}"
    )

    # --------------------------------------------------------
    # Generate transformed spectrogram
    # --------------------------------------------------------

    generated, genre_prediction = model(
        [spectrogram, target_id],
        training=False
    )

    generated = generated.numpy()[0, :, :, 0]

    genre_prediction = genre_prediction.numpy()[0]

    predicted_genre_id = int(
        np.argmax(genre_prediction)
    )

    predicted_genre = GENRES[predicted_genre_id]

    print(
        f"\nGenerated spectrogram shape: "
        f"{generated.shape}"
    )

    print(
        f"Classifier predicted genre: "
        f"{predicted_genre}"
    )

    print(
        "Genre probabilities:"
    )

    for genre, probability in zip(
        GENRES,
        genre_prediction
    ):

        print(
            f"  {genre:10s}: {probability:.4f}"
        )

    # --------------------------------------------------------
    # Save numpy output
    # --------------------------------------------------------

    output_npy = os.path.join(
        OUTPUT_DIR,
        f"{source_genre}_to_{target_genre}_improved.npy"
    )

    np.save(
        output_npy,
        generated
    )

    print(
        f"\nGenerated spectrogram saved:\n"
        f"{output_npy}"
    )

    # --------------------------------------------------------
    # Save visualization
    # --------------------------------------------------------

    output_png = os.path.join(
        OUTPUT_DIR,
        f"{source_genre}_to_{target_genre}_improved.png"
    )

    plt.figure(
        figsize=(12, 5)
    )

    librosa.display.specshow(
        generated,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel"
    )

    plt.colorbar(
        format="%+2.0f dB"
    )

    plt.title(
        f"{source_genre.title()} → "
        f"{target_genre.title()} "
        f"(Improved Model)"
    )

    plt.tight_layout()

    plt.savefig(
        output_png,
        dpi=150
    )

    plt.close()

    print(
        f"Spectrogram image saved:\n"
        f"{output_png}"
    )

    return generated


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("AI MUSIC GENRE STYLE TRANSFER")
    print("IMPROVED MODEL")
    print("=" * 60)

    # --------------------------------------------------------
    # Input:
    # Pop song
    # Target:
    # Classical
    # --------------------------------------------------------

    input_audio = os.path.join(
        BASE_DIR,
        "data",
        "raw",
        "pop",
        "pop.00000.wav"
    )

    source_genre = "pop"
    target_genre = "classical"

    if not os.path.exists(input_audio):

        print(
            f"\nERROR: Input audio not found:\n"
            f"{input_audio}"
        )

        return

    # Load improved trained model
    model = load_trained_model()

    # Perform genre transfer
    transfer_genre(
        model,
        input_audio,
        source_genre,
        target_genre
    )

    print("\n" + "=" * 60)
    print("IMPROVED GENRE TRANSFER COMPLETED! ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()