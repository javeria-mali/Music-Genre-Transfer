import os
import sys
import tensorflow as tf

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SRC_DIR)

from model import ConditionalAutoencoder


BASE_DIR = os.path.dirname(SRC_DIR)

WEIGHTS_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_improved_music_genre_autoencoder.weights.h5"
)


def main():

    print("=" * 60)
    print("VERIFYING IMPROVED MUSIC GENRE AUTOENCODER")
    print("=" * 60)

    print(f"\nWeights path:\n{WEIGHTS_PATH}")

    if not os.path.exists(WEIGHTS_PATH):
        print("\nERROR: Improved model weights not found! ❌")
        return

    print("\nCreating improved model architecture...")

    model = ConditionalAutoencoder()

    # Build the model using the same input shape used during training
    dummy_spectrogram = tf.zeros((1, 128, 1292, 1))
    dummy_genre = tf.zeros((1,), dtype=tf.int32)

    model((dummy_spectrogram, dummy_genre))

    print("Improved model architecture created successfully! ✅")

    print("\nLoading improved trained weights...")

    model.load_weights(WEIGHTS_PATH)

    print("Improved trained weights loaded successfully! ✅")

    print("\n" + "=" * 60)
    print("IMPROVED MODEL VERIFICATION SUCCESSFUL! ✅")
    print("=" * 60)

    print("\nTotal parameters:", model.count_params())


if __name__ == "__main__":
    main()