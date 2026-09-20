
import os
import numpy as np
import tensorflow as tf


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)


# ============================================================
# GENRES
# ============================================================

GENRES = [
    "classical",
    "jazz",
    "metal",
    "pop",
    "rock"
]

GENRE_TO_ID = {
    genre: index
    for index, genre in enumerate(GENRES)
}


# Expected Mel-Spectrogram shape
EXPECTED_SHAPE = (128, 1292)


# ============================================================
# FOLDER CHECK
# ============================================================

def check_folders():

    print("=" * 60)
    print("CHECKING DATASET FOLDERS")
    print("=" * 60)

    if not os.path.exists(PROCESSED_DIR):
        raise FileNotFoundError(
            f"Processed folder not found:\n{PROCESSED_DIR}"
        )

    for genre in GENRES:

        genre_dir = os.path.join(
            PROCESSED_DIR,
            genre
        )

        if not os.path.exists(genre_dir):
            raise FileNotFoundError(
                f"Genre folder not found:\n{genre_dir}"
            )

        if not os.path.isdir(genre_dir):
            raise NotADirectoryError(
                f"Expected folder but found:\n{genre_dir}"
            )

        print(f"✓ {genre} folder found")

    print("\nFolder checks passed! ✅")


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    check_folders()

    spectrograms = []
    genre_ids = []

    print("\n" + "=" * 60)
    print("LOADING PROCESSED DATA")
    print("=" * 60)

    total_files = 0
    skipped_files = 0

    for genre in GENRES:

        genre_dir = os.path.join(
            PROCESSED_DIR,
            genre
        )

        files = sorted([
            file
            for file in os.listdir(genre_dir)
            if file.lower().endswith(".npy")
        ])

        genre_id = GENRE_TO_ID[genre]

        print(
            f"\n{genre}: {len(files)} files "
            f"(genre ID = {genre_id})"
        )

        if len(files) == 0:
            raise ValueError(
                f"No .npy files found in:\n{genre_dir}"
            )

        for filename in files:

            total_files += 1

            file_path = os.path.join(
                genre_dir,
                filename
            )

            try:
                mel = np.load(
                    file_path,
                    allow_pickle=False
                )

            except Exception as error:

                print(
                    f"WARNING: Could not load {filename}"
                )
                print(f"Reason: {error}")

                skipped_files += 1
                continue

            # ------------------------------------------------
            # Shape check
            # ------------------------------------------------

            if mel.shape != EXPECTED_SHAPE:

                print(
                    f"WARNING: Wrong shape in {filename}"
                )

                print(
                    f"Expected: {EXPECTED_SHAPE}"
                )

                print(
                    f"Found: {mel.shape}"
                )

                skipped_files += 1
                continue

            # ------------------------------------------------
            # Numeric check
            # ------------------------------------------------

            if not np.issubdtype(
                mel.dtype,
                np.number
            ):

                print(
                    f"WARNING: Non-numeric data in {filename}"
                )

                skipped_files += 1
                continue

            # ------------------------------------------------
            # NaN / Infinity check
            # ------------------------------------------------

            if not np.isfinite(mel).all():

                print(
                    f"WARNING: NaN/Infinity found in {filename}"
                )

                skipped_files += 1
                continue

            # ------------------------------------------------
            # Add channel dimension
            # ------------------------------------------------

            mel = np.expand_dims(
                mel,
                axis=-1
            )

            spectrograms.append(mel)
            genre_ids.append(genre_id)

    # ========================================================
    # FINAL CHECK
    # ========================================================

    if len(spectrograms) == 0:
        raise ValueError(
            "No valid spectrograms were loaded!"
        )

    X = np.array(
        spectrograms,
        dtype=np.float32
    )

    y = np.array(
        genre_ids,
        dtype=np.int32
    )

    expected_final_shape = (
        len(spectrograms),
        128,
        1292,
        1
    )

    if X.shape != expected_final_shape:

        raise ValueError(
            f"\nFinal shape mismatch!\n"
            f"Expected: {expected_final_shape}\n"
            f"Found: {X.shape}"
        )

    if len(X) != len(y):

        raise ValueError(
            f"\nX and y length mismatch!\n"
            f"X: {len(X)}\n"
            f"y: {len(y)}"
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 60)
    print("DATASET VALIDATION SUMMARY")
    print("=" * 60)

    print(f"Total files found: {total_files}")
    print(f"Valid files loaded: {len(X)}")
    print(f"Skipped files: {skipped_files}")

    print("\nDataset shape:")
    print("X:", X.shape)
    print("y:", y.shape)

    print("\nGenre distribution:")

    for genre in GENRES:

        genre_id = GENRE_TO_ID[genre]

        count = np.sum(
            y == genre_id
        )

        print(
            f"{genre}: {count}"
        )

    print("\nDataset validation passed! ✅")

    return X, y


# ============================================================
# CREATE DATASET
# ============================================================

def create_dataset(
    batch_size=4,
    validation_split=0.2
):

    if batch_size < 1:
        raise ValueError(
            "batch_size must be at least 1."
        )

    if not 0 < validation_split < 1:
        raise ValueError(
            "validation_split must be between 0 and 1."
        )

    X, y = load_data()

    # ========================================================
    # SHUFFLE
    # ========================================================

    indices = np.random.permutation(
        len(X)
    )

    X = X[indices]
    y = y[indices]

    # ========================================================
    # TRAIN / VALIDATION SPLIT
    # ========================================================

    split_index = int(
        len(X) * (1 - validation_split)
    )

    X_train = X[:split_index]
    y_train = y[:split_index]

    X_val = X[split_index:]
    y_val = y[split_index:]

    print("\n" + "=" * 60)
    print("TRAIN / VALIDATION SPLIT")
    print("=" * 60)

    print("\nTraining data:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

    print("\nValidation data:")
    print("X_val:", X_val.shape)
    print("y_val:", y_val.shape)

    # ========================================================
    # TENSORFLOW DATASETS
    # ========================================================
    #
    # Input:
    #   spectrogram
    #   target genre
    #
    # Target:
    #   original spectrogram
    #
    # ========================================================

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (
            (X_train, y_train),
            X_train
        )
    )

    val_dataset = tf.data.Dataset.from_tensor_slices(
        (
            (X_val, y_val),
            X_val
        )
    )

    train_dataset = (
        train_dataset
        .shuffle(min(500, len(X_train)))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_dataset = (
        val_dataset
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    print("\nTensorFlow datasets created! ✅")

    return train_dataset, val_dataset


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    train_dataset, val_dataset = create_dataset()

    print("\n" + "=" * 60)
    print("DATASET LOADER TEST SUCCESSFUL! ✅")
    print("=" * 60)

