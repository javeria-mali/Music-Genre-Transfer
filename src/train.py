import os
import tensorflow as tf

from dataset import create_dataset
from model import build_model


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 0.0001

MODEL_DIR = "models"
BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_improved_music_genre_autoencoder.weights.h5"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("IMPROVED MUSIC GENRE MODEL TRAINING")
    print("=" * 60)


    # --------------------------------------------------------
    # Create dataset
    # --------------------------------------------------------

    print("\nLoading dataset...")

    train_dataset, val_dataset = create_dataset(
        batch_size=BATCH_SIZE
    )

    print("Dataset loaded successfully! ✅")


    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    print("\nBuilding improved model...")

    model = build_model()

    print("Model built successfully! ✅")


    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    )


    # --------------------------------------------------------
    # Loss functions
    # --------------------------------------------------------

    reconstruction_loss_fn = tf.keras.losses.MeanSquaredError()

    classification_loss_fn = (
        tf.keras.losses.SparseCategoricalCrossentropy()
    )


    # --------------------------------------------------------
    # Training step
    # --------------------------------------------------------

    @tf.function
    def train_step(spectrograms, genres):

        with tf.GradientTape() as tape:

            generated_spectrograms, genre_predictions = model(
                [spectrograms, genres],
                training=True
            )

            reconstruction_loss = reconstruction_loss_fn(
                spectrograms,
                generated_spectrograms
            )

            classification_loss = classification_loss_fn(
                genres,
                genre_predictions
            )

            total_loss = (
                reconstruction_loss
                + 0.5 * classification_loss
            )

        gradients = tape.gradient(
            total_loss,
            model.trainable_variables
        )

        optimizer.apply_gradients(
            zip(
                gradients,
                model.trainable_variables
            )
        )

        return (
            total_loss,
            reconstruction_loss,
            classification_loss
        )


    # --------------------------------------------------------
    # Validation step
    # --------------------------------------------------------

    @tf.function
    def validation_step(spectrograms, genres):

        generated_spectrograms, genre_predictions = model(
            [spectrograms, genres],
            training=False
        )

        reconstruction_loss = reconstruction_loss_fn(
            spectrograms,
            generated_spectrograms
        )

        classification_loss = classification_loss_fn(
            genres,
            genre_predictions
        )

        total_loss = (
            reconstruction_loss
            + 0.5 * classification_loss
        )

        return (
            total_loss,
            reconstruction_loss,
            classification_loss
        )


    # --------------------------------------------------------
    # Prepare model folder
    # --------------------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )


    best_val_loss = float("inf")


    # ========================================================
    # TRAINING LOOP
    # ========================================================

    print("\nStarting training...")
    print("=" * 60)


    for epoch in range(EPOCHS):

        print("\n")
        print("=" * 60)
        print(
            f"Epoch {epoch + 1}/{EPOCHS}"
        )
        print("=" * 60)


        # ----------------------------------------------------
        # Training metrics
        # ----------------------------------------------------

        train_total = 0.0
        train_reconstruction = 0.0
        train_classification = 0.0

        train_batches = 0


        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        for (spectrograms, genres), _ in train_dataset:

            (
                total_loss,
                reconstruction_loss,
                classification_loss
            ) = train_step(
                spectrograms,
                genres
            )


            train_total += float(
                total_loss
            )

            train_reconstruction += float(
                reconstruction_loss
            )

            train_classification += float(
                classification_loss
            )

            train_batches += 1


            if train_batches % 20 == 0:

                print(
                    f"Batch {train_batches} | "
                    f"Loss: {float(total_loss):.4f}"
                )


        # ----------------------------------------------------
        # Average training losses
        # ----------------------------------------------------

        train_total /= train_batches

        train_reconstruction /= train_batches

        train_classification /= train_batches


        # ----------------------------------------------------
        # Validation metrics
        # ----------------------------------------------------

        val_total = 0.0
        val_reconstruction = 0.0
        val_classification = 0.0

        val_batches = 0


        for (spectrograms, genres), _ in val_dataset:

            (
                total_loss,
                reconstruction_loss,
                classification_loss
            ) = validation_step(
                spectrograms,
                genres
            )


            val_total += float(
                total_loss
            )

            val_reconstruction += float(
                reconstruction_loss
            )

            val_classification += float(
                classification_loss
            )

            val_batches += 1


        # ----------------------------------------------------
        # Average validation losses
        # ----------------------------------------------------

        val_total /= val_batches

        val_reconstruction /= val_batches

        val_classification /= val_batches


        # ----------------------------------------------------
        # Display results
        # ----------------------------------------------------

        print("\nTraining Results:")

        print(
            f"Total Loss:          "
            f"{train_total:.4f}"
        )

        print(
            f"Reconstruction Loss: "
            f"{train_reconstruction:.4f}"
        )

        print(
            f"Classification Loss: "
            f"{train_classification:.4f}"
        )


        print("\nValidation Results:")

        print(
            f"Total Loss:          "
            f"{val_total:.4f}"
        )

        print(
            f"Reconstruction Loss: "
            f"{val_reconstruction:.4f}"
        )

        print(
            f"Classification Loss: "
            f"{val_classification:.4f}"
        )


        # ----------------------------------------------------
        # Save best weights
        # ----------------------------------------------------

        if val_total < best_val_loss:

            best_val_loss = val_total

            model.save_weights(
                BEST_MODEL_PATH
            )

            print(
                "\n⭐ Best model weights saved! ✅"
            )


    # ========================================================
    # TRAINING COMPLETED
    # ========================================================

    print("\n")
    print("=" * 60)
    print("IMPROVED MODEL TRAINING COMPLETED! ✅")
    print("=" * 60)

    print(
        f"\nBest validation loss: "
        f"{best_val_loss:.4f}"
    )

    print(
        f"\nBest weights saved at:\n"
        f"{BEST_MODEL_PATH}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()