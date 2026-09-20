
import tensorflow as tf
from tensorflow.keras import layers, Model


# ============================================================
# SETTINGS
# ============================================================

NUM_GENRES = 5
LATENT_DIM = 256


# ============================================================
# CONDITIONAL AUTOENCODER
# ============================================================

class ConditionalAutoencoder(Model):

    def __init__(
        self,
        num_genres=NUM_GENRES,
        latent_dim=LATENT_DIM,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Save configuration
        self.num_genres = num_genres
        self.latent_dim = latent_dim

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        self.encoder = tf.keras.Sequential([

            layers.Input(
                shape=(128, 1292, 1)
            ),

            layers.Conv2D(
                32,
                3,
                strides=2,
                padding="same",
                activation="relu"
            ),

            layers.BatchNormalization(),

            layers.Conv2D(
                64,
                3,
                strides=2,
                padding="same",
                activation="relu"
            ),

            layers.BatchNormalization(),

            layers.Conv2D(
                128,
                3,
                strides=2,
                padding="same",
                activation="relu"
            ),

            layers.BatchNormalization(),

            layers.Conv2D(
                256,
                3,
                strides=2,
                padding="same",
                activation="relu"
            ),

            layers.BatchNormalization(),

            layers.Flatten(),

            layers.Dense(
                self.latent_dim,
                activation="relu"
            )

        ], name="encoder")


        # ----------------------------------------------------
        # Genre Embedding
        # ----------------------------------------------------

        self.genre_embedding = layers.Embedding(
            self.num_genres,
            64,
            name="genre_embedding"
        )


        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        self.decoder_dense = layers.Dense(
            8 * 81 * 256,
            activation="relu"
        )


        self.reshape_layer = layers.Reshape(
            (8, 81, 256)
        )


        self.decoder_conv1 = layers.Conv2DTranspose(
            128,
            3,
            strides=2,
            padding="same",
            activation="relu"
        )


        self.decoder_bn1 = layers.BatchNormalization()


        self.decoder_conv2 = layers.Conv2DTranspose(
            64,
            3,
            strides=2,
            padding="same",
            activation="relu"
        )


        self.decoder_bn2 = layers.BatchNormalization()


        self.decoder_conv3 = layers.Conv2DTranspose(
            32,
            3,
            strides=2,
            padding="same",
            activation="relu"
        )


        self.decoder_bn3 = layers.BatchNormalization()


        self.decoder_conv4 = layers.Conv2DTranspose(
            16,
            3,
            strides=2,
            padding="same",
            activation="relu"
        )


        self.output_layer = layers.Conv2D(
            1,
            3,
            padding="same",
            activation="sigmoid"
        )


        # ----------------------------------------------------
        # Genre Classifier
        # ----------------------------------------------------

        self.genre_classifier = tf.keras.Sequential([

            layers.GlobalAveragePooling2D(),

            layers.Dense(
                128,
                activation="relu"
            ),

            layers.Dropout(0.3),

            layers.Dense(
                self.num_genres,
                activation="softmax"
            )

        ], name="genre_classifier")


    # ========================================================
    # FORWARD PASS
    # ========================================================

    def call(
        self,
        inputs,
        training=False
    ):

        spectrogram, target_genre = inputs


        # ----------------------------------------------------
        # Encode spectrogram
        # ----------------------------------------------------

        latent = self.encoder(
            spectrogram,
            training=training
        )


        # ----------------------------------------------------
        # Target genre embedding
        # ----------------------------------------------------

        genre_vector = self.genre_embedding(
            target_genre
        )


        genre_vector = tf.reshape(
            genre_vector,
            (-1, 64)
        )


        # ----------------------------------------------------
        # Combine content + target genre
        # ----------------------------------------------------

        combined = tf.concat(
            [
                latent,
                genre_vector
            ],
            axis=1
        )


        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        x = self.decoder_dense(
            combined
        )


        x = self.reshape_layer(x)


        x = self.decoder_conv1(x)

        x = self.decoder_bn1(
            x,
            training=training
        )


        x = self.decoder_conv2(x)

        x = self.decoder_bn2(
            x,
            training=training
        )


        x = self.decoder_conv3(x)

        x = self.decoder_bn3(
            x,
            training=training
        )


        x = self.decoder_conv4(x)


        # ----------------------------------------------------
        # Generate spectrogram
        # ----------------------------------------------------

        output = self.output_layer(x)


        # Make exact spectrogram size
        output = tf.image.resize(
            output,
            (128, 1292)
        )


        # ----------------------------------------------------
        # Predict genre
        # ----------------------------------------------------

        genre_prediction = self.genre_classifier(
            output,
            training=training
        )


        return (
            output,
            genre_prediction
        )


    # ========================================================
    # CONFIGURATION
    # ========================================================

    def get_config(self):

        config = super().get_config()

        config.update({

            "num_genres": self.num_genres,

            "latent_dim": self.latent_dim

        })

        return config


# ============================================================
# BUILD MODEL
# ============================================================

def build_model():

    model = ConditionalAutoencoder(
        num_genres=NUM_GENRES,
        latent_dim=LATENT_DIM
    )


    # --------------------------------------------------------
    # Build model with dummy input
    # --------------------------------------------------------

    dummy_spectrogram = tf.zeros(
        (1, 128, 1292, 1)
    )


    dummy_genre = tf.zeros(
        (1,),
        dtype=tf.int32
    )


    model(
        [
            dummy_spectrogram,
            dummy_genre
        ]
    )


    return model


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    model = build_model()

    model.summary()

    print(
        "\nIMPROVED MODEL CREATED SUCCESSFULLY! ✅"
    )

