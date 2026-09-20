import os
import sys
import tempfile

import numpy as np
import librosa
import librosa.display
import soundfile as sf
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(os.path.join(BASE_DIR, "src"))

from model import ConditionalAutoencoder


MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_improved_music_genre_autoencoder.weights.h5"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_RATE = 22050
DURATION = 30
N_MELS = 128
HOP_LENGTH = 512
MAX_FRAMES = 1292

GENRES = [
    "classical",
    "jazz",
    "metal",
    "pop",
    "rock"
]


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Music Genre Style Transfer",
    page_icon="🎵",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(
            135deg,
            #12002b 0%,
            #240046 45%,
            #3c096c 100%
        );
        color: white;
    }

    .main-title {
        text-align: center;
        font-size: 45px;
        font-weight: 800;
        color: #ff8fab;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #ffd6e0;
        margin-bottom: 35px;
    }

    .info-box {
        padding: 20px;
        border-radius: 18px;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.15);
        margin-bottom: 20px;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 12px;
        background: linear-gradient(
            90deg,
            #ff4d8d,
            #c77dff
        );
        color: white;
        font-weight: bold;
        border: none;
        padding: 12px;
    }

    div.stButton > button:hover {
        background: linear-gradient(
            90deg,
            #c77dff,
            #ff4d8d
        );
        color: white;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🎵 AI Music Genre Style Transfer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Transform your music using a Conditional Autoencoder'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    print("Loading ConditionalAutoencoder...")

    model = ConditionalAutoencoder(
        num_genres=5,
        latent_dim=256
    )

    dummy_x = tf.zeros(
        (1, 128, 1292, 1),
        dtype=tf.float32
    )

    dummy_genre = tf.zeros(
        (1,),
        dtype=tf.int32
    )

    # Build model
    model(
        [dummy_x, dummy_genre],
        training=False
    )

    print("Model architecture created successfully!")

    # Load trained weights
    model.load_weights(
        MODEL_PATH
    )

    print("Trained weights loaded successfully!")

    return model


try:
    model = load_model()
except Exception as e:
    st.error(f"Model load error: {e}")
    st.stop()


# ============================================================
# AUDIO PREPROCESSING
# ============================================================

def audio_to_mel(audio_path):

    audio, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        duration=DURATION,
        mono=True
    )

    if len(audio) == 0:
        raise ValueError("Audio file is empty.")

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

    mel_db = np.clip(
        mel_db,
        -80,
        0
    )

    mel_norm = (mel_db + 80) / 80

    mel_norm = mel_norm[:, :MAX_FRAMES]

    if mel_norm.shape[1] < MAX_FRAMES:
        mel_norm = np.pad(
            mel_norm,
            (
                (0, 0),
                (0, MAX_FRAMES - mel_norm.shape[1])
            )
        )

    return mel_norm.astype(np.float32)


# ============================================================
# MEL TO AUDIO
# ============================================================

def mel_to_audio(mel_norm):

    mel_norm = np.squeeze(mel_norm)

    mel_norm = np.clip(
        mel_norm,
        0,
        1
    )

    mel_db = (mel_norm * 80) - 80

    mel_power = librosa.db_to_power(
        mel_db
    )

    audio = librosa.feature.inverse.mel_to_audio(
        mel_power,
        sr=SAMPLE_RATE,
        n_fft=2048,
        hop_length=HOP_LENGTH,
        n_iter=32
    )

    return audio


# ============================================================
# SPECTROGRAM
# ============================================================

def show_spectrogram(mel, title):

    fig, ax = plt.subplots(
        figsize=(10, 4)
    )

    librosa.display.specshow(
        mel,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        ax=ax
    )

    ax.set_title(
        title,
        color="black"
    )

    fig.tight_layout()

    return fig


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🎛️ Settings")

    source_genre = st.selectbox(
        "Original Genre",
        GENRES,
        index=3
    )

    target_genre = st.selectbox(
        "Target Genre",
        GENRES,
        index=0
    )

    st.markdown("---")

    st.write("### 📌 Available Genres")

    for genre in GENRES:
        st.write(f"• {genre.title()}")

    st.markdown("---")

    st.caption(
        "AI Music Genre Style Transfer "
        "using a Conditional Autoencoder."
    )


# ============================================================
# UPLOAD
# ============================================================

st.markdown(
    '<div class="info-box">',
    unsafe_allow_html=True
)

st.subheader("🎧 Upload Your Music")

uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=[
        "wav",
        "mp3",
        "ogg",
        "flac",
        "m4a"
    ]
)

st.markdown(
    "</div>",
    unsafe_allow_html=True
)


# ============================================================
# AUDIO PREVIEW
# ============================================================

if uploaded_file is not None:

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )

    st.audio(
        uploaded_file,
        format=uploaded_file.type
    )


# ============================================================
# TRANSFORM
# ============================================================

if uploaded_file is not None:

    st.markdown("---")

    st.subheader("✨ Transform Your Music")

    st.write(
        f"**{source_genre.title()} → "
        f"{target_genre.title()}**"
    )

    if source_genre == target_genre:

        st.info(
            "Source and target genres are the same. "
            "You can still run the model."
        )

    transform_button = st.button(
        "🎵 Transform Music"
    )

    if transform_button:

        with st.spinner(
            "AI is transforming your music... 🎶"
        ):

            try:

                # ------------------------------------------------
                # SAVE UPLOADED FILE TEMPORARILY
                # ------------------------------------------------

                suffix = os.path.splitext(
                    uploaded_file.name
                )[1]

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix
                ) as temp_file:

                    temp_file.write(
                        uploaded_file.getbuffer()
                    )

                    temp_path = temp_file.name


                # ------------------------------------------------
                # PREPROCESS
                # ------------------------------------------------

                mel = audio_to_mel(
                    temp_path
                )

                model_input = np.expand_dims(
                    mel,
                    axis=(0, -1)
                )

                target_index = GENRES.index(
                    target_genre
                )

                genre_input = np.array(
                    [target_index],
                    dtype=np.int32
                )


                # ------------------------------------------------
                # MODEL PREDICTION
                # ------------------------------------------------

                generated_mel, genre_prediction = model.predict(
                    (
                        model_input,
                        genre_input
                    ),
                    verbose=0
                )


                generated_mel = generated_mel[0, :, :, 0]


                # ------------------------------------------------
                # GENERATED AUDIO
                # ------------------------------------------------

                generated_audio = mel_to_audio(
                    generated_mel
                )


                # ------------------------------------------------
                # SAVE OUTPUT
                # ------------------------------------------------

                output_name = (
                    f"{source_genre}_to_"
                    f"{target_genre}_generated.wav"
                )

                output_path = os.path.join(
                    OUTPUT_DIR,
                    output_name
                )

                sf.write(
                    output_path,
                    generated_audio,
                    SAMPLE_RATE
                )


                # ------------------------------------------------
                # SAVE SPECTROGRAM
                # ------------------------------------------------

                spectrogram_name = (
                    f"{source_genre}_to_"
                    f"{target_genre}_spectrogram.png"
                )

                spectrogram_path = os.path.join(
                    OUTPUT_DIR,
                    spectrogram_name
                )

                fig = show_spectrogram(
                    generated_mel,
                    f"{source_genre.title()} → "
                    f"{target_genre.title()}"
                )

                fig.savefig(
                    spectrogram_path,
                    dpi=150,
                    bbox_inches="tight"
                )

                plt.close(fig)


                # ------------------------------------------------
                # CLEAN TEMP FILE
                # ------------------------------------------------

                try:
                    os.remove(temp_path)
                except:
                    pass


                # ------------------------------------------------
                # RESULT
                # ------------------------------------------------

                st.success(
                    "🎉 Music transformation completed!"
                )

                st.markdown("---")

                col1, col2 = st.columns(2)

                with col1:

                    st.subheader(
                        "🎧 Original Music"
                    )

                    st.audio(
                        uploaded_file,
                        format=uploaded_file.type
                    )


                with col2:

                    st.subheader(
                        "🎵 Transformed Music"
                    )

                    st.audio(
                        output_path,
                        format="audio/wav"
                    )


                # ------------------------------------------------
                # SPECTROGRAM
                # ------------------------------------------------

                st.markdown("---")

                st.subheader(
                    "📊 Generated Mel-Spectrogram"
                )

                st.image(
                    spectrogram_path,
                    use_container_width=True
                )


                # ------------------------------------------------
                # DOWNLOAD
                # ------------------------------------------------

                st.markdown("---")

                with open(
                    output_path,
                    "rb"
                ) as audio_file:

                    st.download_button(
                        label="⬇️ Download Transformed Audio",
                        data=audio_file,
                        file_name=output_name,
                        mime="audio/wav"
                    )


                # ------------------------------------------------
                # CLASSIFIER RESULT
                # ------------------------------------------------

                st.markdown("---")

                st.subheader(
                    "🤖 Genre Classification"
                )

                probabilities = genre_prediction[0]

                for i, genre in enumerate(GENRES):

                    probability = (
                        float(probabilities[i])
                        * 100
                    )

                    st.write(
                        f"**{genre.title()}**: "
                        f"{probability:.2f}%"
                    )


            except Exception as e:

                st.error(
                    f"❌ Transformation failed: {e}"
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#ffd6e0;">
        🎵 AI Music Genre Style Transfer |
        Built with TensorFlow + Librosa + Streamlit
    </div>
    """,
    unsafe_allow_html=True
)