import os
import numpy as np
import librosa

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

SAMPLE_RATE = 22050
DURATION = 30
N_MELS = 128
HOP_LENGTH = 512

GENRES = ["classical", "jazz", "metal", "pop", "rock"]


def audio_to_mel(audio_path):
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


def process_genre(genre):
    input_dir = os.path.join(RAW_DIR, genre)
    output_dir = os.path.join(PROCESSED_DIR, genre)

    os.makedirs(output_dir, exist_ok=True)

    audio_files = [
        file for file in os.listdir(input_dir)
        if file.lower().endswith(".wav")
    ]

    print(f"\nProcessing {genre}: {len(audio_files)} files")

    successful = 0

    for index, filename in enumerate(audio_files, start=1):
        input_path = os.path.join(input_dir, filename)

        output_filename = os.path.splitext(filename)[0] + ".npy"
        output_path = os.path.join(output_dir, output_filename)

        try:
            mel = audio_to_mel(input_path)
            np.save(output_path, mel)

            successful += 1

            print(
                f"[{index}/{len(audio_files)}] "
                f"{filename} -> {mel.shape}"
            )

        except Exception as error:
            print(f"ERROR processing {filename}: {error}")

    print(
        f"Completed {genre}: "
        f"{successful}/{len(audio_files)} files"
    )


def main():
    print("=" * 60)
    print("MUSIC GENRE TRANSFER - AUDIO PREPROCESSING")
    print("=" * 60)

    for genre in GENRES:
        process_genre(genre)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()