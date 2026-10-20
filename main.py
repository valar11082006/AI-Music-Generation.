import random
from pathlib import Path

import numpy as np
import pretty_midi
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ============================================
# AI MUSIC GENERATION USING AN LSTM MODEL
# ============================================

DATASET_DIR = Path("midi_dataset")
OUTPUT_DIR = Path("output")
MODEL_DIR = Path("models")

MODEL_PATH = MODEL_DIR / "music_lstm.keras"
OUTPUT_MIDI = OUTPUT_DIR / "generated_music.mid"

# Number of previous notes used to predict the next note
SEQUENCE_LENGTH = 40

# Training settings
EPOCHS = 30
BATCH_SIZE = 64

# Number of notes in the generated music
GENERATED_NOTES = 150

# MIDI pitch range: 0 to 127
VOCAB_SIZE = 128

# Temperature controls randomness:
# Lower = safer/more predictable
# Higher = more varied
TEMPERATURE = 0.8

# Fixed duration for generated notes, in seconds
NOTE_DURATION = 0.35


def load_midi_files():
    """Load MIDI files and extract their note pitches."""

    midi_files = list(DATASET_DIR.rglob("*.mid"))
    midi_files += list(DATASET_DIR.rglob("*.midi"))

    # Avoid processing duplicate paths
    midi_files = list(set(midi_files))

    if not midi_files:
        raise FileNotFoundError(
            "\nNo MIDI files found!\n"
            "Place .mid or .midi files inside the "
            "'midi_dataset' folder."
        )

    all_pitches = []

    print(f"Found {len(midi_files)} MIDI file(s).")

    for midi_file in midi_files:
        try:
            midi = pretty_midi.PrettyMIDI(str(midi_file))

            notes = []

            # Combine notes from non-drum instruments
            for instrument in midi.instruments:
                if instrument.is_drum:
                    continue

                notes.extend(instrument.notes)

            # Sort notes by starting time
            notes.sort(key=lambda note: (note.start, note.pitch))

            pitches = [
                int(note.pitch)
                for note in notes
                if 0 <= note.pitch < VOCAB_SIZE
            ]

            if pitches:
                all_pitches.extend(pitches)
                print(
                    f"Loaded {midi_file.name}: "
                    f"{len(pitches)} notes"
                )
            else:
                print(
                    f"Skipped {midi_file.name}: "
                    "no usable melodic notes"
                )

        except Exception as error:
            print(f"Could not read {midi_file}: {error}")

    if len(all_pitches) <= SEQUENCE_LENGTH:
        raise ValueError(
            f"Not enough notes for training. Found "
            f"{len(all_pitches)} usable notes, but more "
            f"than {SEQUENCE_LENGTH} are required. "
            "Add more MIDI music files."
        )

    return all_pitches


def prepare_sequences(pitches):
    """Create input sequences and next-note labels."""

    sequences = []
    targets = []

    for index in range(
        len(pitches) - SEQUENCE_LENGTH
    ):
        input_sequence = pitches[
            index:index + SEQUENCE_LENGTH
        ]

        next_note = pitches[
            index + SEQUENCE_LENGTH
        ]

        sequences.append(input_sequence)
        targets.append(next_note)

    X = np.array(sequences, dtype=np.int32)
    y = np.array(targets, dtype=np.int32)

    print(f"Training sequences: {len(X)}")

    return X, y


def build_model():
    """Create an LSTM model to predict the next note."""

    model = Sequential([
        tf.keras.Input(shape=(SEQUENCE_LENGTH,)),
        Embedding(
            input_dim=VOCAB_SIZE,
            output_dim=64
        ),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(128),
        Dropout(0.2),
        Dense(VOCAB_SIZE, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    return model


def train_model():
    """Load MIDI data, train the LSTM, and save it."""

    print("\nSTEP 1: Loading MIDI dataset")

    pitches = load_midi_files()

    print("\nSTEP 2: Preparing note sequences")

    X, y = prepare_sequences(pitches)

    print("\nSTEP 3: Building LSTM model")

    model = build_model()

    print("\nSTEP 4: Training the model")

    early_stopping = EarlyStopping(
        monitor="loss",
        patience=5,
        restore_best_weights=True
    )

    model.fit(
        X,
        y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model.save(MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")

    return model, pitches


def sample_next_note(predictions, temperature=0.8):
    """Select a note using temperature-based sampling."""

    predictions = np.asarray(
        predictions,
        dtype=np.float64
    )

    # Avoid taking the logarithm of zero
    predictions = np.log(
        np.maximum(predictions, 1e-8)
    ) / max(temperature, 0.01)

    # Convert values into probabilities
    exp_predictions = np.exp(
        predictions - np.max(predictions)
    )

    probabilities = exp_predictions / np.sum(
        exp_predictions
    )

    return int(
        np.random.choice(
            len(probabilities),
            p=probabilities
        )
    )


def generate_music(model, seed_pitches):
    """Generate new notes and save them as a MIDI file."""

    print("\nSTEP 5: Generating new music")

    # Choose a random starting sequence from the dataset
    start = random.randint(
        0,
        len(seed_pitches) - SEQUENCE_LENGTH - 1
    )

    pattern = list(
        seed_pitches[start:start + SEQUENCE_LENGTH]
    )

    generated = []

    for step in range(GENERATED_NOTES):
        model_input = np.array(
            [pattern[-SEQUENCE_LENGTH:]],
            dtype=np.int32
        )

        probabilities = model.predict(
            model_input,
            verbose=0
        )[0]

        next_pitch = sample_next_note(
            probabilities,
            temperature=TEMPERATURE
        )

        generated.append(next_pitch)
        pattern.append(next_pitch)

        if (step + 1) % 25 == 0:
            print(
                f"Generated {step + 1}/"
                f"{GENERATED_NOTES} notes"
            )

    print("\nSTEP 6: Converting notes to MIDI")

    # Create a MIDI instrument
    instrument = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program(
            "Acoustic Grand Piano"
        )
    )

    # Convert each generated pitch into a MIDI note
    current_time = 0.0

    for pitch in generated:
        note = pretty_midi.Note(
            velocity=90,
            pitch=int(pitch),
            start=current_time,
            end=current_time + NOTE_DURATION
        )

        instrument.notes.append(note)
        current_time += NOTE_DURATION

    midi_output = pretty_midi.PrettyMIDI()
    midi_output.instruments.append(instrument)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    midi_output.write(str(OUTPUT_MIDI))

    print("\nMusic generated successfully!")
    print(f"Output MIDI file: {OUTPUT_MIDI}")
    print("Open the MIDI file in a MIDI player to listen.")


def main():
    # Ensure repeatable initial setup where possible
    np.random.seed(42)
    tf.random.set_seed(42)

    print("=" * 45)
    print("   AI MUSIC GENERATION USING LSTM")
    print("=" * 45)

    # Train a new model and generate music
    model, pitches = train_model()

    generate_music(model, pitches)


if __name__ == "__main__":
    main()
