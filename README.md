This project uses deep learning to generate new musical
note sequences from MIDI music files using an LSTM model.

- Collect MIDI music data.
- Preprocess MIDI notes into training sequences.
- Build and train an LSTM neural network.
- Generate new musical note sequences.
- Export the generated music as a MIDI file.

- Python
- TensorFlow / Keras
- NumPy
- PrettyMIDI
- LSTM Deep Learning


AI-Music-Generation/
- main.py
- requirements.txt
- README.md
- .gitignore
- midi_dataset/
- output/
- models/

Install the required dependencies:

pip install -r requirements.txt

python main.py

Place legally available .mid or .midi files
inside the midi_dataset folder.


The generated music is saved to:

output/generated_music.mid

The trained model is saved to:

models/music_lstm.keras

The LSTM model predicts the next MIDI pitch using
a sequence of previous pitches.

This beginner implementation generates a monophonic
sequence with fixed note durations. It does not
generate complete multi-instrument arrangements.

valarmathi
