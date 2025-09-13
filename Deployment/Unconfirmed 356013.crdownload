from flask import Flask, request, jsonify, render_template
import numpy as np
import librosa
import tensorflow as tf
import joblib
import os

app = Flask(__name__)

# Load trained CNN model and label encoder
model = tf.keras.models.load_model(r"E:\New folder (2)\python\ML_model_Deploy_cnn\cnn2.keras")
labelencoder = joblib.load(r"E:\New folder (2)\python\ML_model_Deploy_cnn\labelencoder_cnn.pkl")

# Constants
SAMPLE_RATE = 22050
DURATION = 4
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
NUM_MELS = 128
EXPECTED_WIDTH = 173  # as used during training

# Mel spectrogram extraction (same as training)
def extract_mel_spectrogram(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    if len(audio) > SAMPLES_PER_TRACK:
        audio = audio[:SAMPLES_PER_TRACK]
    elif len(audio) < SAMPLES_PER_TRACK:
        audio = np.pad(audio, (0, SAMPLES_PER_TRACK - len(audio)))
    
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=NUM_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    if mel_db.shape[1] < EXPECTED_WIDTH:
        pad_width = EXPECTED_WIDTH - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
    elif mel_db.shape[1] > EXPECTED_WIDTH:
        mel_db = mel_db[:, :EXPECTED_WIDTH]

    return mel_db.astype(np.float32)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400

    file = request.files['file']
    file_path = "temp.wav"
    file.save(file_path)

    # Preprocess and predict
    mel_spec = extract_mel_spectrogram(file_path)
    mel_spec = mel_spec[..., np.newaxis]  # shape: (128, 173, 1)
    mel_spec = np.expand_dims(mel_spec, axis=0)  # shape: (1, 128, 173, 1)

    prediction = model.predict(mel_spec)
    predicted_index = np.argmax(prediction, axis=1)
    predicted_class = labelencoder.inverse_transform(predicted_index)[0]

    # Clean up
    os.remove(file_path)

    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
