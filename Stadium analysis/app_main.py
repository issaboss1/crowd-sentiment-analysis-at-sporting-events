from flask import Flask, render_template, request, send_from_directory
import numpy as np
import librosa
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
import joblib

# Initialize the Flask application
app = Flask(__name__)

# Load your trained model (replace with actual model path)
model = tf.keras.models.load_model('sentiment_cnn_model.h5')
le = joblib.load(open('label_encoder.pkl','rb'))

# Audio upload path
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to extract features from audio
def extract_features_from_signal(signal, sr):
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

def segment_audio(audio, sr, segment_duration=2.0, overlap=0.5):
    segment_length = int(segment_duration * sr)
    step = int(segment_length * (1 - overlap))

    segments = []
    timestamps = []

    for start in range(0, len(audio) - segment_length, step):
        end = start + segment_length
        segments.append(audio[start:end])
        timestamps.append((round(start / sr, 2), round(end / sr, 2)))

    return segments, timestamps

# Function to predict sentiment
def predict_sentiment_timeline(audio_path):
    audio, sr = librosa.load(audio_path, sr=22050)

    segments, times = segment_audio(audio, sr)

    features = []
    for seg in segments:
        feat = extract_features_from_signal(seg, sr)
        features.append(feat)

    X = np.array(features)
    X = X.reshape(X.shape[0], X.shape[1], 1, 1)

    preds = model.predict(X)
    labels = le.inverse_transform(np.argmax(preds, axis=1))

    results = []
    for t, l in zip(times, labels):
        results.append({
            "start": t[0],
            "end": t[1],
            "emotion": l
        })

    return results


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get the uploaded file
        file = request.files["audio"]
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            # Predict sentiment
            results = predict_sentiment_timeline(file_path)

            # Return the result to the frontend
            return render_template(
              "index.html",
               results=results,
               audio_path=filename
            )



    return render_template("index.html", sentiment=None, audio_path=None)

# Route to serve uploaded files
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)



if __name__ == "__main__":
    app.run(debug=True)
