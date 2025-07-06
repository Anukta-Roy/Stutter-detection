from flask import Flask, request, render_template
import numpy as np
import librosa
import joblib
import os

app = Flask(__name__)

# Load models
rf_model_Stutter = joblib.load('models/rf_model_stutter.pkl')
rf_model_prolongation = joblib.load('models/rf_model_prolongation.pkl')
rf_model_block = joblib.load('models/rf_model_block.pkl')
rf_model_WordRep = joblib.load('models/rf_model_wordrep.pkl')
rf_model_SoundRep = joblib.load('models/rf_model_soundrep.pkl')
rf_model_Interjection = joblib.load('models/rf_model_interjection.pkl')

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    jitter = np.std(y)  # placeholder
    shimmer = np.max(y) - np.min(y)  # placeholder
    features = np.concatenate([mfcc, [zcr], [jitter], [shimmer]])
    return features.reshape(1, -1)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_message = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"

        file_path = os.path.join('uploads', file.filename)
        os.makedirs('uploads', exist_ok=True)
        file.save(file_path)

        features = extract_features(file_path)

        stutter = rf_model_Stutter.predict(features)[0]
        types = []

        if stutter:
            # Check specific types
            if rf_model_prolongation.predict(features)[0]:
                types.append("Prolongation")
            if rf_model_block.predict(features)[0]:
                types.append("Block")
            if rf_model_WordRep.predict(features)[0]:
                types.append("Word Repetition")
            if rf_model_SoundRep.predict(features)[0]:
                types.append("Sound Repetition")
            if rf_model_Interjection.predict(features)[0]:
                types.append("Interjection")
            
            prediction_message = "This audio has Stutter."
            if types:
                prediction_message += f"<br> Type: {', '.join(types)}."
        else:
            prediction_message = "This audio doesn't have Stutter."

    return render_template('index.html', prediction_message=prediction_message)

if __name__ == '__main__':
    app.run(debug=True)
