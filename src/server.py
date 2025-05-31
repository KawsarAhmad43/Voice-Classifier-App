from flask import Flask, request, render_template, send_from_directory
import librosa
import numpy as np
import joblib
import os
from datetime import datetime
import logging

logging.basicConfig(level=logging.DEBUG)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, '..', 'templates')
STATIC_DIR = os.path.join(BASE_DIR, '..', 'static')

app = Flask(__name__, static_folder=STATIC_DIR, template_folder=TEMPLATE_DIR)
app.logger.info(f"Template folder: {os.path.abspath(TEMPLATE_DIR)}")
app.logger.info(f"Static folder: {os.path.abspath(STATIC_DIR)}")

index_path = os.path.join(TEMPLATE_DIR, 'index.html')
result_path = os.path.join(TEMPLATE_DIR, 'result.html')
if os.path.exists(index_path) and os.access(index_path, os.R_OK):
    app.logger.info(f"index.html found and readable at {index_path}")
else:
    app.logger.error(f"index.html not found or not readable at {index_path}")
if os.path.exists(result_path) and os.access(result_path, os.R_OK):
    app.logger.info(f"result.html found and readable at {result_path}")
else:
    app.logger.error(f"result.html not found or not readable at {result_path}")

script_path = os.path.join(STATIC_DIR, 'script.js')
if os.path.exists(script_path) and os.access(script_path, os.R_OK):
    app.logger.info(f"script.js found and readable at {script_path}")
else:
    app.logger.error(f"script.js not found or not readable at {script_path}")

try:
    model = joblib.load("voice_classifier_model.pkl")
    scaler = joblib.load("scaler.joblib")
    app.logger.info("Model and scaler loaded successfully")
except Exception as e:
    app.logger.error(f"Failed to load model or scaler: {str(e)}")
    raise

def extract_meanfun(audio_data):
    temp_path = f"temp_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav"
    with open(temp_path, 'wb') as f:
        f.write(audio_data)
    y, sr = librosa.load(temp_path, sr=16000, mono=True)
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
    valid_f0 = f0[voiced_flag]
    meanfun = np.mean(valid_f0) / 1000 if len(valid_f0) > 0 else 0.1
    app.logger.debug(f"Valid F0 count: {len(valid_f0)}, Meanfun: {meanfun}, Min F0: {np.min(valid_f0)/1000 if len(valid_f0) > 0 else 'N/A'}, Max F0: {np.max(valid_f0)/1000 if len(valid_f0) > 0 else 'N/A'}")
    os.remove(temp_path)
    return meanfun

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        app.logger.error(f"Template rendering failed for index.html: {str(e)}")
        try:
            return send_from_directory(TEMPLATE_DIR, 'index.html')
        except Exception as static_e:
            app.logger.error(f"Static file serving failed: {str(static_e)}")
            return f"Error: Cannot find or render index.html: {str(e)}", 500

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return "No audio file provided", 400
    audio_file = request.files['audio']
    meanfun = extract_meanfun(audio_file.read())
    meanfun_scaled = scaler.transform([[meanfun]])[0]
    app.logger.debug(f"Extracted meanfun: {meanfun}, Scaled meanfun: {meanfun_scaled}, Prediction input: {meanfun_scaled}")
    prediction = model.predict([meanfun_scaled])[0]
    gender = "Female" if prediction == 1 else "Male"
    app.logger.info(f"Predicted gender: {gender} for meanfun: {meanfun}")
    try:
        return render_template('result.html', gender=gender)
    except Exception as e:
        app.logger.error(f"Template rendering failed for result.html: {str(e)}")
        return f"Error rendering result.html: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)