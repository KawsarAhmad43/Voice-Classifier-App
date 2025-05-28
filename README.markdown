# Voice Classification Project

This project classifies voices as male or female using a machine learning model trained on the Kaggle Gender Recognition by Voice dataset. It features a Flask web app where users can record audio and predict gender.

## Requirements
- Python 3.11
- Browser: Chrome or Firefox
- Kaggle account for dataset
- Dependencies:
  ```bash
  pip install pandas numpy scikit-learn librosa flask joblib
  ```

## Dataset
Uses `voice.csv` from [Gender Recognition by Voice](https://www.kaggle.com/datasets/primaryobjects/voicegender) on Kaggle.
- Feature: `meanfun` (mean fundamental frequency in kHz)
- Labels: male, female
- Setup:
  1. Download `voicegender.zip` from Kaggle.
  2. Unzip and place in `data/`:
     ```bash
     mkdir -p data
     mv path/to/voice.csv data/
     ```

## Project Structure
```
voice_classifier/
├── data/
│   ├── voice.csv
├── src/
│   ├── preprocess.py
│   ├── train_model.py
│   ├── server.py
├── static/
│   ├── script.js
├── templates/
│   ├── index.html
│   └── result.html
├── features.npy
├── labels.npy
├── scaler.joblib
├── voice_classifier_model.pkl
```

## Setup Instructions
1. Navigate to project directory:
   ```bash
   cd ~/Desktop/ML\ Project\ practice/voice_classifier
   ```
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn librosa flask joblib
   ```
3. Verify dataset:
   ```bash
   ls data/voice.csv
   ```

## How to Run
1. Preprocess the dataset:
   ```bash
   python src/preprocess.py
   ```
   - Creates `features.npy`, `labels.npy`, `scaler.joblib`.

2. Train the model:
   ```bash
   python src/train_model.py
   ```
   - Creates `voice_classifier_model.pkl`.

3. Run the Flask server:
   ```bash
   python src/server.py
   ```

4. Open `http://127.0.0.1:5000` in a browser, record audio, and click "Predict".

### UI Overview:
![image](https://github.com/user-attachments/assets/80b7cdf1-08e0-4004-b106-26d18814f934)

![image](https://github.com/user-attachments/assets/b99e2b78-def0-4adb-a13f-d2a98fa2699f)

