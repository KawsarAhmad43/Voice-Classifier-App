import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import logging

logging.basicConfig(level=logging.INFO)

def create_dataset(csv_path):
    logging.info(f"Loading dataset from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Balance dataset
    min_count = df['label'].value_counts().min()
    df = df.groupby('label').sample(min_count, random_state=42)
    logging.info(f"Balanced dataset: {df['label'].value_counts().to_dict()}")
    
    features = df[['meanfun']].values  # Use meanfun (in kHz)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    labels = df['label'].apply(lambda x: 1 if x.lower() == 'female' else 0).values
    logging.info(f"Meanfun stats: mean={features.mean():.3f}, std={features.std():.3f}")
    
    # Save scaler using joblib
    joblib.dump(scaler, "scaler.joblib")
    return features_scaled, labels

if __name__ == "__main__":
    csv_path = "data/voice.csv"
    X, y = create_dataset(csv_path)
    np.save("features.npy", X)
    np.save("labels.npy", y)
    logging.info("Dataset created: features.npy, labels.npy, scaler.joblib")