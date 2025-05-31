import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import logging

logging.basicConfig(level=logging.INFO)

X = np.load("features.npy")
y = np.load("labels.npy")
logging.info(f"Loaded features shape: {X.shape}, labels shape: {y.shape}")
logging.info(f"Class distribution: Male={np.sum(y==0)}, Female={np.sum(y==1)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
logging.info(f"Test Accuracy: {accuracy * 100:.2f}%")
logging.info(f"Classification Report:\n{classification_report(y_test, y_pred, target_names=['Male', 'Female'])}")

joblib.dump(model, "voice_classifier_model.pkl")
logging.info("Model saved as voice_classifier_model.pkl")