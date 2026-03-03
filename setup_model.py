import os
import pandas as pd
from pathlib import Path
import joblib
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
DATA_PATH = BASE_DIR / "data" / "processed" / "processed_sample.csv"

def train_and_save():
    print(f"Loading data from {DATA_PATH}...")
    data = pd.read_csv(DATA_PATH)
    data.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in data.columns]
    print(f"Data shape: {data.shape}")

    X = data.drop(['TARGET', 'SK_ID_CURR'], axis=1, errors='ignore')
    y = data['TARGET']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training RandomForest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    os.makedirs(BASE_DIR / "models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    y_pred = model.predict_proba(X_test)[:, 1]
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred):.4f}")

if __name__ == "__main__":
    train_and_save()