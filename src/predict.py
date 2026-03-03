import pandas as pd
import numpy as np
import joblib
import re
from pathlib import Path


def load_model(models_path: Path):
    """Load the saved best model."""
    model = joblib.load(models_path / "best_model.pkl")
    print("Model loaded successfully")
    return model


def preprocess_input(input_data: dict, 
                     reference_columns: list) -> pd.DataFrame:
    """
    Preprocess a single borrower input for prediction.
    Aligns input with training feature columns.
    """
    df = pd.DataFrame([input_data])
    
    # Add missing columns with 0
    for col in reference_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Keep only training columns in correct order
    df = df[reference_columns]
    
    # Clean column names
    df.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', col) 
                  for col in df.columns]
    return df


def predict_default_probability(model, input_df: pd.DataFrame) -> float:
    """Return probability of default for a borrower."""
    proba = model.predict_proba(input_df)[:, 1][0]
    return round(proba, 4)


def get_risk_band(probability: float) -> str:
    """Assign a risk band based on default probability."""
    if probability < 0.2:
        return "Very Low Risk ✓"
    elif probability < 0.4:
        return "Low Risk ✓"
    elif probability < 0.6:
        return "Medium Risk ⚠"
    elif probability < 0.8:
        return "High Risk ✗"
    else:
        return "Very High Risk ✗"


if __name__ == "__main__":
    BASE_DIR = Path().resolve()
    model = load_model(BASE_DIR / "models")
    print("Predict module ready")