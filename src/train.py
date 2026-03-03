import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE


def load_data(processed_path: Path):
    """Load processed data and split into X and y."""
    import re
    data = pd.read_csv(processed_path / "processed_train.csv")
    data.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', col) 
                    for col in data.columns]
    
    X = data.drop(['TARGET', 'SK_ID_CURR'], axis=1)
    y = data['TARGET']
    return X, y


def split_data(X, y):
    """Split data into train and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train):
    """Apply SMOTE to balance training data."""
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
    
    X_balanced = pd.DataFrame(X_balanced, columns=X_train.columns)
    print(f"After SMOTE: {pd.Series(y_balanced).value_counts().to_dict()}")
    return X_balanced, y_balanced


def train_models(X_train, y_train, X_test, y_test):
    """Train all models and return results."""
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(
            n_estimators=100, random_state=42, 
            eval_metric='auc', verbosity=0),
        'LightGBM': LGBMClassifier(
            n_estimators=100, random_state=42, verbose=-1)
    }
    
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        results[name] = {'model': model, 'auc': auc, 'proba': y_proba}
        print(f"  ROC-AUC: {auc:.4f}")
    
    return results


def save_best_model(results: dict, models_path: Path):
    """Save the best performing model."""
    import os
    os.makedirs(models_path, exist_ok=True)
    
    best_name = max(results, key=lambda x: results[x]['auc'])
    best_model = results[best_name]['model']
    
    joblib.dump(best_model, models_path / "best_model.pkl")
    print(f"Best model saved: {best_name} "
          f"(AUC: {results[best_name]['auc']:.4f})")
    return best_name, best_model


if __name__ == "__main__":
    BASE_DIR = Path().resolve()
    
    print("Loading data...")
    X, y = load_data(BASE_DIR / "data" / "processed")
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    print("Applying SMOTE...")
    X_train_bal, y_train_bal = apply_smote(X_train, y_train)
    
    print("Training models...")
    results = train_models(X_train_bal, y_train_bal, X_test, y_test)
    
    print("Saving best model...")
    save_best_model(results, BASE_DIR / "models")