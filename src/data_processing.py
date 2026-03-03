import pandas as pd
import numpy as np
from pathlib import Path
import os
import re


def load_raw_data(raw_path: Path):
    """Load all raw datasets."""
    app_train = pd.read_csv(raw_path / "application_train.csv")
    bureau = pd.read_csv(raw_path / "bureau.csv")
    installments = pd.read_csv(raw_path / "installments_payments.csv")
    
    print(f"Application Train: {app_train.shape}")
    print(f"Bureau: {bureau.shape}")
    print(f"Installments: {installments.shape}")
    
    return app_train, bureau, installments


def aggregate_bureau(bureau: pd.DataFrame) -> pd.DataFrame:
    """Aggregate bureau data to one row per applicant."""
    bureau_agg = bureau.groupby('SK_ID_CURR').agg(
        BUREAU_LOAN_COUNT              = ('SK_ID_BUREAU', 'count'),
        BUREAU_CREDIT_DAY_OVERDUE_MEAN = ('CREDIT_DAY_OVERDUE', 'mean'),
        BUREAU_AMT_CREDIT_SUM_MEAN     = ('AMT_CREDIT_SUM', 'mean'),
        BUREAU_AMT_CREDIT_SUM_MAX      = ('AMT_CREDIT_SUM', 'max'),
        BUREAU_ACTIVE_LOANS            = ('CREDIT_ACTIVE', 
                                          lambda x: (x == 'Active').sum())
    ).reset_index()
    
    print(f"Bureau aggregated: {bureau_agg.shape}")
    return bureau_agg


def aggregate_installments(installments: pd.DataFrame) -> pd.DataFrame:
    """Aggregate installments data to one row per applicant."""
    installments['DAYS_LATE'] = (installments['DAYS_ENTRY_PAYMENT'] - 
                                  installments['DAYS_INSTALMENT'])
    installments['DAYS_LATE'] = installments['DAYS_LATE'].clip(lower=0)
    
    installments_agg = installments.groupby('SK_ID_CURR').agg(
        INSTALLMENTS_COUNT            = ('SK_ID_PREV', 'count'),
        INSTALLMENTS_AMT_PAYMENT_MEAN = ('AMT_PAYMENT', 'mean'),
        INSTALLMENTS_AMT_PAYMENT_MAX  = ('AMT_PAYMENT', 'max'),
        INSTALLMENTS_DAYS_LATE_MEAN   = ('DAYS_LATE', 'mean'),
        INSTALLMENTS_DAYS_LATE_MAX    = ('DAYS_LATE', 'max')
    ).reset_index()
    
    print(f"Installments aggregated: {installments_agg.shape}")
    return installments_agg


def merge_tables(app_train: pd.DataFrame, 
                 bureau_agg: pd.DataFrame, 
                 installments_agg: pd.DataFrame) -> pd.DataFrame:
    """Merge all tables into one."""
    data = app_train.merge(bureau_agg, on='SK_ID_CURR', how='left')
    data = data.merge(installments_agg, on='SK_ID_CURR', how='left')
    print(f"Merged shape: {data.shape}")
    return data


def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """Drop high missing columns and impute the rest."""
    cols_to_drop = [col for col in data.columns 
                    if data[col].isnull().mean() > 0.5]
    print(f"Dropping {len(cols_to_drop)} columns with >50% missing")
    data = data.drop(columns=cols_to_drop)
    
    for col in data.columns:
        if data[col].dtype in ['float64', 'int64']:
            data[col] = data[col].fillna(data[col].median())
        else:
            data[col] = data[col].fillna(data[col].mode()[0])
    
    print(f"Missing values remaining: {data.isnull().sum().sum()}")
    return data


def clean_column_names(data: pd.DataFrame) -> pd.DataFrame:
    """Remove special characters from column names."""
    data.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', col) 
                    for col in data.columns]
    return data


def save_processed_data(data: pd.DataFrame, processed_path: Path):
    """Save processed data to disk."""
    os.makedirs(processed_path, exist_ok=True)
    data.to_csv(processed_path / "processed_train.csv", index=False)
    print(f"Saved: {processed_path / 'processed_train.csv'}")