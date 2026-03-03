import pandas as pd
import numpy as np


def create_ratio_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create financial ratio features."""
    
    # How large is the loan relative to income
    data['CREDIT_INCOME_RATIO'] = (data['AMT_CREDIT'] / 
                                    data['AMT_INCOME_TOTAL'])
    
    # How much of monthly income goes to repayment
    data['ANNUITY_INCOME_RATIO'] = (data['AMT_ANNUITY'] / 
                                     data['AMT_INCOME_TOTAL'])
    
    # How many years to fully repay the loan
    data['CREDIT_ANNUITY_RATIO'] = (data['AMT_CREDIT'] / 
                                     data['AMT_ANNUITY'])
    
    # Income per family member
    data['INCOME_PER_PERSON'] = (data['AMT_INCOME_TOTAL'] / 
                                  data['CNT_FAM_MEMBERS'])
    
    print("Ratio features created successfully")
    return data


def encode_categoricals(data: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode all categorical columns."""
    categorical_cols = data.select_dtypes(include=['object']).columns
    print(f"Encoding {len(categorical_cols)} categorical columns")
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    print(f"Shape after encoding: {data.shape}")
    return data


def run_feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    """Run full feature engineering pipeline."""
    data = create_ratio_features(data)
    data = encode_categoricals(data)
    return data