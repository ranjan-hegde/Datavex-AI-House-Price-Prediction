"""
Data Loader Module
Loads housing dataset, cleans invalid records, and handles missing values
using documented imputation strategies.
"""

import os
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np

DEFAULT_CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "house_prices.csv"
)

REQUIRED_COLUMNS = ["area_sqft", "bedrooms", "bathrooms", "age_years", "price"]


def load_and_preprocess_data(
    csv_path: str = DEFAULT_CSV_PATH
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Loads CSV dataset, removes invalid entries, and imputes missing fields.

    Parameters:
        csv_path (str): File path to the CSV dataset.

    Returns:
        tuple: (X: pd.DataFrame, y: pd.Series, preprocessing_metadata: dict)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}. Run data/generate_data.py first.")

    # 1. Load CSV using Pandas
    df = pd.read_csv(csv_path)

    # Validate that all required columns are present in the dataset
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    total_raw_rows = len(df)

    # 2. Filter out invalid rows (domain constraints)
    # Area must be > 0, bedrooms >= 1, bathrooms >= 1, price > 0
    valid_mask = (
        (df["area_sqft"] > 0) &
        (df["bedrooms"] >= 1) &
        (df["bathrooms"] >= 1) &
        (df["price"] > 0)
    )
    df_clean = df[valid_mask].copy()

    # 3. Handle missing values: Documented Imputation for age_years
    # Compute median age from valid rows
    median_age = float(df_clean["age_years"].median())
    missing_age_count = int(df_clean["age_years"].isna().sum())

    # Fill missing age_years with median
    df_clean["age_years"] = df_clean["age_years"].fillna(median_age)

    # Drop any remaining rows with NaN in other columns (if any)
    df_clean = df_clean.dropna(subset=["area_sqft", "bedrooms", "bathrooms", "price"])
    clean_rows = len(df_clean)

    # 4. Separate feature matrix (X) and target variable (y)
    feature_cols = ["area_sqft", "bedrooms", "bathrooms", "age_years"]
    X = df_clean[feature_cols].copy()
    y = df_clean["price"].copy()

    metadata = {
        "raw_samples": total_raw_rows,
        "clean_samples": clean_rows,
        "dropped_rows": total_raw_rows - clean_rows,
        "missing_age_count": missing_age_count,
        "imputed_median_age": round(median_age, 2),
        "feature_names": feature_cols,
        "target_name": "price"
    }

    return X, y, metadata


if __name__ == "__main__":
    print("Testing data loader...")
    X, y, meta = load_and_preprocess_data()
    print("Data Loader Metadata:", meta)
    print("Features preview:")
    print(X.head())
    print("\nTarget preview:")
    print(y.head())

