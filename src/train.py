"""
Model Training & Evaluation Module
Trains Linear Regression and Random Forest Regressor models on housing data.
Evaluates models using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
Saves the optimal model artifact along with preprocessing metadata.
"""

import os
import sys

# Ensure project root is in sys.path for direct script execution
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from src.data_loader import load_and_preprocess_data


MODEL_SAVE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "house_price_model.joblib"
)


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Evaluates a trained regression model on test data using MAE, RMSE, and R2.

    Parameters:
        model: Trained scikit-learn regressor.
        X_test (pd.DataFrame): Test feature set.
        y_test (pd.Series): Actual test target values.

    Returns:
        dict: Dictionary containing MAE, RMSE, and R2 metrics.
    """
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return {
        "MAE": round(float(mae), 2),
        "RMSE": round(float(rmse), 2),
        "R2_Score": round(float(r2), 4)
    }


def train_models(
    test_size: float = 0.2,
    random_state: int = 42,
    save_artifact: bool = True
) -> Tuple[Dict[str, Any], str]:
    """
    Executes the complete training workflow:
    1. Loads clean data and imputation metadata.
    2. Splits data into 80% train / 20% test sets.
    3. Trains Linear Regression and Random Forest Regressor.
    4. Evaluates both models using MAE and RMSE.
    5. Saves the winning model package to disk.

    Returns:
        tuple: (results_dict: dict, best_model_name: str)
    """
    print("=" * 60)
    print("           HOUSE PRICE PREDICTION - MODEL TRAINING           ")
    print("=" * 60)

    # 1. Load preprocessed data
    print("\n[Step 1/5] Loading & preprocessing dataset...")
    X, y, meta = load_and_preprocess_data()
    print(f"Loaded {len(X)} records with features: {list(X.columns)}")
    print(f"Imputation rule for missing age: median = {meta['imputed_median_age']} years")

    # 2. Train-Test Split (Requirement 3)
    print(f"\n[Step 2/5] Splitting data into Train ({int((1-test_size)*100)}%) and Test ({int(test_size*100)}%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Training set: {X_train.shape[0]} samples | Testing set: {X_test.shape[0]} samples")

    # 3. Model 1: Linear Regression
    print("\n[Step 3/5] Training Model 1: Linear Regression...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_metrics = evaluate_model(lr_model, X_test, y_test)
    print(f"  > Linear Regression MAE  : INR {lr_metrics['MAE']:,.2f}")
    print(f"  > Linear Regression RMSE : INR {lr_metrics['RMSE']:,.2f}")
    print(f"  > Linear Regression R^2  : {lr_metrics['R2_Score']:.4f}")

    # 4. Model 2: Random Forest Regressor
    print("\n[Step 4/5] Training Model 2: Random Forest Regressor...")
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=random_state)
    rf_model.fit(X_train, y_train)
    rf_metrics = evaluate_model(rf_model, X_test, y_test)
    print(f"  > Random Forest MAE  : INR {rf_metrics['MAE']:,.2f}")
    print(f"  > Random Forest RMSE : INR {rf_metrics['RMSE']:,.2f}")
    print(f"  > Random Forest R^2  : {rf_metrics['R2_Score']:.4f}")

    # 5. Model Selection based on lowest RMSE
    models = {
        "LinearRegression": {"model": lr_model, "metrics": lr_metrics},
        "RandomForestRegressor": {"model": rf_model, "metrics": rf_metrics}
    }

    # Select best model (lowest RMSE)
    best_name = min(models.keys(), key=lambda name: models[name]["metrics"]["RMSE"])
    best_bundle = models[best_name]

    print("\n" + "-" * 60)
    print(f" SELECTED BEST MODEL: {best_name} (Lowest RMSE: INR {best_bundle['metrics']['RMSE']:,.2f})")
    print("-" * 60)

    # 6. Save Model Artifact (Requirement 4 & Acceptance Criteria)
    if save_artifact:
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        artifact = {
            "model_name": best_name,
            "model": best_bundle["model"],
            "metrics": best_bundle["metrics"],
            "all_metrics": {k: v["metrics"] for k, v in models.items()},
            "feature_names": meta["feature_names"],
            "imputed_median_age": meta["imputed_median_age"],
            "training_samples": len(X_train),
            "test_samples": len(X_test)
        }
        joblib.dump(artifact, MODEL_SAVE_PATH)
        print(f"[Step 5/5] Saved model artifact to: {MODEL_SAVE_PATH}")

    return {
        "LinearRegression": lr_metrics,
        "RandomForestRegressor": rf_metrics,
        "BestModel": best_name,
        "BestMetrics": best_bundle["metrics"]
    }, best_name


if __name__ == "__main__":
    train_models()
