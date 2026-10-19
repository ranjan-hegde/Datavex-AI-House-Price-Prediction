"""
Inference & Prediction Module
Loads trained model artifact, validates input data, and produces price estimations.
Directly implements Requirements 6 & 7 and handles test cases TC1 through TC4.
"""

import os
import sys
import json
import joblib
import pandas as pd
from typing import Dict, Any, Union

# Ensure project root is in sys.path for direct execution
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.validator import validate_property_input


DEFAULT_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "house_price_model.joblib"
)


class PricePredictor:
    """
    Service class to load trained model artifact and perform price predictions.
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at: {model_path}. Please train the model first by running src/train.py"
            )
        self.artifact = joblib.load(model_path)
        self.model = self.artifact["model"]
        self.model_name = self.artifact.get("model_name", "RegressionModel")
        self.feature_names = self.artifact.get("feature_names", ["area_sqft", "bedrooms", "bathrooms", "age_years"])
        self.imputed_median_age = self.artifact.get("imputed_median_age", 20.0)

    def predict(
        self,
        property_data: Dict[str, Any],
        allow_imputation: bool = True
    ) -> Dict[str, Any]:
        """
        Validates input details and estimates the property price.

        Parameters:
            property_data (dict): Dictionary with keys: area_sqft, bedrooms, bathrooms, and optional age_years.
            allow_imputation (bool): If True, imputes missing age_years using training median.

        Returns:
            dict: Formatted prediction output matching example requirements:
                  {"predicted_price": 4500000, "details": {...}}

        Raises:
            ValueError: If input fails validation (negative area, 0 bedrooms, etc.).
        """
        # 1. Validate property features (enforces TC2, TC3, TC4)
        clean_input, notice = validate_property_input(
            property_data,
            default_age_imputation=self.imputed_median_age,
            allow_imputation=allow_imputation
        )

        # 2. Convert to DataFrame with correct column ordering
        input_df = pd.DataFrame([clean_input], columns=self.feature_names)

        # 3. Model inference
        raw_prediction = self.model.predict(input_df)[0]
        # Round price to nearest 1,000 for realistic currency valuation
        final_price = int(round(raw_prediction, -3))

        response = {
            "predicted_price": final_price,
            "inputs_used": clean_input
        }
        if notice:
            response["notice"] = notice

        return response


def predict_price(property_data: Dict[str, Any], model_path: str = DEFAULT_MODEL_PATH) -> Dict[str, Any]:
    """
    Convenience function to predict price for a property dictionary.
    """
    predictor = PricePredictor(model_path=model_path)
    return predictor.predict(property_data)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Predict house price from property details.")
    parser.add_argument("--area", type=float, help="Property area in square feet (e.g. 1200)")
    parser.add_argument("--bedrooms", type=int, help="Number of bedrooms (e.g. 2)")
    parser.add_argument("--bathrooms", type=int, help="Number of bathrooms (e.g. 2)")
    parser.add_argument("--age", type=float, default=None, help="Age of house in years (optional)")
    parser.add_argument("--json", type=str, default=None, help="Property details as JSON string")

    args = parser.parse_args()

    if args.json:
        try:
            prop = json.loads(args.json)
        except json.JSONDecodeError as e:
            print(json.dumps({"error": f"Invalid JSON string: {e}"}, indent=2))
            return
    elif args.area is not None and args.bedrooms is not None and args.bathrooms is not None:
        prop = {
            "area_sqft": args.area,
            "bedrooms": args.bedrooms,
            "bathrooms": args.bathrooms,
        }
        if args.age is not None:
            prop["age_years"] = args.age
    else:
        # Default example input as defined in Problem Statement
        print("No input provided. Running with example input from Problem Statement:")
        prop = {
            "area_sqft": 1200,
            "bedrooms": 2,
            "bathrooms": 2,
            "age_years": 5
        }

    print("Input:", json.dumps(prop, indent=2))
    try:
        result = predict_price(prop)
        print("\nOutput:")
        print(json.dumps(result, indent=2))
    except (ValueError, TypeError) as err:
        print(f"\n[Rejected Input Error]: {err}")


if __name__ == "__main__":
    main()

