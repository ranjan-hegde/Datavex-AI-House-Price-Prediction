"""
House Price Prediction System - CLI Entry Point
Provides a unified interface to:
1. Generate synthetic housing dataset: `python main.py generate-data`
2. Train and evaluate regression models: `python main.py train`
3. Run official test cases (TC1-TC5): `python main.py test`
4. Predict property price interactively or via JSON: `python main.py predict`
5. Run full end-to-end automated pipeline: `python main.py demo`
"""

import os
import sys
import json
import argparse

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.generate_data import main as run_data_generation
from src.train import train_models
from src.predict import PricePredictor, predict_price
from tests.test_cases import run_all_tests_with_report


def cmd_generate_data(args):
    print(">>> Generating Synthetic Dataset...")
    run_data_generation()


def cmd_train(args):
    print(">>> Training Machine Learning Models...")
    train_models()


def cmd_test(args):
    print(">>> Executing Test Cases...")
    run_all_tests_with_report()


def cmd_predict(args):
    predictor = PricePredictor()

    if args.json:
        try:
            prop = json.loads(args.json)
        except json.JSONDecodeError as e:
            print(f"[Error] Invalid JSON string provided: {e}")
            return
    elif args.area is not None and args.bedrooms is not None and args.bathrooms is not None:
        prop = {
            "area_sqft": args.area,
            "bedrooms": args.bedrooms,
            "bathrooms": args.bathrooms
        }
        if args.age is not None:
            prop["age_years"] = args.age
    else:
        # Interactive mode
        print("\n--- Interactive House Price Prediction ---")
        try:
            area_str = input("Enter property area (sqft, e.g. 1200): ").strip()
            bed_str = input("Enter number of bedrooms (e.g. 2): ").strip()
            bath_str = input("Enter number of bathrooms (e.g. 2): ").strip()
            age_str = input("Enter house age in years (press Enter to skip for default imputation): ").strip()

            prop = {
                "area_sqft": float(area_str),
                "bedrooms": int(bed_str),
                "bathrooms": int(bath_str)
            }
            if age_str:
                prop["age_years"] = float(age_str)
        except (ValueError, TypeError) as e:
            print(f"[Error] Invalid numerical input: {e}")
            return

    print("\n[Input Received]:")
    print(json.dumps(prop, indent=2))

    try:
        result = predictor.predict(prop)
        print("\n[Prediction Result]:")
        print(json.dumps(result, indent=2))
        print(f"\n>> Estimated Market Price: INR {result['predicted_price']:,} <<")
    except (ValueError, TypeError) as err:
        print(f"\n[Validation Rejected]: {err}")


def cmd_demo(args):
    print("=" * 70)
    print("              FULL DEMO - HOUSE PRICE PREDICTION PIPELINE              ")
    print("=" * 70)

    print("\n[STEP 1]: Generating Dataset...")
    run_data_generation()

    print("\n[STEP 2]: Training & Evaluating Regression Models...")
    train_models()

    print("\n[STEP 3]: Testing Example Prediction (TC1)...")
    sample_input = {"area_sqft": 1200, "bedrooms": 2, "bathrooms": 2, "age_years": 5}
    pred = predict_price(sample_input)
    print("Input:", sample_input)
    print("Prediction Output:", json.dumps(pred, indent=2))

    print("\n[STEP 4]: Running All Official Test Cases (TC1-TC5)...")
    run_all_tests_with_report()

    print("\n Pipeline execution completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="House Price Prediction ML Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: generate-data
    subparsers.add_parser("generate-data", help="Generate synthetic house price CSV dataset")

    # Command: train
    subparsers.add_parser("train", help="Train and evaluate regression models")

    # Command: test
    subparsers.add_parser("test", help="Run automated test suite (TC1-TC5)")

    # Command: demo
    subparsers.add_parser("demo", help="Run complete end-to-end pipeline")

    # Command: predict
    pred_parser = subparsers.add_parser("predict", help="Predict house price")
    pred_parser.add_argument("--area", type=float, help="Area in square feet")
    pred_parser.add_argument("--bedrooms", type=int, help="Bedrooms")
    pred_parser.add_argument("--bathrooms", type=int, help="Bathrooms")
    pred_parser.add_argument("--age", type=float, default=None, help="Age in years (optional)")
    pred_parser.add_argument("--json", type=str, default=None, help="Property details as JSON string")

    args = parser.parse_args()

    if args.command == "generate-data":
        cmd_generate_data(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "test":
        cmd_test(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "demo":
        cmd_demo(args)
    else:
        # Default: show help or run demo
        parser.print_help()


if __name__ == "__main__":
    main()

