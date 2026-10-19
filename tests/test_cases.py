"""
Automated Test Suite for House Price Prediction
Verifies and validates all test cases from the official Problem Statement:
- TC1: Valid property (area 1200, 2 bed, 2 bath) -> Return numerical price
- TC2: Area = -500 -> Reject input
- TC3: Bedrooms = 0 -> Reject input
- TC4: Missing age_years -> Report missing field or apply documented imputation rule
- TC5: Larger properties consistency test (price generally increases with area)
"""

import os
import sys
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predict import predict_price, PricePredictor
from src.validator import validate_property_input


def test_tc1_valid_property():
    """
    TC1: Valid property with area 1200, 2 bedrooms, 2 bathrooms.
    Expected behaviour: Return a numerical price.
    Supports both when age_years is explicitly provided (as in Example Input)
    and when age_years is omitted (as phrased in TC1 table).
    """
    # 1. With age_years provided (from Example Input)
    input_with_age = {
        "area_sqft": 1200,
        "bedrooms": 2,
        "bathrooms": 2,
        "age_years": 5
    }
    result_with_age = predict_price(input_with_age)
    assert "predicted_price" in result_with_age, "Result must contain 'predicted_price'"
    assert isinstance(result_with_age["predicted_price"], (int, float)), "Predicted price must be numeric"
    assert result_with_age["predicted_price"] > 0, "Predicted price must be positive"

    # 2. Without age_years explicitly provided (as phrased in TC1 table)
    input_without_age = {
        "area_sqft": 1200,
        "bedrooms": 2,
        "bathrooms": 2
    }
    result_without_age = predict_price(input_without_age)
    assert "predicted_price" in result_without_age, "Result must contain 'predicted_price'"
    assert isinstance(result_without_age["predicted_price"], (int, float)), "Predicted price must be numeric"
    assert result_without_age["predicted_price"] > 0, "Predicted price must be positive"


def test_tc2_negative_area():
    """
    TC2: Area = -500.
    Expected behaviour: Reject input.
    """
    input_data = {
        "area_sqft": -500,
        "bedrooms": 2,
        "bathrooms": 2,
        "age_years": 5
    }
    with pytest.raises(ValueError) as exc_info:
        predict_price(input_data)
    
    # Verify rejection reason
    assert "TC2" in str(exc_info.value) or "area_sqft" in str(exc_info.value)


def test_tc3_zero_bedrooms():
    """
    TC3: Bedrooms = 0.
    Expected behaviour: Reject input.
    """
    input_data = {
        "area_sqft": 1200,
        "bedrooms": 0,
        "bathrooms": 2,
        "age_years": 5
    }
    with pytest.raises(ValueError) as exc_info:
        predict_price(input_data)

    # Verify rejection reason
    assert "TC3" in str(exc_info.value) or "bedrooms" in str(exc_info.value)


def test_tc4_missing_age_years():
    """
    TC4: Missing age_years.
    Expected behaviour: Report missing field or apply a documented imputation rule.
    """
    input_data = {
        "area_sqft": 1200,
        "bedrooms": 2,
        "bathrooms": 2
        # age_years is deliberately missing
    }
    
    # 1. When imputation is permitted, it applies median age and notes it
    result = predict_price(input_data)
    assert "predicted_price" in result
    assert result["predicted_price"] > 0
    assert "notice" in result, "Should report documented imputation notice"
    assert "TC4" in result["notice"]
    assert "age_years" in result["inputs_used"]

    # 2. When imputation is disabled, it strictly rejects missing age
    predictor = PricePredictor()
    with pytest.raises(ValueError) as exc_info:
        predictor.predict(input_data, allow_imputation=False)
    assert "TC4" in str(exc_info.value) or "missing" in str(exc_info.value).lower()


def test_tc5_monotonicity_area_consistency():
    """
    TC5: Larger properties from a dataset where price generally increases with area.
    Expected behaviour: Predictions should be reasonably consistent with the learned
    relationship (larger area yields higher predicted price when other features are fixed).
    """
    predictor = PricePredictor()

    small_prop = {"area_sqft": 1000, "bedrooms": 3, "bathrooms": 2, "age_years": 10}
    medium_prop = {"area_sqft": 2000, "bedrooms": 3, "bathrooms": 2, "age_years": 10}
    large_prop = {"area_sqft": 3200, "bedrooms": 3, "bathrooms": 2, "age_years": 10}

    price_small = predictor.predict(small_prop)["predicted_price"]
    price_medium = predictor.predict(medium_prop)["predicted_price"]
    price_large = predictor.predict(large_prop)["predicted_price"]

    assert price_small < price_medium < price_large, (
        f"Expected price to increase with area: Small({price_small}) < Medium({price_medium}) < Large({price_large})"
    )


def run_all_tests_with_report():
    """
    Executes all test cases and prints a formatted verification report.
    """
    print("=" * 70)
    print("        OFFICIAL TEST CASES EXECUTION & VERIFICATION REPORT        ")
    print("=" * 70)

    tests = [
        ("TC1", "Valid property (area=1200, bed=2, bath=2)", test_tc1_valid_property),
        ("TC2", "Negative area (area=-500)", test_tc2_negative_area),
        ("TC3", "Zero bedrooms (bedrooms=0)", test_tc3_zero_bedrooms),
        ("TC4", "Missing age_years (imputation rule)", test_tc4_missing_age_years),
        ("TC5", "Larger properties relationship consistency", test_tc5_monotonicity_area_consistency),
    ]

    passed_count = 0
    for code, desc, test_fn in tests:
        print(f"\nRunning {code}: {desc}...")
        try:
            test_fn()
            print(f"  [PASS] {code} passed successfully!")
            passed_count += 1
        except Exception as e:
            print(f"  [FAIL] {code} failed with error: {e}")

    print("\n" + "=" * 70)
    print(f"SUMMARY: {passed_count}/{len(tests)} Test Cases Passed.")
    print("=" * 70)
    return passed_count == len(tests)


if __name__ == "__main__":
    success = run_all_tests_with_report()
    sys.exit(0 if success else 1)

