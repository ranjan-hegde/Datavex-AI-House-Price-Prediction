"""
Input Validator Module
Enforces business rules and validation logic for property features.
Directly handles Test Cases:
- TC2: Reject area_sqft <= 0 (e.g., area = -500)
- TC3: Reject bedrooms <= 0 (e.g., bedrooms = 0)
- TC4: Missing age_years handling (applies documented imputation or flags missing)
"""

from typing import Dict, Any, Tuple, Optional


def validate_property_input(
    data: Dict[str, Any],
    default_age_imputation: Optional[float] = None,
    allow_imputation: bool = True
) -> Tuple[Dict[str, float], Optional[str]]:
    """
    Validates property details dictionary against domain rules.

    Parameters:
        data (dict): Dictionary with property features (e.g. area_sqft, bedrooms, etc.).
        default_age_imputation (float, optional): Imputation value for missing age_years.
        allow_imputation (bool): If True, applies default imputation for missing age_years.

    Returns:
        tuple: (clean_data: dict, info_message: str or None)

    Raises:
        ValueError: If any validation rule fails (e.g., negative area, zero bedrooms).
        TypeError: If input types are non-numeric.
    """
    if not isinstance(data, dict):
        raise TypeError(f"Property input must be a dictionary, got {type(data).__name__}.")

    # Required mandatory fields (excluding age_years which can be imputed under TC4)
    mandatory_fields = ["area_sqft", "bedrooms", "bathrooms"]
    for field in mandatory_fields:
        if field not in data or data[field] is None:
            raise ValueError(f"Missing required field: '{field}'")

    # 1. Validate area_sqft (TC2 check)
    try:
        area_sqft = float(data["area_sqft"])
    except (ValueError, TypeError):
        raise TypeError(f"Invalid area_sqft '{data['area_sqft']}': must be a numeric value.")

    if area_sqft <= 0:
        raise ValueError(
            f"Validation Error (TC2): 'area_sqft' must be strictly positive (> 0). Received: {area_sqft}"
        )

    # 2. Validate bedrooms (TC3 check)
    try:
        bedrooms = int(data["bedrooms"])
    except (ValueError, TypeError):
        raise TypeError(f"Invalid bedrooms '{data['bedrooms']}': must be an integer.")

    if bedrooms <= 0:
        raise ValueError(
            f"Validation Error (TC3): 'bedrooms' must be at least 1. Received: {bedrooms}"
        )

    # 3. Validate bathrooms
    try:
        bathrooms = int(data["bathrooms"])
    except (ValueError, TypeError):
        raise TypeError(f"Invalid bathrooms '{data['bathrooms']}': must be an integer.")

    if bathrooms <= 0:
        raise ValueError(
            f"Validation Error: 'bathrooms' must be at least 1. Received: {bathrooms}"
        )

    # 4. Validate or Impute age_years (TC4 check)
    imputation_notice = None
    age_raw = data.get("age_years", None)

    if age_raw is None:
        if allow_imputation and default_age_imputation is not None:
            age_years = float(default_age_imputation)
            imputation_notice = (
                f"Notice (TC4): 'age_years' was not provided. "
                f"Applied documented imputation rule using median training age ({age_years:.1f} years)."
            )
        else:
            raise ValueError(
                "Validation Error (TC4): 'age_years' is missing and no imputation rule is configured."
            )
    else:
        try:
            age_years = float(age_raw)
        except (ValueError, TypeError):
            raise TypeError(f"Invalid age_years '{age_raw}': must be a numeric value.")

        if age_years < 0:
            raise ValueError(
                f"Validation Error: 'age_years' cannot be negative. Received: {age_years}"
            )

    validated_record = {
        "area_sqft": area_sqft,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "age_years": age_years
    }

    return validated_record, imputation_notice

