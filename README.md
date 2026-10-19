# House Price Prediction (Regression - Supervised Learning)

An end-to-end Machine Learning pipeline that estimates the market price of residential properties based on features such as square footage (`area_sqft`), bedroom count (`bedrooms`), bathroom count (`bathrooms`), and property age (`age_years`).

---

## 1. Project Overview

This project satisfies all requirements outlined in the Problem Statement:
- **Data Ingestion & Cleaning**: Loads a structured CSV dataset and handles missing values using a documented median imputation rule.
- **Model Training & Comparison**: Trains and evaluates multiple supervised regression algorithms (**Linear Regression** and **Random Forest Regressor**) using an 80/20 train-test split.
- **Model Evaluation**: Measures performance using standard regression metrics: **MAE** (Mean Absolute Error) and **RMSE** (Root Mean Squared Error).
- **Validation & Inference**: Implements business domain validation rules (rejecting negative area, zero bedrooms, etc.) and provides a clean prediction interface.
- **Automated Verification**: Complete automated test suite covering all five official test cases (**TC1** to **TC5**).

---

## 2. Project Structure

```
detrax/
├── data/
│   ├── generate_data.py          # Synthetic dataset generator with edge cases
│   └── house_prices.csv          # Generated CSV dataset (1,500 records)
├── src/
│   ├── __init__.py
│   ├── validator.py              # Input validation logic (enforces TC2, TC3, TC4)
│   ├── data_loader.py            # Data loading, cleaning, and median imputation
│   ├── train.py                  # Model training, MAE/RMSE evaluation, artifact export
│   └── predict.py                # Prediction service and CLI interface
├── models/
│   └── house_price_model.joblib  # Serialized model artifact with metadata
├── tests/
│   ├── __init__.py
│   └── test_cases.py             # Automated verification of test cases TC1-TC5
├── explanation/                  # Comprehensive learning & interview preparation guides
├── main.py                       # CLI entry point (train, test, predict, demo)
└── README.md                     # Official project submission documentation
```

---

## 3. Tech Stack & Prerequisites

- **Python**: 3.10+ (Tested on Python 3.14)
- **Data Processing**: `pandas`, `numpy`
- **Machine Learning**: `scikit-learn`
- **Model Serialization**: `joblib`
- **Testing**: `pytest`

### Installation

Install required dependencies:
```bash
pip install numpy pandas scikit-learn joblib pytest
```

---

## 4. Execution Instructions

### A. Run Full End-to-End Pipeline (One Command)
Runs data generation, model training, evaluation, sample prediction, and all test cases:
```bash
python main.py demo
```

### B. Generate Dataset
```bash
python main.py generate-data
```

### C. Train and Evaluate Models
```bash
python main.py train
```

### D. Run Official Test Cases (TC1 – TC5)
```bash
python main.py test
```
Or using pytest:
```bash
pytest tests/test_cases.py -v
```

### E. Predict Property Price
**Example from Problem Statement:**
```bash
python main.py predict --area 1200 --bedrooms 2 --bathrooms 2 --age 5
```

**JSON input format:**
```bash
python main.py predict --json "{\"area_sqft\": 1200, \"bedrooms\": 2, \"bathrooms\": 2, \"age_years\": 5}"
```

**Interactive mode:**
```bash
python main.py predict
```

---

## 5. Model Evaluation Metrics

Evaluated on an independent 20% test set (300 samples) unseen during training:

| Model | MAE (Mean Absolute Error) | RMSE (Root Mean Squared Error) | $R^2$ Score |
| :--- | :--- | :--- | :--- |
| **Linear Regression** *(Selected Best)* | **INR 130,723.06** | **INR 167,320.22** | **0.9940** |
| **Random Forest Regressor** | INR 174,676.64 | INR 225,938.28 | 0.9891 |

*Note: The selected model is serialized and saved to `models/house_price_model.joblib`.*

---

## 6. Official Test Cases & Verification Results

All 5 test cases from the specification document are implemented and verified:

| Test Case | Description | Input | Expected Behaviour | Result |
| :--- | :--- | :--- | :--- | :---: |
| **TC1** | Valid Property | `{"area_sqft": 1200, "bedrooms": 2, "bathrooms": 2, "age_years": 5}` | Returns numerical price | **PASS** |
| **TC2** | Invalid Area | `{"area_sqft": -500, "bedrooms": 2, "bathrooms": 2, "age_years": 5}` | Reject input (`ValueError`) | **PASS** |
| **TC3** | Zero Bedrooms | `{"area_sqft": 1200, "bedrooms": 0, "bathrooms": 2, "age_years": 5}` | Reject input (`ValueError`) | **PASS** |
| **TC4** | Missing `age_years` | `{"area_sqft": 1200, "bedrooms": 2, "bathrooms": 2}` | Reports missing field / applies median imputation | **PASS** |
| **TC5** | Area Consistency | Properties with 1000 vs 2000 vs 3200 sqft | Predicted price increases with area | **PASS** |

### Sample Output from TC1 (Successful Prediction)
```json
{
  "predicted_price": 4928000,
  "inputs_used": {
    "area_sqft": 1200.0,
    "bedrooms": 2,
    "bathrooms": 2,
    "age_years": 5.0
  }
}
```

### Sample Output from TC2 / TC3 (Validation Rejection)
When invalid input is supplied (e.g., `area_sqft = -500`):
```text
[Validation Rejected]: Validation Error (TC2): 'area_sqft' must be strictly positive (> 0). Received: -500.0
```
When invalid bedrooms is supplied (e.g., `bedrooms = 0`):
```text
[Validation Rejected]: Validation Error (TC3): 'bedrooms' must be at least 1. Received: 0
```

### Sample Output from TC4 (Documented Imputation)
```json
{
  "predicted_price": 4554000,
  "inputs_used": {
    "area_sqft": 1200.0,
    "bedrooms": 2,
    "bathrooms": 2,
    "age_years": 22.0
  },
  "notice": "Notice (TC4): 'age_years' was not provided. Applied documented imputation rule using median training age (22.0 years)."
}
```

### Sample Output from Training Execution (`python main.py train`)
```text
============================================================
           HOUSE PRICE PREDICTION - MODEL TRAINING           
============================================================

[Step 1/5] Loading & preprocessing dataset...
Loaded 1500 records with features: ['area_sqft', 'bedrooms', 'bathrooms', 'age_years']
Imputation rule for missing age: median = 22.0 years

[Step 2/5] Splitting data into Train (80%) and Test (20%)...
Training set: 1200 samples | Testing set: 300 samples

[Step 3/5] Training Model 1: Linear Regression...
  > Linear Regression MAE  : INR 130,723.06
  > Linear Regression RMSE : INR 167,320.22
  > Linear Regression R^2  : 0.9940

[Step 4/5] Training Model 2: Random Forest Regressor...
  > Random Forest MAE  : INR 174,676.64
  > Random Forest RMSE : INR 225,938.28
  > Random Forest R^2  : 0.9891

------------------------------------------------------------
 SELECTED BEST MODEL: LinearRegression (Lowest RMSE: INR 167,320.22)
------------------------------------------------------------
[Step 5/5] Saved model artifact to: models/house_price_model.joblib
```

