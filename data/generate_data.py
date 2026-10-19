"""
Data Generation Script: Generates a realistic synthetic housing dataset.
Features: area_sqft, bedrooms, bathrooms, age_years, price.
Includes edge-case handling demonstration (e.g. a small percentage of missing age_years).
"""

import os
import numpy as np
import pandas as pd


def generate_housing_dataset(n_samples: int = 1500, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a synthetic dataset of housing features and market prices.

    Parameters:
        n_samples (int): Number of house records to generate.
        random_state (int): Seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame containing housing features and target price.
    """
    np.random.seed(random_state)

    # 1. Generate area in square feet (typically 500 to 4500 sqft)
    area_sqft = np.random.normal(loc=1600, scale=600, size=n_samples)
    area_sqft = np.clip(area_sqft, 500, 4500).astype(int)
 
    # 2. Generate bedrooms (1 to 5 bedrooms, naturally scaling with area)
    # Larger homes typically have more bedrooms
    bedrooms = np.clip(np.round(area_sqft / 500 + np.random.normal(0, 0.5, n_samples)), 1, 6).astype(int)

    # 3. Generate bathrooms (1 to 5 bathrooms, typically <= bedrooms + 1)
    bathrooms = np.clip(np.round(bedrooms * 0.8 + np.random.normal(0, 0.4, n_samples)), 1, 5).astype(int)

    # 4. Generate age in years (0 = brand new, up to 45 years old)
    age_years = np.random.randint(0, 45, size=n_samples).astype(float)

    # 5. Calculate realistic base price with domain logic:
    # - Base land cost: ~800,000
    # - Price per sqft: ~2,800
    # - Value per bedroom: ~250,000
    # - Value per bathroom: ~180,000
    # - Depreciation per year of age: ~-22,000
    # - Random market noise (location variances, finishes): +/- 150,000
    noise = np.random.normal(0, 150000, size=n_samples)
    price = (
        800000
        + (area_sqft * 2800)
        + (bedrooms * 250000)
        + (bathrooms * 180000)
        - (age_years * 22000)
        + noise
    )

    # Prices should not be unrealistically low
    price = np.clip(price, 1200000, 15000000)
    # Round price to nearest 1,000 for realistic valuation output
    price = np.round(price, -3).astype(int)

    # 6. Inject a small realistic percentage (~3%) of missing values in age_years
    # to demonstrate requirement 2: "Handle missing or invalid values appropriately"
    missing_mask = np.random.rand(n_samples) < 0.03
    age_years[missing_mask] = np.nan

    df = pd.DataFrame({
        "area_sqft": area_sqft,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "age_years": age_years,
        "price": price
    })

    return df


def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_csv = os.path.join(output_dir, "house_prices.csv")
    
    print(f"Generating synthetic housing dataset...")
    df = generate_housing_dataset(n_samples=1500, random_state=42)
    df.to_csv(output_csv, index=False)
    
    print(f"Successfully generated dataset with {len(df)} rows.")
    print(f"Saved to: {output_csv}")
    print("\nDataset Summary:")
    print(df.info())
    print("\nSample Preview (First 5 records):")
    print(df.head())


if __name__ == "__main__":
    main()

