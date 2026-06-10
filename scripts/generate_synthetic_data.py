from __future__ import annotations

import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from kv_comp_agent.config import (
    ALBERTA_CITY_COORDS,
    CONDITION_MULTIPLIERS,
    DATA_DIR,
    PROPERTY_TYPE_BASE_PPSF,
)


RANDOM_SEED = 42
N_PROPERTIES = 1200


NEIGHBORHOODS = {
    "Edmonton": [
        "Glenora",
        "Windermere",
        "Oliver",
        "Summerside",
        "Terwillegar",
        "Laurel",
        "Rutherford",
        "Westmount",
    ],
    "Calgary": [
        "Beltline",
        "Kensington",
        "Mahogany",
        "Bridgeland",
        "Aspen Woods",
        "Seton",
        "Inglewood",
        "Auburn Bay",
    ],
    "Red Deer": ["Clearview", "Timberlands", "West Park", "Anders", "Lancaster"],
    "St. Albert": ["Erin Ridge", "Lacombe Park", "Oakmont", "Mission", "Riverside"],
    "Sherwood Park": ["Summerwood", "Emerald Hills", "Clarkdale", "Nottingham", "Davidson Creek"],
    "Leduc": ["Meadowview", "Southfork", "Black Stone", "Tribute", "West Haven"],
    "Airdrie": ["Coopers Crossing", "Kings Heights", "Bayside", "Reunion", "Hillcrest"],
    "Fort McMurray": ["Timberlea", "Thickwood", "Eagle Ridge", "Abasand", "Beacon Hill"],
}


NEIGHBORHOOD_PREMIUMS = {
    "Glenora": 1.24,
    "Windermere": 1.18,
    "Oliver": 1.08,
    "Summerside": 1.06,
    "Terwillegar": 1.10,
    "Laurel": 1.00,
    "Rutherford": 0.98,
    "Westmount": 1.13,
    "Beltline": 1.10,
    "Kensington": 1.18,
    "Mahogany": 1.09,
    "Bridgeland": 1.17,
    "Aspen Woods": 1.30,
    "Seton": 0.99,
    "Inglewood": 1.14,
    "Auburn Bay": 1.05,
}


CITY_MULTIPLIERS = {
    "Calgary": 1.08,
    "Edmonton": 1.00,
    "St. Albert": 1.06,
    "Sherwood Park": 1.03,
    "Airdrie": 0.96,
    "Leduc": 0.90,
    "Red Deer": 0.86,
    "Fort McMurray": 0.92,
}


PROPERTY_TYPES = ["Detached", "Semi-Detached", "Townhouse", "Duplex", "Condo"]
CONDITIONS = ["Poor", "Fair", "Average", "Good", "Excellent"]


def random_sale_date() -> date:
    """Generate a sale date mostly within the last 24 months."""
    today = date.today()

    if random.random() < 0.85:
        days_back = random.randint(1, 730)
    else:
        days_back = random.randint(731, 1460)

    return today - timedelta(days=days_back)


def jitter_coordinates(city: str) -> tuple[float, float]:
    """Create nearby coordinates around a city center."""
    lat, lon = ALBERTA_CITY_COORDS[city]
    return (
        round(lat + np.random.normal(0, 0.035), 6),
        round(lon + np.random.normal(0, 0.045), 6),
    )


def generate_property_characteristics(property_type: str) -> dict:
    """Generate realistic residential property characteristics."""
    if property_type == "Detached":
        living_area = int(np.random.normal(1950, 450))
        lot_size = int(np.random.normal(5200, 1200))
        bedrooms = random.choices([3, 4, 5, 6], weights=[25, 45, 25, 5])[0]
        bathrooms = random.choices([2, 2.5, 3, 3.5, 4], weights=[15, 20, 35, 20, 10])[0]
        garage_spaces = random.choices([1, 2, 3], weights=[15, 70, 15])[0]

    elif property_type == "Semi-Detached":
        living_area = int(np.random.normal(1550, 300))
        lot_size = int(np.random.normal(3200, 800))
        bedrooms = random.choices([2, 3, 4], weights=[15, 65, 20])[0]
        bathrooms = random.choices([1.5, 2, 2.5, 3], weights=[15, 30, 35, 20])[0]
        garage_spaces = random.choices([0, 1, 2], weights=[20, 45, 35])[0]

    elif property_type == "Townhouse":
        living_area = int(np.random.normal(1350, 250))
        lot_size = int(np.random.normal(1800, 500))
        bedrooms = random.choices([2, 3, 4], weights=[25, 60, 15])[0]
        bathrooms = random.choices([1.5, 2, 2.5, 3], weights=[20, 30, 35, 15])[0]
        garage_spaces = random.choices([0, 1, 2], weights=[25, 55, 20])[0]

    elif property_type == "Duplex":
        living_area = int(np.random.normal(1450, 280))
        lot_size = int(np.random.normal(2800, 700))
        bedrooms = random.choices([2, 3, 4], weights=[20, 60, 20])[0]
        bathrooms = random.choices([1.5, 2, 2.5, 3], weights=[15, 35, 35, 15])[0]
        garage_spaces = random.choices([0, 1, 2], weights=[25, 45, 30])[0]

    else:
        living_area = int(np.random.normal(850, 220))
        lot_size = None
        bedrooms = random.choices([1, 2, 3], weights=[35, 50, 15])[0]
        bathrooms = random.choices([1, 1.5, 2], weights=[55, 15, 30])[0]
        garage_spaces = random.choices([0, 1, 2], weights=[45, 45, 10])[0]

    living_area = max(450, living_area)
    if lot_size is not None:
        lot_size = max(900, lot_size)

    return {
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "living_area_sqft": living_area,
        "lot_size_sqft": lot_size,
        "garage_spaces": garage_spaces,
    }


def estimate_sale_price(row: dict) -> int:
    """Generate a realistic synthetic sale price from property characteristics."""
    base_ppsf = PROPERTY_TYPE_BASE_PPSF[row["property_type"]]
    city_factor = CITY_MULTIPLIERS[row["city"]]
    neighborhood_factor = NEIGHBORHOOD_PREMIUMS.get(row["neighborhood"], 1.0)
    condition_factor = CONDITION_MULTIPLIERS[row["condition"]]

    ppsf = base_ppsf * city_factor * neighborhood_factor * condition_factor

    age = max(0, date.today().year - row["year_built"])
    age_adjustment = max(0.82, 1 - age * 0.003)

    feature_adjustment = 1.0
    if row["renovated"]:
        feature_adjustment += 0.06
    if row["finished_basement"]:
        feature_adjustment += 0.04
    if row["near_transit"]:
        feature_adjustment += 0.025
    if row["backs_onto_park"]:
        feature_adjustment += 0.035
    if row["garage_spaces"] >= 2:
        feature_adjustment += 0.025

    lot_adjustment = 0
    if row["lot_size_sqft"]:
        lot_adjustment = max(0, row["lot_size_sqft"] - 2500) * 12

    noise = np.random.normal(1.0, 0.055)

    price = (
        row["living_area_sqft"]
        * ppsf
        * age_adjustment
        * feature_adjustment
        * noise
        + lot_adjustment
    )

    return int(round(price / 1000) * 1000)


def build_dataset(n: int = N_PROPERTIES) -> pd.DataFrame:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    city_weights = {
        "Edmonton": 0.32,
        "Calgary": 0.32,
        "Red Deer": 0.08,
        "St. Albert": 0.07,
        "Sherwood Park": 0.07,
        "Leduc": 0.05,
        "Airdrie": 0.05,
        "Fort McMurray": 0.04,
    }

    rows = []
    cities = list(city_weights.keys())
    weights = list(city_weights.values())

    for i in range(1, n + 1):
        city = random.choices(cities, weights=weights)[0]
        neighborhood = random.choice(NEIGHBORHOODS[city])
        property_type = random.choices(
            PROPERTY_TYPES,
            weights=[0.43, 0.12, 0.18, 0.10, 0.17],
        )[0]

        characteristics = generate_property_characteristics(property_type)
        latitude, longitude = jitter_coordinates(city)

        year_built = random.randint(1960, 2025)
        condition = random.choices(
            CONDITIONS,
            weights=[0.04, 0.13, 0.43, 0.30, 0.10],
        )[0]

        row = {
            "property_id": f"AB-{i:05d}",
            "address": f"{random.randint(100, 9999)} {random.choice(['Maple', 'River', 'Park', 'Cedar', 'Aspen', 'Prairie'])} {random.choice(['St', 'Ave', 'Rd', 'Blvd', 'Way'])}",
            "city": city,
            "neighborhood": neighborhood,
            "latitude": latitude,
            "longitude": longitude,
            "property_type": property_type,
            "year_built": year_built,
            "sale_date": random_sale_date().isoformat(),
            "condition": condition,
            "finished_basement": random.random() < 0.42,
            "renovated": random.random() < 0.28,
            "near_transit": random.random() < 0.35,
            "backs_onto_park": random.random() < 0.12,
            **characteristics,
        }

        row["sale_price"] = estimate_sale_price(row)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Intentional edge cases to test graceful handling.
    df.loc[df.sample(frac=0.025, random_state=RANDOM_SEED).index, "lot_size_sqft"] = np.nan
    df.loc[df.sample(frac=0.015, random_state=RANDOM_SEED + 1).index, "year_built"] = np.nan

    return df


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_path = DATA_DIR / "alberta_residential_sales.csv"
    df = build_dataset()
    df.to_csv(output_path, index=False)

    print(f"Generated {len(df):,} synthetic Alberta residential sales.")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()