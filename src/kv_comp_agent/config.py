from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DATA_PATH = DATA_DIR / "alberta_residential_sales.csv"


SCORING_WEIGHTS = {
    "location": 0.30,
    "property_type": 0.20,
    "living_area": 0.15,
    "bed_bath": 0.10,
    "year_built": 0.10,
    "recency": 0.10,
    "features": 0.05,
}


ALBERTA_CITY_COORDS = {
    "Edmonton": (53.5461, -113.4938),
    "Calgary": (51.0447, -114.0719),
    "Red Deer": (52.2681, -113.8112),
    "St. Albert": (53.6305, -113.6256),
    "Sherwood Park": (53.5412, -113.2957),
    "Leduc": (53.2594, -113.5493),
    "Airdrie": (51.2917, -114.0144),
    "Fort McMurray": (56.7267, -111.3790),
}


PROPERTY_TYPE_BASE_PPSF = {
    "Detached": 360,
    "Semi-Detached": 330,
    "Townhouse": 310,
    "Duplex": 320,
    "Condo": 285,
}


CONDITION_MULTIPLIERS = {
    "Poor": 0.86,
    "Fair": 0.93,
    "Average": 1.00,
    "Good": 1.07,
    "Excellent": 1.15,
}