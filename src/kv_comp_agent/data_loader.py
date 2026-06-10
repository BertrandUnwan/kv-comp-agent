from __future__ import annotations

from pathlib import Path

import pandas as pd

from kv_comp_agent.config import DEFAULT_DATA_PATH


REQUIRED_COLUMNS = {
    "property_id",
    "address",
    "city",
    "neighborhood",
    "latitude",
    "longitude",
    "property_type",
    "year_built",
    "sale_date",
    "condition",
    "finished_basement",
    "renovated",
    "near_transit",
    "backs_onto_park",
    "bedrooms",
    "bathrooms",
    "living_area_sqft",
    "lot_size_sqft",
    "garage_spaces",
    "sale_price",
}


class DataLoadError(Exception):
    """Raised when the property sales dataset cannot be loaded or validated."""


def load_sales_data(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """
    Load and lightly validate the historical sales dataset.

    The app is intentionally designed to fail clearly if the dataset is missing
    required columns, while still tolerating missing values in fields like
    lot_size_sqft or year_built.
    """
    path = Path(path)

    if not path.exists():
        raise DataLoadError(f"Sales dataset not found at: {path}")

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise DataLoadError(f"Could not read sales dataset: {exc}") from exc

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise DataLoadError(f"Sales dataset is missing required columns: {missing}")

    df = clean_sales_data(df)

    if df.empty:
        raise DataLoadError("Sales dataset loaded successfully but contains no usable rows.")

    return df


def clean_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize data types and remove rows that cannot support comp analysis.

    Missing lot size and year built are allowed because the scoring engine can
    reweight around missing optional fields.
    """
    df = df.copy()

    text_columns = ["property_id", "address", "city", "neighborhood", "property_type", "condition"]
    for column in text_columns:
        df[column] = df[column].astype(str).str.strip()

    numeric_columns = [
        "latitude",
        "longitude",
        "year_built",
        "bedrooms",
        "bathrooms",
        "living_area_sqft",
        "lot_size_sqft",
        "garage_spaces",
        "sale_price",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    boolean_columns = ["finished_basement", "renovated", "near_transit", "backs_onto_park"]
    for column in boolean_columns:
        df[column] = df[column].map(_to_bool)

    df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce")

    # Required for any meaningful comp analysis.
    df = df.dropna(
        subset=[
            "property_id",
            "city",
            "neighborhood",
            "property_type",
            "latitude",
            "longitude",
            "sale_date",
            "bedrooms",
            "bathrooms",
            "living_area_sqft",
            "sale_price",
        ]
    )

    # Remove obviously invalid records.
    df = df[df["living_area_sqft"] > 250]
    df = df[df["sale_price"] > 25000]
    df = df[df["bedrooms"] >= 0]
    df = df[df["bathrooms"] >= 0]

    return df.reset_index(drop=True)


def _to_bool(value: object) -> bool:
    """
    Convert common CSV boolean representations into Python bools.
    """
    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return False

    value_str = str(value).strip().lower()

    if value_str in {"true", "1", "yes", "y"}:
        return True

    if value_str in {"false", "0", "no", "n"}:
        return False

    return False