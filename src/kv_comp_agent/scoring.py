from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from kv_comp_agent.config import SCORING_WEIGHTS
from kv_comp_agent.schema import SubjectProperty


def score_comps(subject: SubjectProperty, candidates: pd.DataFrame) -> pd.DataFrame:
    """
    Score candidate comparable properties from 0 to 100.

    The scoring model is intentionally deterministic and explainable.
    It does not rely on the LLM to decide value or similarity.
    """
    if candidates.empty:
        return candidates.copy()

    scored = candidates.copy()

    scored["price_per_sqft"] = scored["sale_price"] / scored["living_area_sqft"]

    scored["location_score"] = scored.apply(lambda row: score_location(subject, row), axis=1)
    scored["property_type_score"] = scored.apply(lambda row: score_property_type(subject, row), axis=1)
    scored["living_area_score"] = scored.apply(lambda row: score_living_area(subject, row), axis=1)
    scored["bed_bath_score"] = scored.apply(lambda row: score_bed_bath(subject, row), axis=1)
    scored["year_built_score"] = scored.apply(lambda row: score_year_built(subject, row), axis=1)
    scored["recency_score"] = scored.apply(lambda row: score_recency(row), axis=1)
    scored["features_score"] = scored.apply(lambda row: score_features(subject, row), axis=1)

    scored["total_score"] = (
        scored["location_score"] * SCORING_WEIGHTS["location"]
        + scored["property_type_score"] * SCORING_WEIGHTS["property_type"]
        + scored["living_area_score"] * SCORING_WEIGHTS["living_area"]
        + scored["bed_bath_score"] * SCORING_WEIGHTS["bed_bath"]
        + scored["year_built_score"] * SCORING_WEIGHTS["year_built"]
        + scored["recency_score"] * SCORING_WEIGHTS["recency"]
        + scored["features_score"] * SCORING_WEIGHTS["features"]
    )

    scored["total_score"] = scored["total_score"].round(2)
    scored["reason_selected"] = scored.apply(lambda row: explain_comp_match(subject, row), axis=1)

    return scored.sort_values("total_score", ascending=False).reset_index(drop=True)


def score_location(subject: SubjectProperty, comp: pd.Series) -> float:
    """Score location similarity using city and neighborhood match."""
    comp_city = str(comp.get("city", "")).lower().strip()
    comp_neighborhood = str(comp.get("neighborhood", "")).lower().strip()

    subject_city = subject.city.lower().strip()
    subject_neighborhood = (subject.neighborhood or "").lower().strip()

    if subject_neighborhood and comp_city == subject_city and comp_neighborhood == subject_neighborhood:
        return 100.0

    if comp_city == subject_city:
        return 78.0

    return 45.0


def score_property_type(subject: SubjectProperty, comp: pd.Series) -> float:
    """Score property-type similarity."""
    comp_type = str(comp.get("property_type", "")).strip()

    if comp_type == subject.property_type.value:
        return 100.0

    compatible_types = {
        "Detached": {"Semi-Detached", "Duplex"},
        "Semi-Detached": {"Detached", "Duplex", "Townhouse"},
        "Duplex": {"Semi-Detached", "Townhouse", "Detached"},
        "Townhouse": {"Duplex", "Semi-Detached", "Condo"},
        "Condo": {"Townhouse"},
    }

    if comp_type in compatible_types.get(subject.property_type.value, set()):
        return 65.0

    return 35.0


def score_living_area(subject: SubjectProperty, comp: pd.Series) -> float:
    """Score living-area similarity."""
    comp_area = float(comp.get("living_area_sqft", np.nan))

    if np.isnan(comp_area) or comp_area <= 0:
        return 50.0

    pct_diff = abs(comp_area - subject.living_area_sqft) / subject.living_area_sqft

    if pct_diff <= 0.05:
        return 100.0
    if pct_diff <= 0.10:
        return 92.0
    if pct_diff <= 0.20:
        return 80.0
    if pct_diff <= 0.35:
        return 62.0
    if pct_diff <= 0.50:
        return 45.0

    return 25.0


def score_bed_bath(subject: SubjectProperty, comp: pd.Series) -> float:
    """Score bedroom and bathroom similarity."""
    comp_beds = float(comp.get("bedrooms", np.nan))
    comp_baths = float(comp.get("bathrooms", np.nan))

    if np.isnan(comp_beds) or np.isnan(comp_baths):
        return 50.0

    bed_diff = abs(comp_beds - subject.bedrooms)
    bath_diff = abs(comp_baths - subject.bathrooms)

    bed_score = max(0.0, 100.0 - bed_diff * 22.0)
    bath_score = max(0.0, 100.0 - bath_diff * 18.0)

    return round((bed_score + bath_score) / 2, 2)


def score_year_built(subject: SubjectProperty, comp: pd.Series) -> float:
    """Score age/year-built similarity while tolerating missing data."""
    if subject.year_built is None:
        return 70.0

    comp_year = comp.get("year_built", np.nan)

    if pd.isna(comp_year):
        return 60.0

    year_diff = abs(float(comp_year) - subject.year_built)

    if year_diff <= 3:
        return 100.0
    if year_diff <= 7:
        return 90.0
    if year_diff <= 15:
        return 75.0
    if year_diff <= 30:
        return 55.0

    return 35.0


def score_recency(comp: pd.Series) -> float:
    """Score sale recency."""
    sale_date = comp.get("sale_date")

    if pd.isna(sale_date):
        return 45.0

    if isinstance(sale_date, str):
        sale_date = pd.to_datetime(sale_date, errors="coerce")

    if pd.isna(sale_date):
        return 45.0

    today = pd.Timestamp(date.today())
    days_old = max(0, (today - pd.Timestamp(sale_date)).days)

    if days_old <= 90:
        return 100.0
    if days_old <= 180:
        return 90.0
    if days_old <= 365:
        return 78.0
    if days_old <= 730:
        return 60.0
    if days_old <= 1095:
        return 42.0

    return 25.0


def score_features(subject: SubjectProperty, comp: pd.Series) -> float:
    """Score feature similarity across boolean and small categorical attributes."""
    comparisons = []

    feature_names = [
        "finished_basement",
        "renovated",
        "near_transit",
        "backs_onto_park",
    ]

    for feature in feature_names:
        subject_value = getattr(subject, feature)
        comp_value = bool(comp.get(feature, False))
        comparisons.append(1.0 if subject_value == comp_value else 0.0)

    comp_garage = comp.get("garage_spaces", np.nan)
    if not pd.isna(comp_garage):
        garage_diff = abs(float(comp_garage) - subject.garage_spaces)
        comparisons.append(max(0.0, 1.0 - garage_diff * 0.35))

    comp_condition = str(comp.get("condition", "")).strip()
    comparisons.append(1.0 if comp_condition == subject.condition.value else 0.65)

    if not comparisons:
        return 60.0

    return round(float(np.mean(comparisons) * 100), 2)


def explain_comp_match(subject: SubjectProperty, comp: pd.Series) -> str:
    """Create a concise reason why a comp was selected."""
    reasons = []

    if str(comp.get("neighborhood", "")).lower().strip() == (subject.neighborhood or "").lower().strip():
        reasons.append("same neighborhood")
    elif str(comp.get("city", "")).lower().strip() == subject.city.lower().strip():
        reasons.append("same city")

    if str(comp.get("property_type", "")).strip() == subject.property_type.value:
        reasons.append("same property type")

    area = float(comp.get("living_area_sqft", np.nan))
    if not np.isnan(area):
        pct_diff = abs(area - subject.living_area_sqft) / subject.living_area_sqft
        reasons.append(f"{pct_diff:.0%} size difference")

    if comp.get("sale_date") is not None:
        reasons.append("recent sale considered")

    if not reasons:
        return "Selected by fallback comp search."

    return "Selected because it has " + ", ".join(reasons) + "."