from __future__ import annotations

import numpy as np
import pandas as pd

from kv_comp_agent.schema import SubjectProperty, ValuationEstimate


def estimate_value(
    subject: SubjectProperty,
    scored_comps: pd.DataFrame,
    top_n: int = 5,
) -> ValuationEstimate:
    """
    Estimate subject property value using weighted comparable sales.

    The base estimate is calculated from weighted price-per-square-foot, where
    higher-scoring comps receive more influence. The final output is a range
    rather than a false-precision single value.
    """
    if scored_comps.empty:
        return ValuationEstimate(
            low_estimate=0,
            base_estimate=0,
            high_estimate=0,
            confidence="Low",
            confidence_score=0.0,
            methodology="No comparable sales were available, so no reliable valuation could be produced.",
            risk_flags=["No comparable sales found."],
        )

    comps = scored_comps.head(top_n).copy()

    if "price_per_sqft" not in comps.columns:
        comps["price_per_sqft"] = comps["sale_price"] / comps["living_area_sqft"]

    weights = _normalized_weights(comps["total_score"])

    weighted_ppsf = float(np.average(comps["price_per_sqft"], weights=weights))
    base_estimate = int(round(weighted_ppsf * subject.living_area_sqft / 1000) * 1000)

    confidence_score = calculate_confidence_score(comps)
    confidence = confidence_label(confidence_score)

    margin = uncertainty_margin(confidence_score)
    low_estimate = int(round(base_estimate * (1 - margin) / 1000) * 1000)
    high_estimate = int(round(base_estimate * (1 + margin) / 1000) * 1000)

    risk_flags = generate_risk_flags(subject, comps, confidence_score)

    methodology = (
        "Estimated using a weighted price-per-square-foot approach from the top "
        f"{len(comps)} comparable sales. Higher-scoring comps receive more weight. "
        "The range widens when comp quality, recency, or similarity is weaker."
    )

    return ValuationEstimate(
        low_estimate=low_estimate,
        base_estimate=base_estimate,
        high_estimate=high_estimate,
        confidence=confidence,
        confidence_score=round(confidence_score, 2),
        methodology=methodology,
        risk_flags=risk_flags,
    )


def _normalized_weights(scores: pd.Series) -> np.ndarray:
    """
    Convert comp scores into normalized positive weights.

    Squaring the scores gives stronger comps more influence without fully
    ignoring weaker comps.
    """
    raw = np.clip(scores.astype(float).to_numpy(), 1.0, 100.0)
    weighted = raw**2

    if weighted.sum() == 0:
        return np.ones(len(weighted)) / len(weighted)

    return weighted / weighted.sum()


def calculate_confidence_score(comps: pd.DataFrame) -> float:
    """
    Produce a 0-1 confidence score from comp quality and count.
    """
    if comps.empty:
        return 0.0

    avg_score = float(comps["total_score"].mean()) / 100
    count_factor = min(len(comps) / 5, 1.0)

    recency_factor = 0.7
    if "recency_score" in comps.columns:
        recency_factor = float(comps["recency_score"].mean()) / 100

    property_type_factor = 0.7
    if "property_type_score" in comps.columns:
        property_type_factor = float(comps["property_type_score"].mean()) / 100

    confidence = (
        avg_score * 0.55
        + count_factor * 0.15
        + recency_factor * 0.15
        + property_type_factor * 0.15
    )

    return float(np.clip(confidence, 0.0, 1.0))


def confidence_label(confidence_score: float) -> str:
    if confidence_score >= 0.82:
        return "High"
    if confidence_score >= 0.62:
        return "Medium"
    return "Low"


def uncertainty_margin(confidence_score: float) -> float:
    """
    Convert confidence into a valuation range width.
    """
    if confidence_score >= 0.82:
        return 0.07

    if confidence_score >= 0.62:
        return 0.11

    return 0.16


def generate_risk_flags(
    subject: SubjectProperty,
    comps: pd.DataFrame,
    confidence_score: float,
) -> list[str]:
    """
    Generate human-readable risk flags for underwriting review.
    """
    flags: list[str] = []

    if len(comps) < 5:
        flags.append("Fewer than five comparable sales were available.")

    if confidence_score < 0.62:
        flags.append("Overall comp quality is low; human review is strongly recommended.")

    if "location_score" in comps.columns and comps["location_score"].mean() < 80:
        flags.append("Some selected comps are outside the subject's immediate location.")

    if "property_type_score" in comps.columns and comps["property_type_score"].mean() < 90:
        flags.append("Some selected comps differ from the subject property type.")

    if "living_area_score" in comps.columns and comps["living_area_score"].mean() < 75:
        flags.append("Selected comps have meaningful living-area differences from the subject.")

    if "recency_score" in comps.columns and comps["recency_score"].mean() < 70:
        flags.append("Some selected comps are older sales, which may reduce reliability.")

    if subject.year_built is None:
        flags.append("Subject year built is missing, so age similarity could not be fully assessed.")

    if subject.lot_size_sqft is None:
        flags.append("Subject lot size is missing, so land-size differences were not fully assessed.")

    if not flags:
        flags.append("No major comp-quality risks detected in the selected sales.")

    return flags