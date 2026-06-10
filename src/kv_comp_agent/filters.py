from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from kv_comp_agent.schema import SubjectProperty


SearchStage = Literal[
    "same_neighborhood",
    "same_city",
    "same_city_wider",
    "nearby_markets",
    "all_markets_fallback",
]


@dataclass(frozen=True)
class CandidateSearchResult:
    """Candidate comparable properties plus metadata about how they were found."""

    candidates: pd.DataFrame
    search_stage: SearchStage
    message: str


NEARBY_MARKETS = {
    "Edmonton": ["St. Albert", "Sherwood Park", "Leduc"],
    "St. Albert": ["Edmonton", "Sherwood Park"],
    "Sherwood Park": ["Edmonton", "St. Albert", "Leduc"],
    "Leduc": ["Edmonton", "Sherwood Park"],
    "Calgary": ["Airdrie", "Red Deer"],
    "Airdrie": ["Calgary", "Red Deer"],
    "Red Deer": ["Calgary", "Airdrie", "Edmonton"],
    "Fort McMurray": ["Edmonton"],
}


def find_candidate_comps(
    subject: SubjectProperty,
    sales_data: pd.DataFrame,
    min_candidates: int = 8,
) -> CandidateSearchResult:
    """
    Find candidate comparable sales using staged fallback logic.

    The goal is to prefer tight, high-quality matches while avoiding hard failure
    when a subject property has sparse local sales.
    """
    if sales_data.empty:
        return CandidateSearchResult(
            candidates=sales_data.copy(),
            search_stage="all_markets_fallback",
            message="No sales data is available for comp search.",
        )

    df = sales_data.copy()
    df["city_norm"] = df["city"].astype(str).str.lower().str.strip()
    df["neighborhood_norm"] = df["neighborhood"].astype(str).str.lower().str.strip()
    df["property_type_norm"] = df["property_type"].astype(str).str.lower().str.strip()

    subject_city = subject.city.lower().strip()
    subject_neighborhood = (subject.neighborhood or "").lower().strip()
    subject_type = subject.property_type.value.lower().strip()

    if subject_neighborhood:
        stage_1 = _base_quality_filter(
            df[
                (df["city_norm"] == subject_city)
                & (df["neighborhood_norm"] == subject_neighborhood)
                & (df["property_type_norm"] == subject_type)
            ],
            months_back=18,
        )
        stage_1 = _size_filter(stage_1, subject, tolerance=0.40)

        if len(stage_1) >= min_candidates:
            return CandidateSearchResult(
                candidates=_drop_helper_columns(stage_1),
                search_stage="same_neighborhood",
                message="Found enough recent same-neighborhood, same-property-type comparables.",
            )

    stage_2 = _base_quality_filter(
        df[(df["city_norm"] == subject_city) & (df["property_type_norm"] == subject_type)],
        months_back=24,
    )
    stage_2 = _size_filter(stage_2, subject, tolerance=0.50)

    if len(stage_2) >= min_candidates:
        return CandidateSearchResult(
            candidates=_drop_helper_columns(stage_2),
            search_stage="same_city",
            message="Neighborhood-level comps were limited, so the search expanded to same-city, same-property-type sales.",
        )

    stage_3 = _base_quality_filter(df[df["city_norm"] == subject_city], months_back=36)
    stage_3 = _size_filter(stage_3, subject, tolerance=0.65)

    if len(stage_3) >= min_candidates:
        return CandidateSearchResult(
            candidates=_drop_helper_columns(stage_3),
            search_stage="same_city_wider",
            message="Same-property-type comps were limited, so the search expanded to similar same-city residential sales.",
        )

    nearby_cities = [city.lower() for city in NEARBY_MARKETS.get(subject.city, [])]
    stage_4 = _base_quality_filter(
        df[df["city_norm"].isin([subject_city, *nearby_cities])],
        months_back=36,
    )
    stage_4 = _size_filter(stage_4, subject, tolerance=0.75)

    if len(stage_4) >= min_candidates:
        return CandidateSearchResult(
            candidates=_drop_helper_columns(stage_4),
            search_stage="nearby_markets",
            message="Local comps were sparse, so the search expanded to nearby Alberta markets. Confidence should be reduced.",
        )

    fallback = _base_quality_filter(df, months_back=48)
    fallback = _size_filter(fallback, subject, tolerance=1.00)

    return CandidateSearchResult(
        candidates=_drop_helper_columns(fallback),
        search_stage="all_markets_fallback",
        message="Very few close comps were available, so the search used a broad Alberta fallback. Treat the valuation as low confidence.",
    )


def _base_quality_filter(df: pd.DataFrame, months_back: int) -> pd.DataFrame:
    """Keep sales within a recency window and with required numeric fields."""
    if df.empty:
        return df.copy()

    result = df.copy()
    latest_date = result["sale_date"].max()
    cutoff = latest_date - pd.DateOffset(months=months_back)

    result = result[result["sale_date"] >= cutoff]
    result = result.dropna(subset=["living_area_sqft", "sale_price", "bedrooms", "bathrooms"])
    result = result[result["living_area_sqft"] > 250]
    result = result[result["sale_price"] > 25000]

    return result


def _size_filter(df: pd.DataFrame, subject: SubjectProperty, tolerance: float) -> pd.DataFrame:
    """Filter out properties with extreme living-area differences."""
    if df.empty:
        return df.copy()

    lower = subject.living_area_sqft * (1 - tolerance)
    upper = subject.living_area_sqft * (1 + tolerance)

    return df[(df["living_area_sqft"] >= lower) & (df["living_area_sqft"] <= upper)].copy()


def _drop_helper_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove internal normalized columns before returning candidates."""
    helper_columns = ["city_norm", "neighborhood_norm", "property_type_norm"]
    return df.drop(columns=[col for col in helper_columns if col in df.columns]).reset_index(drop=True)