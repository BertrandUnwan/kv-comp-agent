import pandas as pd

from kv_comp_agent.schema import PropertyType, SubjectProperty
from kv_comp_agent.valuation import estimate_value


def test_estimate_value_returns_range():
    subject = SubjectProperty(
        city="Edmonton",
        property_type=PropertyType.DETACHED,
        bedrooms=4,
        bathrooms=3,
        living_area_sqft=2000,
    )

    comps = pd.DataFrame(
        {
            "sale_price": [700000, 720000, 680000, 710000, 705000],
            "living_area_sqft": [2000, 2050, 1950, 1980, 2020],
            "price_per_sqft": [350, 351.22, 348.72, 358.59, 349.01],
            "total_score": [95, 90, 88, 86, 84],
            "recency_score": [95, 90, 85, 80, 75],
            "property_type_score": [100, 100, 100, 100, 100],
            "location_score": [100, 100, 90, 90, 90],
            "living_area_score": [100, 95, 95, 98, 98],
        }
    )

    valuation = estimate_value(subject, comps)

    assert valuation.low_estimate < valuation.base_estimate < valuation.high_estimate
    assert valuation.confidence in {"High", "Medium", "Low"}
    assert valuation.base_estimate > 0


def test_empty_comps_returns_low_confidence_zero_estimate():
    subject = SubjectProperty(
        city="Edmonton",
        property_type=PropertyType.DETACHED,
        bedrooms=4,
        bathrooms=3,
        living_area_sqft=2000,
    )

    valuation = estimate_value(subject, pd.DataFrame())

    assert valuation.base_estimate == 0
    assert valuation.confidence == "Low"
    assert valuation.confidence_score == 0.0