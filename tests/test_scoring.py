import pandas as pd

from kv_comp_agent.schema import PropertyType, SubjectProperty
from kv_comp_agent.scoring import (
    score_living_area,
    score_property_type,
    score_recency,
)


def test_same_property_type_scores_higher_than_different_type():
    subject = SubjectProperty(
        city="Edmonton",
        property_type=PropertyType.DETACHED,
        bedrooms=4,
        bathrooms=3,
        living_area_sqft=2100,
    )

    same_type = pd.Series({"property_type": "Detached"})
    different_type = pd.Series({"property_type": "Condo"})

    assert score_property_type(subject, same_type) > score_property_type(subject, different_type)


def test_similar_living_area_scores_higher_than_distant_living_area():
    subject = SubjectProperty(
        city="Edmonton",
        property_type=PropertyType.DETACHED,
        bedrooms=4,
        bathrooms=3,
        living_area_sqft=2000,
    )

    close_comp = pd.Series({"living_area_sqft": 2050})
    far_comp = pd.Series({"living_area_sqft": 3200})

    assert score_living_area(subject, close_comp) > score_living_area(subject, far_comp)


def test_recent_sale_scores_higher_than_old_sale():
    recent = pd.Series({"sale_date": pd.Timestamp.today() - pd.Timedelta(days=30)})
    old = pd.Series({"sale_date": pd.Timestamp.today() - pd.Timedelta(days=900)})

    assert score_recency(recent) > score_recency(old)