import pytest
from pydantic import ValidationError

from kv_comp_agent.schema import PropertyType, SubjectProperty


def test_valid_subject_property_creation():
    subject = SubjectProperty(
        city="Edmonton",
        neighborhood="Glenora",
        property_type=PropertyType.DETACHED,
        bedrooms=4,
        bathrooms=3,
        living_area_sqft=2100,
        lot_size_sqft=5500,
        year_built=2015,
    )

    assert subject.city == "Edmonton"
    assert subject.property_type == PropertyType.DETACHED
    assert subject.living_area_sqft == 2100


def test_invalid_living_area_rejected():
    with pytest.raises(ValidationError):
        SubjectProperty(
            city="Edmonton",
            property_type=PropertyType.DETACHED,
            bedrooms=4,
            bathrooms=3,
            living_area_sqft=100,
        )


def test_future_year_built_rejected():
    with pytest.raises(ValidationError):
        SubjectProperty(
            city="Edmonton",
            property_type=PropertyType.DETACHED,
            bedrooms=4,
            bathrooms=3,
            living_area_sqft=2100,
            year_built=2035,
        )