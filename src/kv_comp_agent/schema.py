from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, computed_field, field_validator


class PropertyType(str, Enum):
    DETACHED = "Detached"
    SEMI_DETACHED = "Semi-Detached"
    TOWNHOUSE = "Townhouse"
    CONDO = "Condo"
    DUPLEX = "Duplex"


class PropertyCondition(str, Enum):
    POOR = "Poor"
    FAIR = "Fair"
    AVERAGE = "Average"
    GOOD = "Good"
    EXCELLENT = "Excellent"


class SubjectProperty(BaseModel):
    """
    Input property provided by the user.

    This represents the property we want to value using comparable sales.
    """

    city: str = Field(..., min_length=2)
    neighborhood: Optional[str] = None
    property_type: PropertyType
    bedrooms: int = Field(..., ge=0, le=12)
    bathrooms: float = Field(..., ge=0, le=10)
    living_area_sqft: int = Field(..., ge=250, le=12000)
    lot_size_sqft: Optional[int] = Field(default=None, ge=0, le=100000)
    year_built: Optional[int] = Field(default=None, ge=1850, le=2026)
    garage_spaces: int = Field(default=0, ge=0, le=6)
    condition: PropertyCondition = PropertyCondition.AVERAGE
    finished_basement: bool = False
    renovated: bool = False
    near_transit: bool = False
    backs_onto_park: bool = False

    @field_validator("city", "neighborhood")
    @classmethod
    def clean_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        value = value.strip()
        return value if value else None


class ComparableProperty(BaseModel):
    """
    One historical sold property from the dataset.
    """

    property_id: str
    address: str
    city: str
    neighborhood: str
    latitude: float
    longitude: float
    property_type: PropertyType
    bedrooms: int = Field(..., ge=0, le=12)
    bathrooms: float = Field(..., ge=0, le=10)
    living_area_sqft: int = Field(..., ge=250, le=12000)
    lot_size_sqft: Optional[int] = Field(default=None, ge=0, le=100000)
    year_built: Optional[int] = Field(default=None, ge=1850, le=2026)
    sale_date: date
    sale_price: int = Field(..., ge=25000)
    garage_spaces: int = Field(default=0, ge=0, le=6)
    condition: PropertyCondition = PropertyCondition.AVERAGE
    finished_basement: bool = False
    renovated: bool = False
    near_transit: bool = False
    backs_onto_park: bool = False

    @computed_field
    @property
    def price_per_sqft(self) -> float:
        return round(self.sale_price / self.living_area_sqft, 2)


class ValuationEstimate(BaseModel):
    """
    Final valuation estimate returned by the engine.
    """

    low_estimate: int
    base_estimate: int
    high_estimate: int
    confidence: str
    confidence_score: float = Field(..., ge=0, le=1)
    methodology: str
    risk_flags: list[str] = Field(default_factory=list)


class CompAnalysisResult(BaseModel):
    """
    Full output of the comp-analysis pipeline.
    """

    subject: SubjectProperty
    valuation: ValuationEstimate
    top_comps: list[dict]
    memo: str