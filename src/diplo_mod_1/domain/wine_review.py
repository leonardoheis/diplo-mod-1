"""WineReview — validated representation of a single raw wine review."""

from __future__ import annotations

from pydantic import Field

from diplo_mod_1.domain.base import BaseDomainModel


class WineReview(BaseDomainModel):
    """A single wine review as it appears in the raw CSV.

    Fields mirror the source dataset columns that the pipeline consumes.
    Optional fields are ``None`` where the raw data may be missing; the
    ``DataCleaner`` handles downstream imputation.

    Usage::

        review = WineReview(
            description="Aromas include tropical fruit...",
            points=90,
            title="Nicosia 2013 Vulkà Bianco",
            winery="Nicosia",
        )
    """

    country: str | None = None
    description: str
    designation: str | None = None
    points: int = Field(ge=80, le=100)
    price: float | None = Field(default=None, gt=0)
    province: str | None = None
    region_1: str | None = None
    taster_name: str | None = None
    title: str
    variety: str | None = None
    winery: str
