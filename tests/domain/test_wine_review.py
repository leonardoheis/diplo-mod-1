"""Tests for WineReview domain model."""

import pytest
from pydantic import ValidationError

from diplo_mod_1.constants import POINTS_MAX, POINTS_MIN
from diplo_mod_1.domain.wine_review import WineReview

_VALID = dict(
    description="Aromas of dark fruit and tobacco.",
    points=90,
    title="Chateau A 2015 Reserve",
    winery="Chateau A",
)


def test_valid_instance_creates() -> None:
    review = WineReview(**_VALID)
    assert review.points == 90
    assert review.winery == "Chateau A"


def test_optional_fields_default_to_none() -> None:
    review = WineReview(**_VALID)
    assert review.country is None
    assert review.designation is None
    assert review.taster_name is None


def test_points_below_minimum_raises() -> None:
    with pytest.raises(ValidationError):
        WineReview(**{**_VALID, "points": POINTS_MIN - 1})


def test_points_above_maximum_raises() -> None:
    with pytest.raises(ValidationError):
        WineReview(**{**_VALID, "points": POINTS_MAX + 1})


def test_negative_price_raises() -> None:
    with pytest.raises(ValidationError):
        WineReview(**{**_VALID, "price": -1.0})


def test_zero_price_raises() -> None:
    with pytest.raises(ValidationError):
        WineReview(**{**_VALID, "price": 0.0})


def test_frozen_rejects_mutation() -> None:
    review = WineReview(**_VALID)
    with pytest.raises(ValidationError):
        review.points = 91  # type: ignore[misc]
