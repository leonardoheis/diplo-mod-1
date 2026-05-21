"""Shared fixtures available to all tests."""

import pandas as pd
import pytest


@pytest.fixture(scope="session")
def small_raw_df() -> pd.DataFrame:
    """60-row Wine Reviews DataFrame covering all 5 points bins.

    Includes deliberate NaN values in price, designation, and taster_name.
    Titles mix entries with 4-digit vintage years (2010-2017) and without.
    """
    n = 60
    # 12 rows per bin: (79,85], (85,88], (88,91], (91,94], (94,100]
    points = (
        [80, 81, 82, 83, 84, 85, 80, 81, 82, 83, 84, 85]
        + [86, 87, 88, 86, 87, 88, 86, 87, 88, 86, 87, 88]
        + [89, 90, 91, 89, 90, 91, 89, 90, 91, 89, 90, 91]
        + [92, 93, 94, 92, 93, 94, 92, 93, 94, 92, 93, 94]
        + [95, 96, 97, 98, 99, 100, 95, 96, 97, 98, 99, 100]
    )

    countries = ["US"] * 30 + ["France"] * 15 + ["Italy"] * 15
    varieties = ["Cabernet Sauvignon"] * 20 + ["Pinot Noir"] * 20 + ["Chardonnay"] * 20
    wineries = ["Chateau A"] * 20 + ["Domaine B"] * 20 + ["Villa C"] * 20

    titles = []
    for i in range(n):
        if i % 5 == 0:
            titles.append(f"Winery {i} NV Reserve")
        else:
            year = 2010 + (i % 8)
            titles.append(f"Winery {i} {year} Reserve")

    prices: list[float | None] = []
    for i in range(n):
        prices.append(None if i % 7 == 0 else 10.0 + i * 2.0)

    designations: list[str | None] = [None if i % 4 == 0 else f"Reserve {i}" for i in range(n)]
    taster_names: list[str | None] = [None if i % 5 == 0 else f"Taster {i % 3}" for i in range(n)]

    return pd.DataFrame(
        {
            "country": countries,
            "description": [
                f"Aromas of cherry and blackberry. Long finish. Wine {i}." for i in range(n)
            ],
            "designation": designations,
            "points": points,
            "price": prices,
            "province": ["California"] * 30 + ["Burgundy"] * 15 + ["Tuscany"] * 15,
            "region_1": [f"Region {i % 5}" for i in range(n)],
            "region_2": [None] * n,
            "taster_name": taster_names,
            "taster_twitter_handle": [f"@taster{i % 3}" for i in range(n)],
            "title": titles,
            "variety": varieties,
            "winery": wineries,
        }
    )
