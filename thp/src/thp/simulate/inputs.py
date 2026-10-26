"""Simulator inputs from engine tables.

``forecast_knots`` turns a ``forecast`` table into the knots array the simulator
samples from. A lot is refused, not patched, when any of its series lacks a week
the tenor needs: a missing week from an external source is not filled from
another source.
"""

from collections.abc import Sequence

import numpy as np
import polars as pl

from thp.schema import QUANTILE_COLS
from thp.simulate.quantiles import FloatArray, check_knots

# How many missing (series, week) pairs an error message lists.
MAX_LISTED = 10


class MissingForecastError(ValueError):
    """The forecast does not cover every series and week the lot needs."""


def series_key(sku_id: str, location_id: str) -> str:
    """The id a SKU x location is simulated (and seeded) under."""
    return f"{sku_id}|{location_id}"


def forecast_knots(
    forecast: pl.DataFrame, series: Sequence[tuple[str, str]], n_weeks: int
) -> FloatArray:
    """Knots ``[len(series), n_weeks, 7]`` for (sku_id, location_id) pairs, horizons 1..n_weeks.

    ``forecast`` must hold one model version for one as_of. Raises
    ``MissingForecastError`` when any series lacks any of the horizons.
    """
    for col in ("model_version", "as_of"):
        distinct = forecast[col].unique()
        if len(distinct) > 1:
            raise ValueError(f"forecast mixes {col} values {distinct.to_list()}")
    wanted = pl.DataFrame(
        {
            "sku_id": [sku for sku, _ in series for _ in range(n_weeks)],
            "location_id": [loc for _, loc in series for _ in range(n_weeks)],
            "horizon_week": list(range(1, n_weeks + 1)) * len(series),
        },
        schema={"sku_id": pl.String, "location_id": pl.String, "horizon_week": pl.Int16},
    )
    rows = wanted.join(
        forecast.select("sku_id", "location_id", "horizon_week", *QUANTILE_COLS),
        on=["sku_id", "location_id", "horizon_week"],
        how="left",
        maintain_order="left",
    )
    missing = rows.filter(pl.any_horizontal(pl.col(QUANTILE_COLS).is_null()))
    if missing.height:
        listed = ", ".join(
            f"{sku}|{loc} week {h}"
            for sku, loc, h in missing.select("sku_id", "location_id", "horizon_week")
            .head(MAX_LISTED)
            .iter_rows()
        )
        more = f" and {missing.height - MAX_LISTED} more" if missing.height > MAX_LISTED else ""
        raise MissingForecastError(
            f"forecast lacks {missing.height} of {rows.height} series-weeks the lot needs: "
            f"{listed}{more}"
        )
    values = rows.select(QUANTILE_COLS).to_numpy().astype(np.float32)
    return check_knots(np.reshape(values, (len(series), n_weeks, len(QUANTILE_COLS))))
