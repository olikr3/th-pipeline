"""Tenant weeks: a week runs from the tenant's ``week_start`` weekday and is labelled
by its last day (``week_end``)."""

from datetime import date, timedelta

import polars as pl


def last_complete_week_end(as_of: date, week_start: int) -> date:
    """The last day of the latest week that ends on or before ``as_of``."""
    offset = (as_of.isoweekday() - week_start) % 7
    return as_of if offset == 6 else as_of - timedelta(days=offset + 1)


def week_end(day: pl.Expr, week_start: int) -> pl.Expr:
    """The last day of the week containing ``day``."""
    days_into_week = (day.dt.weekday() - week_start + 7) % 7
    return day + pl.duration(days=6 - days_into_week)
