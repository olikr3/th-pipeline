# ADR 0002: Seasonal-naive baseline and forecasts from external sources

- **Status:** Accepted
- **Date:** 2026-10-18
- **Relates to:** [Inventory Risk Engine Technical Spec](../specs/inventory-risk-engine.md), sections "Demand model" and "Canonical data model", principles 2 (distributions, not points) and 4 (time-honest), Milestone 3; [ADR 0001](0001-as-of-access-seeding-and-parquet.md)

## Context

The simulator (Milestone 5) needs a demand distribution per SKU × location for the next 13 weeks. The spec plans a LightGBM quantile model for this, and says every model must be compared against a seasonal-naive baseline. Neither existed after Milestone 2: the pipeline ended at the weekly feature panel.

Separately, many merchants already forecast demand in a planning tool or ERP. Those forecasts carry knowledge the sales history does not, such as planned promotions, listings and delistings. The engine should be able to score with them instead of, or alongside, its own model. Three questions needed answers:

1. What the engine forecasts with before the LightGBM model exists.
2. How external forecasts enter the engine without breaking the as-of guarantee.
3. How a point forecast, which is what most tools export, becomes the distribution the simulator needs.

## Decision 1: a seasonal-naive baseline is the first forecast source

`seasonal_naive` (`thp/src/thp/demand/baseline.py`) writes `forecast` rows with model version `seasonal_naive_v1`:

- **Point:** demand in the same week a year earlier. Series with under 65 weeks of history (a year plus 13 weeks of errors) use the mean of their last four weeks.
- **Quantiles:** the point plus the empirical quantiles of that method's errors over the trailing `demand.residual_weeks` (default 52), clipped at zero.
- **Cold start:** series with under 13 weeks of history have too few errors of their own. They take the error quantiles of their category × location, relative to the forecast level, falling back to the whole tenant. They are flagged `is_cold_start`.

Horizon 1 is the week after `as_of`'s last complete week, matching the last row of the feature panel.

**Amended 2026-10-26 (`seasonal_naive_v2`):** the seasonal point is scaled by the series' level change, the mean of its last 8 weeks over the same weeks a year earlier, clipped to [0.1, 4]. Plain seasonal naive forecast discontinued SKUs at last year's level, so the simulator expected dead stock to sell through (see `theory/observations/wrong-assumptions.md`). The default horizon is also 14 weeks, not 13, so that a 90-day tenor from a mid-week `as_of` is covered.

**Why.** The spec needs this baseline anyway as the bar for LightGBM, and it gives Milestones 4–5 real forecasts to build on now. It needs no new dependencies and no training.

**Known limitation.** The spread does not widen with the horizon. That is exact for seasonal naive, whose forecast for a week uses the week a year earlier whatever the origin, but it understates the spread of the four-week mean at long horizons.

## Decision 2: external forecasts are a fact table dated by their issue date

External forecasts are ingested into a canonical `external_forecast` table (`thp/src/thp/schema/canonical.py`), one row per source, series, issue date and target week. `AsOfStore` treats it as a fact table with `issued_date` as its event date. A forecast is therefore invisible to any `as_of` before the source made it.

At forecast time (`external_forecast` in `thp/src/thp/demand/external.py`), each series uses the latest issue on or before `as_of`, and no older than `demand.external_max_age_days` (default 35). Its target weeks are re-based onto `as_of`'s horizons. Weeks already in the past are dropped, and weeks the source did not forecast stay missing rather than being filled. Ingest batches are appended, and when the same key arrives twice the later `ingested_at` wins. The output is an ordinary `forecast` table with model version `external_<source>`, so downstream code does not know where a forecast came from (principle 1).

**Why.** A backtest at a past `as_of` must only use forecasts that existed then. Storing the issue date and reading through `AsOfStore` gives the same guarantee as for sales, with no extra code in the forecast logic. Keeping each issue, rather than overwriting with the latest, makes the source's accuracy measurable later by comparing old issues with realised sales.

**Alternatives considered.**

- *Overwrite one current forecast per source.* Rejected: it leaks the future into backtests and discards the history needed to measure the source.
- *Import external forecasts straight into the `forecast` table.* Rejected: `forecast` rows are tied to an `as_of` and the engine's horizon convention, while sources issue on their own schedule.

## Decision 3: missing quantiles are filled from the baseline's spread

Each external row needs a point forecast (`qty`) or a median (`q50`); any other quantile is optional. A missing quantile is the source's centre (`q50`, else `qty`) scaled by the baseline's ratio of that quantile to its median, for the same series and horizon. Where the baseline median is zero it is shifted by their difference instead. Quantiles the source gives are kept. The quantiles are then clipped at zero and sorted so they cannot cross.

**Why.** The simulator needs a full distribution (principle 2), and most planning tools export a single number. The baseline's spread is the best available estimate of how uncertain the series is. Scaling by ratio rather than shifting keeps the relative uncertainty when the source's level differs from the baseline's.

**Alternatives considered.**

- *Spread from the source's own past errors.* Preferred in the long run, but it needs months of issued forecasts compared with realised sales, which a new tenant does not have. This should replace the baseline spread once that history exists.
- *A fixed relative spread (e.g. ±30 %).* Rejected: it ignores how much series differ in volatility, especially intermittent ones.

## Consequences

- A lot whose lines lack any forecast week its tenor needs is refused by the simulator (`MissingForecastError` in `thp/src/thp/simulate/inputs.py`), not filled from the baseline or another source. A source that forecasts fewer weeks than the longest tenor needs can only score shorter tenors.
- Rows that still miss a quantile after filling are skipped and reported. In practice these are SKUs without sales history when the source sends a point forecast only. New SKUs are therefore covered only by sources that send every quantile.
- External forecasts must be weekly. Daily or monthly forecasts have to be converted before ingest, because quantiles cannot be summed across periods. Any date in the target week is accepted and mapped to the tenant's week end.
- Ingest rejects a whole file on any invalid row (no centre, negative value, decreasing quantiles, a repeated series and week). The spec's quarantine-and-report validation can replace this when the CSV/Excel connector is built (Milestone 8).
- Computing an external forecast always computes the baseline first, for its spread.
- `forecast` and `external_forecast` are stored as Parquet under `data/forecasts/forecast/{model_version}/{tenant}/as_of={date}/` and `data/canonical/external_forecast/{tenant}/{source}/` until Postgres arrives (ADR 0001, Decision 3).
- The `forecast` table adds `week_end` and `is_cold_start` to the columns listed in the spec, and `external_forecast` is not yet in the spec's data model. The spec should be updated to match.
- The week helpers moved to `thp/src/thp/core/weeks.py` so that ingest and features share one definition of a tenant week.
