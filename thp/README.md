# thp

The inventory risk engine: canonical data, demand features and demand forecasts. See the [technical spec](../docs/specs/inventory-risk-engine.md) and the [ADRs](../docs/adr/).

## Setup

```
uv sync
uv run pytest          # also: uv run ruff check, uv run ruff format --check, uv run mypy
```

Configuration is layered YAML: `configs/default.yaml`, then any `--config` files, later layers winning.

## Commands

All data lives under `data/` by default: `data/canonical/` for canonical tables, `data/features/` and `data/forecasts/` for engine output.

| Command | Does |
| --- | --- |
| `thp ingest-m5 [--stores CA_1 ...]` | Converts the M5 CSVs in `data/raw/m5/` into canonical Parquet for tenant `m5` |
| `thp build-features --tenant T --as-of D [--latest-only]` | Writes the weekly demand feature panel (or only its last week) |
| `thp forecast --tenant T --as-of D` | Writes the seasonal-naive baseline: q05…q95 for the next 13 weeks per SKU × location |
| `thp ingest-forecasts --tenant T --source S --file F [--issued D]` | Loads forecasts from another source into canonical Parquet |
| `thp forecast --tenant T --as-of D --source S` | Writes forecasts from source `S`, re-based onto `as_of` |

`--as-of` is the date the computation pretends to be at: only data dated on or before it is read. Horizon 1 is the week after `as_of`'s last complete week.

## Forecasts from another source

A merchant's planning tool or ERP can supply forecasts in place of the engine's own. Upload a CSV or Parquet file with one row per SKU, location and target week:

| Column | Required | Meaning |
| --- | --- | --- |
| `sku_id`, `location_id` | yes | The series |
| `week` | yes | Any date in the target week; mapped to the tenant's week end |
| `qty` | `qty` or `q50` | The source's point forecast, in units |
| `q05`, `q10`, `q25`, `q50`, `q75`, `q90`, `q95` | no | Demand quantiles, in units |
| `issued_date` | no | When the source made the forecast; otherwise pass `--issued` |

```
sku_id,location_id,week,qty
FOODS_1_001,CA_1,2026-10-20,12
FOODS_1_001,CA_1,2026-10-27,15
```

```
thp ingest-forecasts --tenant m5 --source planner --file planner.csv --issued 2026-10-15
thp forecast --tenant m5 --as-of 2026-10-18 --source planner
```

How the forecast is formed:

- **Which issue.** Each series uses the latest issue on or before `as_of`, no older than `demand.external_max_age_days` (35). Issues dated after `as_of` are invisible, so backtests only see forecasts that existed at the time.
- **Missing quantiles.** These come from the baseline's spread for the same series and week, scaled to the source's level. Quantiles in the file are kept.
- **Skipped rows.** A SKU with no sales history and a point forecast only has no spread to borrow. It is skipped, and the command prints how many series were affected.
- **Uploads.** Re-uploading the same series, issue date and week replaces the earlier values. Source names use lowercase letters, digits and `_`.

The whole file is rejected if a row has neither `qty` nor `q50`, a value is negative, quantiles decrease, or a series repeats a week. Forecasts must be weekly.

Output goes to `data/forecasts/forecast/external_<source>/<tenant>/as_of=<date>/forecast.parquet`. It has the same columns as the baseline's `seasonal_naive_v1` output.

## Layout

```
src/thp/
  core/      as-of data access, config, seeding, tenant weeks
  schema/    canonical table schemas and the forecast quantiles
  ingest/    M5 connector, external forecast ingest
  features/  weekly demand features
  demand/    seasonal-naive baseline, external forecasts, forecast output
```
