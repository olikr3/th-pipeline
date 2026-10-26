# thp

The inventory risk engine: canonical data, demand features, demand forecasts, and a cash-conversion simulator that scores lots. See the [technical spec](../docs/specs/inventory-risk-engine.md) and the [ADRs](../docs/adr/).

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
| `thp forecast --tenant T --as-of D` | Writes the seasonal-naive baseline: q05…q95 for the next 14 weeks per SKU × location |
| `thp ingest-forecasts --tenant T --source S --file F [--issued D]` | Loads forecasts from another source into canonical Parquet |
| `thp forecast --tenant T --as-of D --source S` | Writes forecasts from source `S`, re-based onto `as_of` |
| `thp make-demo [--skus N --weeks W --seed S]` | Generates the synthetic demo tenant `demo` and three lot requests in `data/demo/lots/` |
| `thp score-lot LOT.json --tenant T [--source S]` | Scores a lot request: forecast, simulate, advance; prints JSON |

`--as-of` is the date the computation pretends to be at: only data dated on or before it is read. Horizon 1 is the week after `as_of`'s last complete week.

## Demo: score a lot end to end

```
uv run thp make-demo
uv run thp score-lot data/demo/lots/po_disciplined.json --tenant demo
uv run thp score-lot data/demo/lots/po_overbought.json --tenant demo
uv run thp score-lot data/demo/lots/stock_slow_movers.json --tenant demo
```

`make-demo` writes a synthetic merchant (40 SKUs, two locations, three years) with every canonical table, including weekly stock snapshots and purchase orders that M5 lacks. Demand is seasonal and partly intermittent with shared shocks; an order-up-to buyer with disciplined, average and overbuying SKUs creates real stock-outs and excess stock. See `src/thp/ingest/demo.py`.

The three lots, as of the demo's last day:

| Lot | What it is | Tenor |
|---|---|---|
| `po_disciplined` | a new PO for the 12 WH1 best sellers, 8 weeks of recent demand, arriving in 6 weeks | 90 days |
| `po_overbought` | the same PO at 30 weeks of demand | 90 days |
| `stock_slow_movers` | stock financing of on-hand stock covering over 20 weeks | 60 days |

`score-lot` prints the lot cost, the cash-ratio quantiles at 30/60/90 days, P(cash ratio < 1), expected unsold value, and the advance: q05 of the cash ratio at the tenor × lot cost ÷ (1 + fee), capped at 80% of cost. The decision rules (decline, reduce, refer) are not built yet, so every lot gets an advance amount, however small. A lot request is the spec's API request as JSON; add `"options": {"n_paths": 500, "seed": 1}` to change the run.

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

Output goes to `data/forecasts/forecast/external_<source>/<tenant>/as_of=<date>/forecast.parquet`. It has the same columns as the baseline's `seasonal_naive_v2` output.

## Layout

```
src/thp/
  core/      as-of data access, config, seeding, tenant weeks
  schema/    canonical table schemas and the forecast quantiles
  ingest/    M5 connector, external forecast ingest, synthetic demo tenant
  features/  weekly demand features
  demand/    seasonal-naive baseline, external forecasts, forecast output
  simulate/  cash-conversion simulator: demand sampling, copula, engine, lot assembly
  underwrite/ advance amount
  scoring.py  forecast -> assemble -> simulate -> advance
```
