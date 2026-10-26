# Tradeoffs

Choices made between alternatives, with what was given up. Newest first. See [the observations index](README.md) for the entry format.

## Refuse lots with missing forecast weeks

- **Date:** 2026-10-25
- **Where:** `thp/src/thp/simulate/inputs.py`, `forecast_knots`
- **Chosen:** a lot whose lines lack any forecast week its tenor needs is refused with `MissingForecastError`.
- **Alternative:** fill the missing weeks from the baseline and flag the share of filled demand.
- **Given up:** a source that forecasts fewer weeks than the longest tenor can only score shorter tenors.
- **Gained:** no lot is scored on a mix of sources it didn't ask for.
- **Revisit if:** many lots are refused because an otherwise good source has a short horizon.

## Goods sell from the next full week after arrival

- **Date:** 2026-10-25
- **Where:** `thp/src/thp/simulate/engine.py`, `receipt_week_probs`
- **Chosen:** goods arriving during a week sell from the next full week.
- **Alternative:** sell the remaining fraction of the arrival week.
- **Given up:** up to 6 days of sales per lot, so cash ratios are slightly low.
- **Gained:** simpler weekly mechanics, and a bias in the lender's favour.
- **Revisit if:** short tenors (30 days) on PO lots turn out systematically under-predicted in the backtest.

## One markdown step per 4 weeks

- **Date:** 2026-10-25
- **Where:** `simulate.markdown` in `thp/configs/default.yaml`
- **Chosen:** a 20% step of list price, at most once per `interval_weeks` (4), down to 50% off.
- **Alternative:** re-evaluate every week, as the spec's wording allows.
- **Given up:** faster reactions to very slow stock.
- **Gained:** without a cadence, slow stock would hit the 50% floor within 3 weeks.
- **Revisit if:** merchants' markdown histories (`markdown_event`) show a different cadence.

## No B2B receivable delays yet

- **Date:** 2026-10-25
- **Where:** `thp/src/thp/simulate/engine.py`
- **Chosen:** all cash counts in the week of the sale.
- **Alternative:** draw a receivable delay for the B2B share, as the spec describes.
- **Given up:** cash ratios are optimistic for merchants with a large B2B share on payment terms.
- **Gained:** no model built on data that doesn't exist yet. The spec lists receivables as optional Phase 3+, and M5 is all B2C.
- **Revisit if:** a pilot merchant sells B2B on terms.

## Assumed per-SKU persistence parameters

- **Date:** 2026-10-25
- **Where:** `simulate.rho_sku`, `simulate.sku_persistence` in `thp/configs/default.yaml`
- **Chosen:** ρs = 0.20 and φs = 0.9, set by judgement.
- **Alternative:** wait for PIT-based estimates before adding the component.
- **Given up:** the values are not evidence-based.
- **Gained:** the dominant over-lending effect (see [wrong assumptions](wrong-assumptions.md)) is covered now, rather than left at zero.
- **Revisit when:** matured forecasts allow estimating them from the within-series autocorrelation of PIT values.

## 64-bit hashed substream keys

- **Date:** 2026-10-25
- **Where:** `thp/src/thp/simulate/copula.py`, `stream_key`
- **Chosen:** a 64-bit BLAKE2b hash of each category and series id.
- **Alternative:** CRC32, or Python's `hash()`.
- **Given up:** nothing material; slightly more code than `hash()`.
- **Gained:** `hash()` changes between processes, which breaks replay. With 32 bits, some of 50,000 SKUs would share draws (a birthday collision is likely).

## Truncated upper demand tail

- **Date:** 2026-10-18
- **Where:** `thp/src/thp/simulate/quantiles.py`, `U_EPS = 1e-4`
- **Chosen:** clip draws to [1e-4, 1 − 1e-4].
- **Alternative:** clip at 1e-6 and compute in float64.
- **Given up:** demand beyond about 9 halvings of tail probability past q95, about 1 draw in 10,000.
- **Gained:** float32 accuracy (see [bugs](bugs.md)) and half the memory of float64.

## Exponential tails beyond q05 and q95

- **Date:** 2026-10-18
- **Where:** `thp/src/thp/simulate/quantiles.py`
- **Chosen:** the spec's "log-linear tail", read as Q linear in log(1−u) above q95 and in log(u) below q05, continuing the outermost step per halving of tail probability.
- **Alternative:** a heavier tail, linear in log-quantity.
- **Given up:** may understate big orders for lumpy SKUs.
- **Gained:** works with quantiles of 0, which log-quantity tails don't.
- **Revisit if:** the backtest shows too many realised outcomes above q95 for lumpy SKUs.

## Gaussian copula, not t-copula

- **Date:** 2026-10-18
- **Where:** `thp/src/thp/simulate/copula.py`
- **Chosen:** a Gaussian factor copula.
- **Alternative:** a t-copula, with one extra parameter ν.
- **Given up:** joint extremes are under-represented: at ρ = 0.25, given one SKU is below its q01, another is too only 4% of the time, against 17% under a t-copula with ν = 4. See [the copula](../copula.md).
- **Gained:** simpler estimation and explanation. Underwriting uses the moderate q05 tail.
- **Revisit if:** realised cash falls below q05 in more than 7% of lots in bad quarters.

## Missing external quantiles filled from the baseline's spread

- **Date:** 2026-10-18
- **Where:** `thp/src/thp/demand/external.py`; [ADR 0002](../../docs/adr/0002-baseline-and-external-forecasts.md)
- **Chosen:** scale the baseline's quantile-to-median ratios to the source's centre.
- **Alternative:** the source's own past errors, or a fixed relative spread.
- **Given up:** the spread reflects the baseline's uncertainty, not the source's. SKUs without sales history and a point forecast only are skipped.
- **Gained:** works for a new tenant on day one.
- **Revisit when:** months of the source's issued forecasts can be compared with realised sales.

## Baseline spread does not widen with the horizon

- **Date:** 2026-10-18
- **Where:** `thp/src/thp/demand/baseline.py`
- **Chosen:** one set of error quantiles for all 13 horizons.
- **Alternative:** errors per horizon.
- **Given up:** understates the four-week-mean fallback's spread at long horizons. Seasonal naive is exact.
- **Gained:** simpler, and per-horizon errors would need more history than most series have.
