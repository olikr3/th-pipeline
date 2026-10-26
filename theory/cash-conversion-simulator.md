# Cash-conversion simulator: how it works

For what the engine is for as a whole, start with [what the engine solves](what-the-engine-solves.md).

How the Monte Carlo simulator in the [technical spec](../docs/specs/inventory-risk-engine.md) ("Cash-conversion simulator") turns demand forecasts into a cash-ratio distribution, with the sampling step in detail. It assumes forecasts are already provided: the `forecast` table from any source (baseline or external, see [ADR 0002](../docs/adr/0002-baseline-and-external-forecasts.md)).

## What goes in and what comes out

A **lot** is a purchase order to be financed: lines of (SKU, location, qty, unit_cost), an `as_of` date (the order date) and a lot cost V.

The simulator answers: *across 2,000 plausible futures, what fraction of V comes back as cash by day 30, 60 and 90?*

**Inputs per lot line:**

- the forecast quantiles q05…q95 for horizons 1–13, from the `forecast` table at the lot's `as_of`;
- the current price;
- existing stock ahead of the lot (on hand plus earlier open POs);
- a return rate;
- a receipt-week distribution (when the goods arrive).

**Output:** cash_ratio quantiles at d30/d60/d90, P(cash_ratio_90 < 1) and expected unsold value. Underwriting turns these into an advance: L\* = q05(C_T) · V / (1+f).

Everything runs on arrays shaped `[N paths, S SKUs, W weeks]`. A 200-line lot is 2000 × 200 × 13 ≈ 5M float32 values, about 21 MB per array, so it fits in memory at once.

## The steps

| Step | What happens |
|---|---|
| 1. Shocks | Draw the shared random factors: one for the merchant's season per path, one per category per path. |
| 2. Correlate | Mix the shared factors with noise per SKU and week into a correlated uniform value `u ∈ (0,1)`. |
| 3. **Sample demand** | Turn each `u` into units using that SKU-week's forecast quantiles (the inverse CDF). |
| 4. Receipt | Draw the week the lot arrives; lot units can't sell before it. |
| 5. Sell | Each week: sales = min(demand, available). Older stock sells first (FIFO), then lot stock. Unmet demand is lost, not carried over. |
| 6. Markdown | If lot stock is older than 12 weeks or cover exceeds 26 weeks, cut the price 20% (up to 50% in total) and scale demand by elasticity. |
| 7. Cash | Revenue from lot units × (1 − return rate); B2B cash is delayed by a sampled receivable lag. |
| 8. Aggregate | Cumulative cash at days 30/60/90 ÷ V gives 2,000 cash_ratio samples per tenor; take their quantiles. |

Steps 2 and 3 are where the forecasts enter.

## Step 2: correlated uniforms (Gaussian copula)

For every random draw, a worked example over two paths and the seeding scheme, see [the draw logic](simulator-draws.md).
For why the shared shocks are needed at all, see [why shocks](why-shocks.md).
For what the copula is, how ρ is estimated and its tail weakness, see [the copula](copula.md).

For SKU *i* in category *c*, week *t*, path *n*:

```
x[n,i,t] = √ρm · zm[n,t] + √ρc · zc[n,c,t] + √(1−ρm−ρc) · ε[n,i,t]
u[n,i,t] = Φ(x[n,i,t])
```

- **zm:** the merchant-season shock. In a bad path, everything sells worse.
- **zc:** a category shock, e.g. one category falling out of fashion.
- **ε:** independent noise per SKU and week.

**Why it works:** the weights square-sum to 1 (ρm + ρc + (1−ρm−ρc) = 1), so `x` is exactly standard normal and `u` is exactly uniform. Each SKU-week's demand therefore follows its forecast distribution exactly, whatever the correlation. The copula only decides *which* SKUs have bad weeks together.

With the defaults ρm = 0.15 and ρc = 0.10, two SKUs in the same category correlate at 0.25 on the latent scale, and SKUs in different categories at 0.15.

**Persistence:** the spec says to persist the shocks across weeks with AR(1), φ = 0.7, so a bad season stays bad. To keep the marginals exact, the innovation has to be scaled:

```
zm[n,t] = φ · zm[n,t−1] + √(1−φ²) · η[n,t],   zm[n,0] ~ N(0,1)
```

zc works the same way. Without the √(1−φ²) factor, the variance drifts away from 1 and the simulated demand no longer matches the forecast.

## Step 3: sampling demand from the quantiles

Each SKU-week has seven known points of its quantile function Q(u):

```
u:  0.05  0.10  0.25  0.50  0.75  0.90  0.95
Q:  q05   q10   q25   q50   q75   q90   q95
```

Sampling means evaluating Q at the drawn `u`. That needs a rule between the points and beyond the ends.

### Between q05 and q95: linear interpolation in u

Example forecast: q05=2, q10=4, q25=7, q50=10, q75=14, q90=19, q95=23. For u = 0.6, the value sits between (0.50, 10) and (0.75, 14):

```
Q(0.6) = 10 + (0.6 − 0.5)/(0.75 − 0.5) · (14 − 10) = 11.6 units
```

### Beyond q05 and q95: the tails

These hold 10% of the probability, and the lower tail is exactly where the risk sits, since underwriting uses q05 of the cash ratio. The spec only says "log-linear tail". The reading used here is an exponential tail fitted to the last two points, so Q is linear in log(1−u) above q95 and linear in log(u) below q05:

```
upper (u > 0.95):  Q(u) = q95 + β_hi · ln(0.05 / (1−u)),   β_hi = (q95 − q90) / ln 2
lower (u < 0.05):  Q(u) = max(0, q05 − β_lo · ln(0.05 / u)),   β_lo = (q10 − q05) / ln 2
```

- **Why ln 2:** the probability gap from q90 to q95 is 0.10 → 0.05, a halving, so β is the increase in units per halving of tail probability, and the tail continues that slope.
- **Example:** u = 0.98 gives 23 + 5.77 · ln(2.5) = 28.3. u = 0.02 gives 2 − 2.89 · 0.92 < 0, so it's clipped to 0.
- **Clip u:** clip `u` to [1e-4, 1 − 1e-4] so the log stays finite. 1e-4 rather than smaller because float32 cannot represent `1 − u` accurately much closer to 1; the cap sits about 9 halvings of tail probability past q95 and affects about 1 draw in 10,000.

### What this handles without extra work

- **Intermittent SKUs:** if q05 = q10 = q25 = 0, all of u ≤ 0.25 maps to 0. The flat segment produces the point mass at zero ("no sale this week").
- **Upper tail:** the exponential tail is light-to-medium. For lumpy SKUs it may understate big orders. A heavier option is linear in log-quantity, but that breaks when quantiles are 0, so the exponential version is the default, to be checked in validation.
- **Fractional units:** the samples are continuous. For cash this doesn't matter. Integer sampling only matters for very slow movers, where samples can be rounded stochastically.

### Vectorised

The seven knots are an array `[S, W, 7]`. `np.searchsorted` on the fixed u-grid finds each `u`'s segment, then one gather and one multiply-add. It's about a dozen NumPy operations for the whole `[N, S, W]` cube, well within the spec's target of under 1 s per lot.

### Markdown rescales the sample (step 6)

Demand × (p_new/p_old)^elasticity. With elasticity −1.5, a 20% markdown gives 0.8^−1.5 = 1.40×, so 40% more demand at a lower price.

## Gaps in the spec to decide before building

1. **No persistence within a SKU (the most important one).** ε is independent per week, so one SKU's 13 weekly draws only move together through the shared shocks. The forecast's real errors are mostly *level* errors: if a SKU is overforecast, it's overforecast every week. Summing independent weeks shrinks the relative spread of 13-week cumulative demand by roughly 1/√13. That narrows the cash_ratio distribution, which inflates q05 and over-advances, worst on small lots dominated by a few SKUs. **Fix:** add a persistent per-SKU component, `x = √ρm·zm + √ρc·zc + √ρs·zs[i] + √(1−ρm−ρc−ρs)·ε`, with zs AR(1) per SKU. Estimate ρs from the autocorrelation of PIT values within each series. PIT (probability integral transform) means where the actual sale fell within the forecast distribution. **Status:** implemented in `thp/src/thp/simulate/copula.py` with assumed defaults ρs = 0.20 and φs = 0.9, until they can be estimated.
2. **The tenor clock vs the 13-week horizon.** The pseudo-loan labels count the tenor *from receipt*, and lead time is about 6 weeks. If the simulator also starts at receipt, day 90 falls around week 19, past the forecast horizon. Either count the tenor from the advance (order) date, or extend the forecasts, e.g. horizon 26 or repeating the last weeks. This must match the label definition, or the validation compares different things. **Decided:** the tenor counts from disbursement; see [the tenor clock](tenor-clock.md).
3. **Missing weeks from external sources.** ADR 0002 leaves weeks the source didn't forecast empty. The simulator needs a rule: fill them from the baseline (simplest) or refuse the lot. **Decided:** refuse the lot. `forecast_knots` in `thp/src/thp/simulate/inputs.py` raises `MissingForecastError` listing the missing series-weeks.
4. **Mapping days to weeks.** Day 30 is week 4.3. Interpolate cumulative cash linearly within the week, rather than rounding to week 4 or 5, which shifts cash_ratio_30 noticeably. **Implemented** in the engine: a tenor ending mid-week gets that fraction of the week's cash.
5. **The ρ estimates need forecast history.** ρm, ρc (and ρs) come from PIT values of matured forecasts. With externally provided forecasts, that requires issues kept over time. ADR 0002's append-only `external_forecast` table already allows it. Until then the defaults apply, and the backtest coverage check (realised below q05 in 3–7% of lots) is what tells you if they're wrong.

## Proposed build order (`simulate/`)

1. `quantiles.py`: inverse CDF and tails, with property tests (the output is monotone in u; sampling with uniform u reproduces the input quantiles).
2. `copula.py`: shocks, AR(1) persistence, correlated uniforms.
3. `engine.py`: receipt, FIFO selling, markdown, cash and aggregation.

This keeps the sampling step tested on its own before any stock logic sits on top of it.
