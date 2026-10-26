# What the engine solves, and how

The problem the inventory risk engine addresses, the idea behind its answer, and how each part of the pipeline contributes to it. It is the entry point to the theory notes; the build details are in the [technical spec](../docs/specs/inventory-risk-engine.md).

## The problem

### For the merchant

A merchant pays for stock before it sells. Every purchase order ties up cash for the lead time plus however long the goods take to sell, and stock that sells slowly ties it up for months. Growing merchants feel this most: the better they sell, the bigger the next order, and the more cash sits in the warehouse. They need financing for purchase orders and stock, and they need to know where their cash is trapped.

### For the lender

Lenders are reluctant to lend against inventory, because its value is uncertain. A pallet of stock is worth its cost only if it sells, at full price, in time. Whether it does depends on demand for each individual SKU, which is noisy and often intermittent, and on things that happen after the loan is made: late deliveries, markdowns, a bad season.

The common workaround is to ignore the inventory and lend against revenue, e.g. an advance of 1.0 × trailing monthly revenue. That rule doesn't look at *what* is being bought. A merchant who overbuys a slow-moving range gets the same advance as one who buys exactly what will sell, so the rule over-lends exactly when the stock is riskiest.

### The question to answer

> If I advance L against this purchase order or stock position, how sure can I be that it turns into at least L × (1 + fee) of cash by the due date?

Answering it takes four things the revenue rule doesn't have:

1. **SKU-level demand as a distribution**, not a single number, because the risk is in the low outcomes.
2. **The joint behaviour of all SKUs in the lot**, because a bad season hits them together, and that doesn't average out.
3. **The mechanics between demand and cash:** stock already on hand sells first, goods arrive late, unsold stock gets marked down, customers return goods and pay late.
4. **Evidence that the answer is calibrated**, before any loan has ever been made.

## The idea

**Model cash conversion as a probability distribution, and advance against its low end.**

For each lot, the engine simulates 2,000 plausible futures, computes in each how much cash the lot produces by day 30, 60 and 90, and reads off a low quantile. If cash covers the loan plus fee in 95% of futures, the loan is safe at 95% confidence:

```
L* = q05(cash_ratio at the tenor) × lot cost / (1 + fee)
```

Everything else in the engine exists to make that distribution right, and to prove that it is.

## How the pipeline solves it

```
merchant data ─▶ canonical schema ─▶ features ─▶ demand forecast ─▶ simulator ─▶ underwriting ─▶ advance + reasons
  (CSV, ERP)      (one model,         (weekly,      (quantiles per      (2,000          (L*, caps,
                   as-of access)       per SKU)      SKU-week)           futures)        rules)
                                                         │
                                                         └──────────────▶ trapped-cash report (merchant)

history ─▶ pseudo-loans ─▶ backtest: simulated vs realised cash_ratio ─▶ calibration evidence (lender)
```

### 1. One canonical schema, read as of a date

Merchants' data arrives from CSV exports, ERPs and marketplaces, all in different shapes. Every source is mapped into one set of tables (sales, prices, stock, purchase orders, SKUs), so nothing downstream depends on where the data came from. Ingestion also checks the data: inconsistent stock records lower a **data-trust score**, which becomes an underwriting input, because bad data is itself a risk.

Every read goes through an **as-of store** that only returns data dated on or before a given day ([ADR 0001](../docs/adr/0001-as-of-access-seeding-and-parquet.md)). This is what makes the evidence in step 6 trustworthy: a backtest at a past date can't accidentally see what happened next.

### 2. Demand features

Weekly sales per SKU × location are turned into the inputs a forecast needs: recent sales, rolling statistics, intermittency class, trend, seasonality, price and lifecycle (`thp/src/thp/features/demand.py`).

### 3. Demand forecasts as quantiles

For each SKU × location and each of the next 14 weeks, the forecast gives seven quantiles, q05 to q95: a full distribution of demand, not a point. The engine is agnostic about where it comes from ([ADR 0002](../docs/adr/0002-baseline-and-external-forecasts.md)):

- the **level-adjusted seasonal-naive baseline** (built), the bar every model must beat;
- a **LightGBM quantile model** (planned), trained across all SKUs of a tenant;
- **external forecasts** from the merchant's own planning tool (built), with missing quantiles filled from the baseline's spread.

This solves requirement 1: the risk lives in the low quantiles, so they are modelled explicitly.

### 4. The cash-conversion simulator

The simulator turns per-SKU forecasts into 2,000 futures for the whole lot ([overview](cash-conversion-simulator.md)):

- **Correlated demand** ([the copula](copula.md), [why shocks](why-shocks.md), [the draw logic](simulator-draws.md)). Shared merchant and category shocks make SKUs have bad weeks together, and a per-SKU level makes a SKU that is overforecast stay overforecast. Each SKU-week's demand still follows its forecast exactly. This solves requirement 2.
- **Mechanics.** Each future runs week by week: the goods arrive at a sampled week, older stock sells first, sales are capped by stock, unsold stock is marked down (and demand responds), returns are deducted and B2B cash arrives late. This solves requirement 3.
- **Output.** Cumulative cash at days 30/60/90 ÷ lot cost gives 2,000 cash ratios per tenor. Their quantiles, P(cash_ratio_90 < 1) and expected unsold value are the lot's score.

The tenor's T days count from disbursement ([the tenor clock](tenor-clock.md)), and a lot whose forecast lacks a week the tenor needs is refused rather than filled.

### 5. Underwriting: from a distribution to a decision

A deterministic rule turns the score into an advance:

- **Advance:** L* = q05(C_T) × V / (1 + f), capped at a maximum advance rate (default 80% of lot cost).
- **Liquidation floor:** what the unsold stock would recover in a default (at a haircut). It adds to the advance only once the haircuts are validated.
- **Rules:** decline on poor data quality or a high probability of shortfall, reduce on excessive weeks of cover, refer to manual review when cold-start SKUs dominate.
- **Reasons:** every decision carries machine-readable reason codes, so it can be explained to the merchant and audited by the lender.
- **Replay:** every score stores the model version, config hash and seed, so any number can be reproduced exactly.

### 6. Evidence before the first loan: pseudo-loans and backtests

A new lender has no loan history, so the engine creates one. Every historical purchase order is a loan that *could* have been made: score it as of its order date, using only data available then, and compare the predicted cash ratio with what its stock actually produced. For the M5 sandbox, which has no purchase orders, synthetic orders are generated with ordering policies from disciplined to overbuying.

The acceptance test is calibration: realised cash should fall below the predicted q05 in 3–7% of lots, overall and within each merchant profile and quarter. If it falls below more often, the engine is over-lending, and the backtest shows it before real money does. This solves requirement 4.

The same backtest runs the revenue rule on the same lots, so the engine is always compared side by side with the status quo: capital deployed and loss rate for both.

### 7. The trapped-cash report: the same forecasts, for the merchant

The forecasts also tell the merchant where cash is stuck. Each SKU × location is classed by weeks of cover (dead, slow, excess, healthy, at risk), with the cash tied up in the excess, recommended actions ranked by cash freed, reorder proposals, and a 13-week cash-from-inventory projection with a P10–P90 band. It also shows what the underwriting would advance on current stock, which links the merchant's view to the lender's.

## A worked example

The spec's API example: a purchase order of two lines, total cost V = €10,890, tenor 90 days, fee 3%.

1. The forecasts for both SKUs, the existing stock and the expected receipt go into the simulator, which returns d90 cash-ratio quantiles q05 = 0.75, q50 = 1.18, q95 = 1.52, and P(cash_ratio_90 < 1) = 17%.
2. The advance at 95% confidence: L* = 0.75 × 10,890 ÷ 1.03 = **€7,930**. That's below the 80% cap (€8,712).
3. 46% of the lot's value would sit above 26 weeks of cover, so the HIGH_COVER rule reduces the advance by 25%: **€5,947**, an advance rate of 55%, decision `reduce`.
4. The 17% shortfall probability is under the 20% decline threshold, so the lot isn't declined.

The median future returns 118% of cost by day 90, so a point forecast would have suggested lending nearly the full amount. The engine lends 55%, because in the bad futures much of this stock is still unsold by day 90, and it says why.

## What it does not solve

- **Willingness to repay.** The engine estimates whether the stock *produces* the cash, not whether the merchant hands it over. Merchant credit, fraud and diversion of funds need separate checks.
- **Unprecedented shocks.** Correlations and spreads are estimated from history. A shock outside that history (a pandemic, a supplier collapse) is not in the distribution.
- **Bad data.** Calibration holds only for data that reflects reality. The data-trust score flags inconsistencies but can't detect consistent misreporting.
- **Uncalibrated forecasts.** If the demand quantiles are too narrow, the cash-ratio distribution is too, whatever the simulator does. That's why forecasts are backtested on their own (coverage of P10–P90 and below P05) before lots are.

## Build status (2026-10-26)

| Part | Status |
|---|---|
| Canonical schema (incl. stock snapshots, purchase orders), as-of store, config, seeding | built |
| M5 ingest, demand features | built (full-dataset timing not yet verified) |
| Synthetic demo tenant and demo lots (`thp make-demo`) | built |
| Seasonal-naive baseline (level-adjusted), external forecasts | built |
| LightGBM model, forecast evaluation harness | planned |
| Simulator: sampling, copula, engine, lot assembly; `thp score-lot` | built (B2B receivable delays not yet) |
| Underwriting | advance formula and cap built; liquidation floor and decision rules planned |
| Pseudo-loans, trapped-cash report, API | planned |
