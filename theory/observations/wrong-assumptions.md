# Wrong assumptions

Assumptions, in the spec, the code or our own reasoning, that turned out to be wrong or incomplete. Newest first. See [the observations index](README.md) for the entry format.

## Plain seasonal naive is good enough to score lots with

- **Date:** 2026-10-26
- **Source:** our reasoning, from the spec's baseline (`thp/src/thp/demand/baseline.py`, v1)
- **Assumption:** the spec's seasonal-naive baseline, last year's demand for the same week, is a usable forecast to score lots with until LightGBM exists.
- **Why it's wrong:** it cannot see a level change. A discontinued demo SKU selling 7.75 units a week was forecast at 32, last year's level, so the simulator expected the slow-movers lot of dead stock to return 1.42× its cost in 60 days, and the advance hit the 80% cap.
- **Impact:** over-advancing on exactly the stock a lender should worry about: collapsed, discontinued or declining SKUs.
- **Status:** fixed in `seasonal_naive_v2`: last year's week is scaled by the series' level change (last 8 weeks over the same weeks a year earlier), clipped to [0.1, 4] (see [tradeoffs](tradeoffs.md)). The slow-movers lot now gets 59%. Recorded as an amendment in ADR 0002.

## A 13-week forecast covers 90-day tenors

- **Date:** 2026-10-26
- **Source:** spec ("horizons 1–13"), and our own [tenor clock](../tenor-clock.md) note
- **Assumption:** 13 weeks (91 days) covers a 90-day tenor from any as_of.
- **Why it's wrong:** horizon week 1 starts after as_of's last *complete* week, up to 6 days before as_of, and the tenor runs from the day after disbursement. From an as_of other than the last or first day of a week, 90 days reach into week 14, so the lot was refused.
- **Impact:** 90-day tenors could only be scored on two weekdays out of seven.
- **Status:** default `demand.horizon_weeks` is now 14; the spec's demand-model section is updated. External sources that forecast only 13 weeks still can't score a 90-day tenor from mid-week (the lot is refused, by design).

## A buyer ordering from recent sales keeps stock healthy

- **Date:** 2026-10-26
- **Source:** our reasoning, in the demo generator (`thp/src/thp/ingest/demo.py`)
- **Assumption:** an order-up-to buyer using the mean of the last eight weeks' sales gives realistic stock levels.
- **Why it's wrong:** sales are censored by stock-outs, so a stock-out lowers the next order, which causes the next stock-out. The buyer also ignored seasonality, ordering for today's rate with 6–8 weeks of lead time. 29% of series-weeks ended out of stock, against roughly 5–10% for real merchants.
- **Impact:** demo data would have shown mostly stock-out behaviour, not the excess stock the engine is about.
- **Status:** fixed: the buyer estimates the rate from in-stock days only and scales it by the known seasonal curve over the coming lead time. Active SKUs are now out of stock 10.5% of the time. The same censoring trap applies to real merchants' forecasts: features already carry an availability flag in the spec, and the demand model must down-weight censored weeks.

## The spec's purchase_order table is enough to know where stock lands

- **Date:** 2026-10-26
- **Source:** spec, canonical data model
- **Assumption:** `purchase_order` (keyed by tenant, po_id, sku, line_no) carries what the engine needs.
- **Why it's wrong:** it has no location, so open POs can't be attributed to a SKU × location's stock ahead.
- **Impact:** lot assembly could not compute stock ahead per location.
- **Status:** `location_id` added to `PURCHASE_ORDER` (`thp/src/thp/schema/canonical.py`), keyed by (tenant, po_id, line_no); the spec's table is updated.

## "More stock ahead never raises the cash ratio" fails with markdowns

- **Date:** 2026-10-25
- **Source:** spec, testing strategy (simulator property tests)
- **Assumption:** more existing stock ahead of a lot can only delay its sales, so its cash ratio never rises.
- **Why it's wrong:** more stock ahead raises cover, which can trigger an earlier markdown. With elasticity −1.5, a 20% price cut raises demand 1.40× and revenue per week by 12%, so the lot can sell faster than it would with less stock ahead and no markdown. Hypothesis found a case (stock ahead 52 vs 76): 1 path in 200, by 0.7% of lot cost.
- **Impact:** the property holds for the selling mechanics alone. With markdowns it also exposes that elastic demand (|ε| > 1) makes markdowns raise revenue, which flatters lots heading for markdown.
- **Status:** the property test runs with markdowns off. Open question: whether the default elasticity, or the revenue effect of markdowns, should be capped for underwriting.

## The most likely receipt week is not the median week

- **Date:** 2026-10-25
- **Source:** our reasoning, in `test_receipt_probabilities`
- **Assumption:** with a lognormal lead time whose median falls in week 7, week 7 is also the most likely week.
- **Why it's wrong:** a lognormal is right-skewed, so its mode lies below its median. With σ = 0.3 the most likely selling week is 6, while the median is still 7.
- **Impact:** test expectation only; the receipt distribution was right.
- **Status:** the test now checks the median.

## The tenor clock was inconsistent between labels and simulator

- **Date:** 2026-10-18
- **Source:** spec, pseudo-loan generator and cash-conversion simulator
- **Assumption:** the pseudo-loan labels and the simulator measure the same tenor.
- **Why it's wrong:** labels counted the tenor from receipt, while the simulator starts at the order date with a 13-week forecast. From receipt, day 90 falls around week 19, past the forecast. Validation would have compared different quantities.
- **Impact:** the calibration check, the acceptance test for the whole simulator, would have been meaningless.
- **Status:** decided 2026-10-25: the tenor counts from disbursement. See [the tenor clock](../tenor-clock.md).

## Independent weekly noise understates a SKU's cumulative uncertainty

- **Date:** 2026-10-18
- **Source:** spec, cash-conversion simulator (copula)
- **Assumption:** shared merchant and category shocks plus independent noise per SKU-week capture the dependence the simulator needs.
- **Why it's wrong:** real forecast errors are mostly *level* errors: an overforecast SKU is overforecast every week. With independent weekly noise, a SKU's 13 weeks partly cancel, shrinking the spread of cumulative demand by up to about 1/√13. That narrows the cash-ratio distribution and inflates q05, so the engine over-advances, worst on small lots dominated by a few SKUs.
- **Impact:** systematic over-lending.
- **Status:** fixed with a persistent per-SKU component zs in `thp/src/thp/simulate/copula.py`. Its parameters (ρs = 0.20, φs = 0.9) are assumed until estimated (see [tradeoffs](tradeoffs.md)).

## AR(1) shocks without innovation scaling drift in variance

- **Date:** 2026-10-18
- **Source:** spec, cash-conversion simulator ("AR(1) with φ = 0.7")
- **Assumption:** the spec's AR(1) persistence can be applied as written.
- **Why it's wrong:** the spec gives no formula. Persisting a shock as z[t] = φ·z[t−1] + η[t] lets the variance drift away from 1, so the uniforms stop being uniform and the simulated demand no longer matches the forecast.
- **Impact:** would silently distort every SKU's demand distribution.
- **Status:** implemented as z[t] = φ·z[t−1] + √(1 − φ²)·η[t] (`ar1` in `copula.py`); a test checks unit variance in every week.
