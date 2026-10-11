# Inventory Risk Engine Technical Spec

Oct 11, 2026 · @Oliver kremser

A Python service that turns merchant inventory and sales data into SKU-level demand distributions, simulates how much cash a purchase order or stock position converts to by day 30/60/90, and derives a safe advance amount and a trapped-cash report from that distribution. Companion to the Inventory Finance Build Plan; phase numbers below refer to it.

## Scope and design principles

Version 1 covers Phases 1–3: a batch scoring engine, an HTTP API and generated reports for up to about 50 tenants with up to 50,000 SKUs each. Live lending integration (Phase 4) is out of scope but the interfaces anticipate it.

**In scope:** ingestion from CSV/Excel and one ERP connector, canonical data model, features, quantile demand model, pseudo-loan generation, Monte Carlo cash-conversion simulation, underwriting rule, trapped-cash report, REST API, nightly jobs, model registry and calibration monitoring.

**Out of scope for v1:** customer-facing web app (Streamlit internal UI only), payments or lending flows, real-time scoring under 1 second, deep learning models.

**Principles:**

1. **One canonical schema.** Every source maps into it; nothing downstream knows where data came from.
2. **Distributions, not points.** Every forecast and score is a set of quantiles or samples. Point values are derived at the edge.
3. **Reproducible by construction.** Every score stores model version, feature snapshot date, config hash and random seed, so a lender can replay any number.
4. **Time-honest.** Every computation takes an explicit `as_of` date and may only read data with `event_date <= as_of`. This is enforced in the data access layer, not by convention.
5. **Pure core, thin shell.** Modeling and simulation are pure functions on DataFrames/arrays, testable without a database. I/O lives in adapters.
6. **Boring infrastructure.** Postgres, Parquet, Docker, cron-style jobs. Add complexity only when a measured limit is hit.

## System overview

&#91;embedded content: system architecture · sources to interfaces\]

Data enters through connectors, is validated into one canonical schema, and every forecast, lot score and report is computed by the same engine from that schema; the simulator is the core that the underwriting and the backtests both depend on.

## Canonical data model

Eight core tables, all keyed by `tenant_id`; facts are append-only with `ingested_at`, so any past state can be reconstructed for an `as_of` date.

**Reference tables**

| Table | Key | Columns |
| --- | --- | --- |
| `tenant` | `tenant_id` (uuid) | name, currency (ISO 4217), timezone, week\_start, fiscal\_year\_start, created\_at |
| `location` | `tenant_id, location_id` | name, type (warehouse, store, online), country |
| `sku` | `tenant_id, sku_id` | name, category\_l1..l3, brand, supplier\_id, uom, pack\_size, lifecycle\_status (new, active, phase\_out, discontinued), launch\_date, is\_perishable, shelf\_life\_days |
| `supplier` | `tenant_id, supplier_id` | name, country, lead\_time\_days\_p50, lead\_time\_days\_p90, moq, payment\_terms\_days |

**Fact tables**

| Table | Grain | Columns |
| --- | --- | --- |
| `sales` | tenant, sku, location, date, channel | qty, net\_revenue, discount, returned\_qty, customer\_type (b2b, b2c) |
| `price` | tenant, sku, valid\_from | list\_price, unit\_cost (landed), currency |
| `stock_snapshot` | tenant, sku, location, snapshot\_date | on\_hand\_qty, on\_order\_qty, reserved\_qty, unit\_cost, oldest\_receipt\_date |
| `purchase_order` | tenant, po\_id, sku, line\_no | supplier\_id, order\_date, expected\_receipt\_date, actual\_receipt\_date, qty, unit\_cost, status |

**Optional for Phase 3+:** `receivable` (invoice\_date, due\_date, paid\_date, amount, customer\_id) to model B2B payment delay, and `markdown_event` (sku, date, discount\_pct) to estimate elasticity.

**Derived tables (written by the engine):**

- `feature_snapshot` (tenant, sku, location, as\_of, feature vector as Parquet path)
- `forecast` (tenant, sku, location, as\_of, horizon\_week, q05…q95, model\_version)
- `lot` and `lot_line` (a scored PO or stock position: lot\_id, as\_of, tenor\_days, lines with sku, qty, unit\_cost)
- `lot_score` (lot\_id, model\_version, seed, cash\_ratio quantiles at d30/d60/d90, advance\_amount, liquidation\_value, config\_hash)
- `trapped_cash_item` (tenant, sku, as\_of, health\_class, excess\_qty, cash\_tied, suggested\_action)

**Conventions:** money as `NUMERIC(18,4)` plus currency; dates as `DATE` in tenant timezone; weekly aggregation is ISO week unless the tenant sets otherwise; quantities are non-negative, returns are separate columns.

## Ingestion

Every source implements one connector interface and writes to a raw zone first; a separate validation step promotes data into the canonical tables or rejects it with a report.

**Connector interface:**

```python
class Connector(Protocol):
    source_name: str

    def discover(self) -> list[EntityInfo]: ...          # which entities and date ranges exist
    def extract(self, entity: str, since: datetime | None) -> Iterator[pa.RecordBatch]: ...
    def mapping(self) -> dict[str, FieldMap]: ...        # source field -> canonical column + transform
```

**Implementations, in order:** `CsvExcelConnector` (Phase 1–2, with a column-mapping YAML per tenant), `M5Connector` (sandbox), then one ERP connector in Phase 3 (JTL-Wawi, Business Central or SAP Business One, decided by interview results).

**Pipeline:** extract → raw Parquet in `raw/{tenant}/{source}/{entity}/{ingest_ts}/` → map to canonical → validate → upsert into Postgres facts → write an `ingest_run` record (rows read, accepted, rejected, checks failed).

**Validation checks (Pandera schemas plus custom rules):**

| Check | Rule | On failure |
| --- | --- | --- |
| Schema | Required columns, types, non-null keys | Reject batch |
| Referential | Every sales and stock SKU exists in `sku` | Quarantine rows |
| Ranges | qty ≥ 0, unit\_cost > 0, price ≥ unit\_cost × 0.2 | Flag, keep |
| Continuity | No gap over 14 days in sales history for an active SKU | Flag |
| Stock reconciliation | stock(t) ≈ stock(t−1) + receipts − sales ± 5% | Flag; feeds a data-trust score |
| Duplicates | Same grain twice in one batch | Keep last, log |
| Freshness | Latest sales date within 3 days of ingest | Alert |

**Data-trust score:** a 0–1 score per tenant from the share of rows passing reconciliation and continuity. It is stored with every score and later becomes an underwriting input, because inconsistent stock data is itself a risk signal.

## Feature pipeline

Features are computed weekly per SKU × location for a given `as_of`, using only data dated on or before it, and stored as Parquet snapshots so training and scoring read identical inputs.

**Demand features**

- Lags: weekly sales at t−1 … t−4, t−8, t−13, t−26, t−52
- Rolling stats over 4/13/26/52 weeks: mean, std, median, max, share of zero weeks
- Intermittency: ADI and CV² over 52 weeks, Syntetos-Boylan class (smooth, erratic, intermittent, lumpy)
- Demand cluster id (k-means on normalized 52-week profiles, fitted per tenant)
- Trend: slope of a 13-week linear fit; ratio of 13-week to 52-week mean
- Seasonality: week of year (cyclical sin/cos), holiday and event flags per country
- Price: current price relative to its 13-week mean, discount depth, days since last price change
- Lifecycle: weeks since launch, lifecycle\_status, is\_new (under 13 weeks history)
- Availability: share of days with on\_hand = 0 in last 4 weeks (censored demand flag)
- Hierarchy aggregates: category-level and tenant-level rolling means, for pooling

**Lot and merchant features** (used by the underwriting layer, not the demand model)

- Weeks of cover = lot qty ÷ P50 weekly forecast
- Lot concentration: share of lot value in top 5 SKUs; Herfindahl index over SKUs
- Season alignment: weeks between expected receipt and the SKU's seasonal peak
- Merchant turns (COGS ÷ average inventory, trailing 12 months) and their 6-month trend
- Aged-stock share: value with oldest\_receipt\_date over 180 days ÷ total stock value
- Order discipline: median over historical POs of (qty ordered ÷ qty sold within lead time + 8 weeks)
- Data-trust score from ingestion

**Leakage rules (enforced in code):**

1. The feature builder receives `as_of` and a data access object that filters every query to `event_date <= as_of` and `ingested_at <= as_of + grace`.
2. Rolling windows end at `as_of`, never centered.
3. Clusters, scalers and encoders are fitted only on the training window and versioned with the model.
4. A unit test shifts all future data by +1,000 units and asserts features are unchanged.

## Demand model

One global LightGBM model per quantile, trained across all SKUs of a tenant (later across tenants), predicting weekly demand for horizons 1–13 weeks.

**Formulation:**

- Target: weekly units sold for SKU × location at horizon h, scaled by the SKU's 52-week mean (inverse-transformed at output). Weeks with stock-outs over 50% of days are down-weighted to 0.3 (censored demand).
- Direct multi-horizon: horizon h is a feature; one row per (SKU, as\_of, h). Avoids recursive error build-up.
- Quantiles: 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, each a separate `objective='quantile'` model. Enforce monotonicity after prediction by sorting quantiles per row.
- Cold start: SKUs with under 13 weeks of history get predictions from category-level features only; flag `is_cold_start` in the output.

**Starting hyperparameters:** `num_leaves=63`, `learning_rate=0.05`, `min_data_in_leaf=100`, `feature_fraction=0.8`, `bagging_fraction=0.8`, `lambda_l2=1.0`, early stopping on pinball loss with 100 rounds patience. Tune with Optuna on the validation fold only.

**Validation scheme:** rolling-origin, 4 folds, each training up to cutoff c and testing the next 13 weeks; plus a group k-fold by tenant or by store (M5) for the out-of-entity test.

**Metrics, logged to MLflow per fold and per demand class:**

| Metric | Purpose | Acceptance (sandbox) |
| --- | --- | --- |
| Pinball loss per quantile | Training objective | Beat seasonal-naive quantiles by 15%+ |
| CRPS (from quantiles) | Overall distribution quality | Beat baseline by 15%+ |
| Coverage of P10–P90 | Calibration | 76–84% |
| Coverage below P05 | Tail calibration | 3–7% |
| WMAPE of P50 | Comparability with point models | Report only |

**Baselines to always report:** seasonal naive with empirical residual quantiles; Croston/TSB for intermittent SKUs (via `statsforecast`). If LightGBM does not beat them on intermittent items, use the baseline for that class.

**Output:** a `forecast` row per SKU × location × horizon with q05…q95. The simulator samples from these by inverse-CDF interpolation between quantiles, with a log-linear tail beyond q05 and q95.

## Pseudo-loan generator

Creates labeled training and evaluation lots before any real loan exists: every lot is a PO (real or synthetic) whose realized cash conversion can be computed from later sales.

**Two sources:**

1. **Historical POs** (merchant data): every real `purchase_order` with receipt date at least tenor + 13 weeks before the latest data becomes a lot.
2. **Synthetic POs** (M5 sandbox, which has no POs): simulate ordering behavior per SKU with an order-up-to policy, `order_qty = max(0, S − inventory_position)`, where `S = μ_LT+R × k` and the bias factor k is drawn from a distribution per merchant profile (disciplined k \~ U(0.9, 1.2), average k \~ U(1.0, 1.6), overbuyer k \~ U(1.4, 2.5)). Review period R = 4 weeks, lead time from a lognormal around 6 weeks. A lot groups the SKU lines ordered in one store in one week.

**Realized label for a lot with tenor T, received at r:**

- Allocate sales FIFO: units of the lot are sold only after earlier stock of the same SKU at that location is sold.
- `cash_realized(T) = Σ sold_units_from_lot × net_price × (1 − return_rate)`, shifted by the receivable delay for B2B sales where known.
- `cash_ratio(T) = cash_realized(T) ÷ lot_cost`, computed for T = 30, 60, 90 days.
- Also store `unsold_value(T)` at cost for the liquidation analysis.

**Output table** `pseudo_lot`: lot\_id, source (historical, synthetic), profile, as\_of (= order date), receipt\_date, lot\_cost, n\_lines, cash\_ratio\_30/60/90, unsold\_value\_90. Target size for the sandbox: 10,000+ lots spanning at least two years and all M5 stores.

## Cash-conversion simulator

A vectorized Monte Carlo (NumPy, N = 2,000 paths by default) that pushes demand samples through stock, pricing and payment mechanics to produce a distribution of `cash_ratio` per lot at day 30, 60 and 90.

**Inputs:** lot lines (sku, qty, unit\_cost), current stock and open POs per SKU, forecast quantiles for horizons 1–13, price and markdown policy, return rate per category, receivable delay distribution, receipt date distribution, seed.

**Algorithm per lot (all paths at once, arrays shaped \[N, SKUs, weeks\]):**

1. **Draw shared shocks.** One merchant-season factor per path, z\_m \~ N(0, 1), and one category factor per path and category, z\_c \~ N(0, 1).
2. **Correlate.** For each SKU, latent u = Φ(√ρ\_m · z\_m + √ρ\_c · z\_c + √(1 − ρ\_m − ρ\_c) · ε), with ε \~ N(0, 1) independent per SKU and week (Gaussian copula).
3. **Sample demand.** Weekly demand = inverse CDF of the SKU's forecast quantiles evaluated at u.
4. **Draw receipt week** from the lot's lead-time distribution; lot stock is available only from then.
5. **Step weekly:** sell from existing stock first (FIFO), then from lot stock, capped by available units.
6. **Apply markdown policy:** if lot stock age exceeds a threshold (default 12 weeks) or weeks of cover exceed 26, price drops by the configured step (default 20%, max 50%), and demand is scaled by an elasticity factor (default −1.5 or estimated per category).
7. **Convert to cash:** revenue × (1 − return rate), shifted by the sampled receivable delay for the B2B share.
8. **Aggregate:** cumulative cash at days 30/60/90 ÷ lot cost → cash\_ratio samples; also unsold units at day 90.

The correlated latent in step 2, written out:

```latex
u_{i,t} = \Phi\left(\sqrt{\rho_m}\, z_m + \sqrt{\rho_c}\, z_{c(i)} + \sqrt{1-\rho_m-\rho_c}\, \varepsilon_{i,t}\right)
```

**Estimating ρ\_m and ρ\_c:** from historical forecast residuals, as the average pairwise correlation of PIT values within a merchant-week and within a category-week. Defaults before estimation: ρ\_m = 0.15, ρ\_c = 0.10. Persist the shock across weeks of one path (AR(1) with φ = 0.7) so a bad season stays bad.

**Outputs per lot:** cash\_ratio quantiles (q01, q05, q10, q25, q50, q75, q90) at each horizon, P(cash\_ratio\_90 < 1), expected unsold value at day 90, and the seed and config hash.

**Performance target:** a 200-line lot with N = 2,000 and 13 weeks in under 1 second on one core; a full tenant stock position (50,000 SKUs) in under 5 minutes, chunked by SKU blocks. Use float32 and `numpy.random.Generator(PCG64)`; move to Numba only if profiling says so.

**Validation:** run the simulator on historical pseudo-lots with `as_of` = order date and compare realized cash\_ratio to the predicted quantiles. Accept when realized falls below q05 in 3–7% of lots overall and within each merchant profile and quarter.

## Underwriting engine

A deterministic rule on top of the simulator output that turns a cash-conversion distribution into a maximum advance, a fee-adjusted repayment check and a decision with reasons.

**Advance amount.** The largest loan L such that, with confidence level α (default 0.95), cash collected by the tenor covers principal plus fee:

```latex
L^{*} = \max\left\{ L : P\left(C_T \cdot V \ge L\,(1+f)\right) \ge \alpha \right\} = \frac{q_{1-\alpha}(C_T)\cdot V}{1+f}
```

Here C\_T is the cash ratio at tenor T, V the lot cost and f the flat fee. Also cap L at a maximum advance rate (default 80% of V).

**Liquidation floor.** Recovery if the merchant fails at day 30: unsold units × unit\_cost × haircut, with haircut by category and age (default 40% of cost, 20% for perishable or seasonal past peak). Compute `liquidation_value` from the simulator's q10 unsold units. Final advance = min(cap × V, L\* + w × liquidation\_value), with w = 0 by default (cash-only) and up to 0.5 once haircuts are validated on real recoveries.

**Decision rules (config-driven, YAML):**

| Rule | Default | Effect |
| --- | --- | --- |
| Data-trust score below 0.7 | Required ≥ 0.7 | Decline: insufficient data quality |
| Weeks of cover above 26 for over 40% of lot value | — | Reduce advance by 25% |
| Cold-start SKUs over 30% of lot value | — | Refer to manual review |
| P(cash\_ratio\_90 < 1) above 20% | — | Decline |
| Lot value below €5,000 | — | Decline: below minimum |

**Output:** `decision` (approve, reduce, refer, decline), `advance_amount`, `advance_rate`, `reasons[]` (machine codes plus text), the cash\_ratio quantiles, `liquidation_value`, and the inputs' versions. Reason codes make every decision explainable to a merchant and auditable by a lender.

**Benchmark mode:** the same engine computes a revenue-based advance (default 1.0 × trailing monthly revenue) on the same lots, so every backtest reports capital deployed and loss rate for both rules side by side.

## Trapped-cash report

Classifies every SKU × location by how much of its stock value is excess, using the same forecasts, and ranks actions by cash freed.

**Health classes** (cover = on\_hand ÷ P50 weekly demand; target cover = lead time + review period + safety weeks):

| Class | Rule | Default action |
| --- | --- | --- |
| Dead | No sales in 26 weeks, or P90 demand over 26 weeks < 5% of stock | Liquidate or return to supplier |
| Slow | Cover over 26 weeks | Markdown; stop reordering |
| Excess | Cover between target × 1.5 and 26 weeks | Pause reorders until within target |
| Healthy | Cover between target × 0.7 and target × 1.5 | None |
| At risk | Cover below target × 0.7 | Reorder; flag lost sales risk |

**Cash metrics per item:** `excess_qty = on_hand − target_cover × P50`, `cash_tied = excess_qty × unit_cost`, and `cash_freeable_90d` = expected revenue from the recommended markdown over 90 days, from a single-SKU simulator run.

**Report sections (Jinja2 → HTML → PDF):** summary waterfall of stock value by class; top 50 actions by cash freed; 8-week reorder proposals with cash impact; 13-week cash-from-inventory projection with P10/P50/P90 band; financing readiness (what the underwriting engine would advance on current stock); data quality appendix with the data-trust score and failed checks.

**Reorder proposal logic:** order-up-to with S = P(service level) quantile of demand over lead time + review period, default service level 0.95 for A items, 0.90 for B, 0.80 for C (ABC by revenue). Proposals respect MOQ and pack size.

## API specification

FastAPI, JSON over HTTPS, versioned under `/v1`, authenticated with per-tenant API keys (Phase 2) and OAuth2 client credentials for lender partners (Phase 4). Long jobs return 202 with a job id.

| Method | Path | Purpose | Sync |
| --- | --- | --- | --- |
| POST | `/v1/tenants/{tenant_id}/uploads` | Upload CSV/Excel with mapping id | 202 + ingest job |
| GET | `/v1/jobs/{job_id}` | Job status, counts, errors | Yes |
| GET | `/v1/tenants/{tenant_id}/data-quality` | Latest checks and data-trust score | Yes |
| POST | `/v1/tenants/{tenant_id}/forecasts:run` | Recompute forecasts for an as\_of | 202 |
| GET | `/v1/tenants/{tenant_id}/forecasts` | Query quantile forecasts by sku, location, horizon | Yes |
| POST | `/v1/tenants/{tenant_id}/lots:score` | Score a PO or stock position | Yes up to 500 lines, else 202 |
| GET | `/v1/lots/{lot_id}/score` | Retrieve stored score and decision | Yes |
| POST | `/v1/tenants/{tenant_id}/reports/trapped-cash` | Generate report | 202 → PDF and JSON |
| GET | `/v1/health`, `/v1/version` | Liveness, model and code versions | Yes |

**Request: score a lot**

```json
{
  "as_of": "2026-11-02",
  "tenor_days": 90,
  "fee_rate": 0.03,
  "expected_receipt_date": "2026-12-07",
  "lines": [
    {"sku_id": "A-1042", "location_id": "WH1", "qty": 1200, "unit_cost": 4.35},
    {"sku_id": "B-0977", "location_id": "WH1", "qty": 300, "unit_cost": 18.90}
  ],
  "options": {"n_paths": 2000, "seed": 42, "confidence": 0.95}
}
```

**Response**

```json
{
  "lot_id": "lot_01JB7...",
  "decision": "reduce",
  "advance_amount": 5947.00,
  "advance_rate": 0.55,
  "currency": "EUR",
  "cash_ratio": {
    "d30": {"q05": 0.18, "q50": 0.41, "q95": 0.66},
    "d60": {"q05": 0.49, "q50": 0.83, "q95": 1.12},
    "d90": {"q05": 0.75, "q50": 1.18, "q95": 1.52}
  },
  "p_cash_ratio_90_below_1": 0.17,
  "liquidation_value": 3120.00,
  "reasons": [{"code": "HIGH_COVER", "text": "46% of lot value exceeds 26 weeks of cover"}],
  "versions": {"model": "demand-q-2026.11.01", "simulator": "0.4.2", "config_hash": "9f1c...", "seed": 42}
}
```

(Values illustrative: lot cost €10,890, L\* = 0.75 × 10,890 ÷ 1.03 = €7,930, reduced 25% by the HIGH\_COVER rule.) Errors follow RFC 9457 problem details. All endpoints are idempotent on an optional `Idempotency-Key` header. Generate the OpenAPI schema from Pydantic models and keep it in the repo.

## Storage, multi-tenancy and security

Postgres holds canonical and result tables with row-level security per tenant; Parquet in S3-compatible object storage holds raw extracts, feature snapshots and simulation samples. Everything runs in an EU region.

**Storage layout:**

| Store | Contents | Retention |
| --- | --- | --- |
| Postgres 16 | Reference and fact tables, forecasts, lots, scores, jobs, audit log | Facts indefinitely while customer is active; deleted 30 days after contract end |
| Object storage (S3 API) | `raw/`, `features/`, `sim_samples/`, `reports/`, MLflow artifacts | Raw 24 months; features 36 months; samples 12 months |
| MLflow (Postgres backend) | Runs, metrics, model registry | Indefinitely |

**Tenancy:** single database, `tenant_id` on every row, Postgres row-level security policies keyed on a session variable set by the API per request. Object storage paths prefixed by tenant. A cross-tenant training set (Phase 3+) is built only from tenants whose contract permits anonymized pooling, recorded in a `consent` table.

**Security baseline:**

- TLS everywhere; encryption at rest for database and buckets
- API keys stored hashed (argon2); scoped per tenant; rotation endpoint
- Secrets in environment variables from the host's secret manager, never in the repo
- Audit log of every data access, score request and config change (who, what, when)
- Nightly encrypted backups with a monthly restore test
- Least-privilege service accounts: the job runner cannot read API keys; the API cannot delete raw data
- GDPR: a data processing agreement (AVV) per customer; data is business data, but avoid ingesting personal customer data (strip names and emails from receivables at the connector)

**Hosting (start):** one Hetzner or AWS Frankfurt VM with Docker Compose (API, worker, Postgres, MLflow, MinIO if self-hosting object storage), managed Postgres once there are paying customers.

## Orchestration, MLOps and monitoring

Scheduled jobs run through Prefect (or plain cron plus a job table at first); every job is idempotent for a given tenant and `as_of`, so reruns are safe.

**Jobs:**

| Job | Schedule | Steps |
| --- | --- | --- |
| `sync_tenant` | Nightly 02:00 per tenant | Extract → validate → upsert → data-trust score |
| `build_features` | After sync | Feature snapshot for as\_of = last complete week |
| `forecast` | After features | Load production model, predict q05…q95, write `forecast` |
| `score_open_pos` | After forecast | Score open POs and full stock position |
| `retrain` | Monthly, 1st Sunday | Train on rolling window, evaluate, register candidate |
| `calibration_monitor` | Weekly | Compare matured forecasts and lots with realized outcomes |
| `report` | On demand | Trapped-cash report |

**Model lifecycle:**

1. `retrain` logs a run to MLflow with params, metrics per fold and demand class, feature importance and the data window.
2. The candidate is registered as `demand-q` version N in stage `candidate`.
3. Promotion gate (automated): CRPS no worse than production by more than 2%, P05 coverage within 3–7%, no demand class degrading over 10%. Pass → `production`; previous version kept for rollback.
4. Every forecast and score row stores the model version that produced it.

**Monitoring:**

- **Calibration drift:** rolling 8-week share of realized demand below q05 and above q95, per tenant and demand class; alert outside 2–10%.
- **Lot calibration:** as lots mature (tenor elapsed), realized cash\_ratio vs predicted quantiles; this is the metric a lender will ask for.
- **Data drift:** population stability index on key features (sales level, price ratio, share of zero weeks) vs training window; alert above 0.25.
- **Pipeline health:** job duration, failures, freshness lag; alerts to email or Slack.
- **Dashboard:** internal Streamlit app reading the monitoring tables.

## Repository layout, configuration and testing

One Python monorepo managed with `uv`, a `src/` package split by domain, and config in versioned YAML validated by Pydantic.

```
stockcash/
  pyproject.toml              # uv, ruff, mypy, pytest config
  configs/
    default.yaml              # simulator, underwriting, report defaults
    tenants/{tenant_id}.yaml  # overrides + CSV column mappings
  src/stockcash/
    core/        # as_of-aware data access, money/date types, config loader, seeding
    schema/      # Pandera and Pydantic models for canonical tables
    ingest/      # connectors (csv_excel.py, m5.py, erp_*.py), validation, trust score
    features/    # demand.py, lot.py, merchant.py, clustering.py
    demand/      # train.py, predict.py, baselines.py, metrics.py, quantiles.py
    pseudoloans/ # policies.py, generator.py, labels.py (FIFO allocation)
    simulate/    # copula.py, engine.py, markdown.py, receivables.py
    underwrite/  # advance.py, liquidation.py, rules.py, benchmark.py
    report/      # classify.py, reorder.py, render.py, templates/
    api/         # app.py, routers/, auth.py, models.py
    jobs/        # flows for sync, features, forecast, score, retrain, monitor
    storage/     # postgres repo classes, object store client, migrations (alembic)
  apps/streamlit/              # internal monitoring and exploration UI
  tests/
    unit/  property/  integration/  backtest/
  docker/  docker-compose.yml  Makefile
```

**Configuration:** `default.yaml` → tenant YAML → request options, merged and validated into a frozen Pydantic `Settings` object; its SHA-256 hash is the `config_hash` stored with every result.

**Testing strategy:**

| Level | What | Tools |
| --- | --- | --- |
| Unit | Feature functions, quantile interpolation, FIFO allocation, advance formula | pytest |
| Property | Simulator invariants: cash\_ratio ≥ 0; sold ≤ available; zero demand → zero cash; more existing stock ahead of the lot never raises its cash\_ratio; same seed → same output | hypothesis |
| Leakage | Perturbing data after as\_of leaves features and forecasts unchanged | pytest |
| Integration | Upload → ingest → forecast → score on a small fixture tenant via the API | pytest, testcontainers (Postgres) |
| Backtest | Calibration of demand and lot quantiles on M5 holdout; fails CI if coverage leaves bounds | pytest marker, run nightly |
| Golden files | Fixed seed and fixture → exact score JSON; detects unintended changes | pytest snapshot |

**Tooling:** ruff (lint and format), mypy (strict on `core`, `simulate`, `underwrite`), pre-commit, GitHub Actions running unit, property and integration tests on every push; backtests nightly.

## Implementation order

Build the pure core first on M5, add I/O and API second, and only then multi-tenant operations. Each milestone ends with a passing test suite and a short MLflow-backed write-up.

| # | Milestone | Modules | Build plan phase | Exit criterion |
| --- | --- | --- | --- | --- |
| 1 | Skeleton | `core`, `schema`, repo tooling, CI | 1 | CI green; as\_of data access with leakage test |
| 2 | M5 ingest + features | `ingest/m5`, `features/demand` | 1 | Feature snapshots for all series in under 10 min |
| 3 | Demand model | `demand` | 1 | Beats baselines; P10–P90 coverage 76–84% |
| 4 | Pseudo-loans | `pseudoloans` | 1 | 10,000+ labeled lots across profiles |
| 5 | Simulator | `simulate` | 1 | Property tests pass; P05 lot coverage 3–7% |
| 6 | Underwriting + benchmark | `underwrite` | 1 | Side-by-side backtest vs revenue rule |
| 7 | API v0 (local) | `api`, `storage` (Postgres) | 1 | Score a lot via HTTP in under 1 s |
| 8 | CSV ingest + validation | `ingest/csv_excel`, trust score | 2 | A real merchant export ingested with a quality report |
| 9 | Trapped-cash report | `report` | 2 | PDF generated for a real merchant |
| 10 | Jobs + monitoring | `jobs`, Streamlit monitor | 3 | Nightly pipeline runs unattended for 2 weeks |
| 11 | ERP connector | `ingest/erp_*` | 3 | Daily sync for first pilot customer |
| 12 | Multi-tenant hardening | RLS, auth, audit, backups | 3 | Security checklist complete; restore test passed |

Milestones 1–7 map to weeks 1–10 of the build plan; 8–9 to Phase 2; 10–12 to Phase 3.

## Open technical decisions

- [ ] Granularity: weekly only, or daily demand for fast movers? (Default weekly; revisit if lots with 30-day tenor need finer resolution.)
- [ ] Quantile set vs full sample output from the demand model: are seven quantiles enough for the tail, or train a distributional model (e.g. negative binomial with LightGBM-predicted parameters)?
- [ ] Copula correlations: estimate per tenant or pool across tenants? Pooling is more stable but may understate idiosyncratic risk.
- [ ] Elasticity: fixed category defaults or estimated from `markdown_event`? Depends on how much markdown history merchants have.
- [ ] Prefect vs cron + job table for Phase 2: Prefect adds a service to run; cron is enough below \~10 tenants.
- [ ] Which ERP to integrate first: decide from Phase 0 interview results.
- [ ] Self-hosted object storage (MinIO on the VM) vs managed S3-compatible storage: cost vs operational burden.
- [ ] License for the engine if parts are ever open-sourced as a portfolio piece: keep the underwriting rules private either way.
