# ADR 0001: As-of data access, seeded randomness and Parquet storage

- **Status:** Accepted
- **Date:** 2026-10-11
- **Relates to:** [Inventory Risk Engine Technical Spec](../specs/inventory-risk-engine.md), principles 3 (reproducible by construction) and 4 (time-honest), Milestone 1

## Context

The engine makes dated claims about the future: a demand forecast, a cash-conversion distribution and an advance amount, each as of a given day. Before any real loan exists, the only evidence that these claims are any good comes from replaying history: score a lot as if it were its order date, then compare with what actually happened. A lender will rely on the resulting calibration figures, and must be able to replay any individual score.

Three properties of the core follow from that, and each needed a decision in Milestone 1 because every later module is built on top of them:

1. How code reads historical data without seeing past the date it is pretending to be at.
2. How a Monte Carlo result is made repeatable.
3. What format the engine stores its data in.

## Decision 1: all fact reads go through an as-of store

Fact tables are read only through `AsOfStore` (`thp/src/thp/core/asof.py`). A store is created for one tenant and one `as_of` date, and every scan it returns is already filtered to rows with an event date on or before `as_of`. Optionally it also filters to rows ingested no later than the end of `as_of` plus a grace period.

**Why.** During a backtest the whole dataset is on disk, including the period being predicted. If any feature reads past the pretend date, backtests look better than live performance will be. This kind of leakage usually enters through small mistakes (a centred rolling window, a join that picks up a later price) and is hard to spot in review. Putting the cut-off in the access layer means downstream code cannot make the mistake, however it is written.

**How it is checked.** A test adds 1,000 units to every sale after `as_of` and asserts the read is unchanged; a property test asserts that no row dated after `as_of` is returned for arbitrary dates.

**Alternatives considered.**

- *Filter by convention in each feature function.* Rejected: it relies on every author remembering, and one omission silently invalidates the calibration numbers.
- *Materialise a separate dataset per backtest cutoff.* Rejected: safe, but it multiplies storage and makes rolling-origin validation over many cutoffs slow.

## Decision 2: all randomness comes from an explicit seed

Random draws come only from `make_rng(seed, *stream)` (`thp/src/thp/core/seeding.py`), which returns a NumPy `Generator` on PCG64. The seed is stored with every score. The optional `stream` arguments select an independent substream of the same seed.

**Why.** The simulator draws 2,000 demand paths per lot, so without a fixed seed the same lot would produce a slightly different advance amount on every run. A fixed seed gives:

- **Audit:** the same inputs, config and seed reproduce a score exactly.
- **Tests:** golden-file tests and the "same seed, same output" invariant are only possible if runs repeat.
- **Debugging:** a suspicious score can be re-run exactly, not approximately.

Substreams exist because a large stock position is simulated in SKU blocks. With one shared generator the result would depend on the order the blocks ran in; with one substream per block it does not, so the work can be chunked or parallelised freely.

**Alternatives considered.**

- *Global `np.random.seed`.* Rejected: hidden global state, which breaks as soon as code runs in a different order or in parallel.
- *No seed, with enough paths that noise is negligible.* Rejected: it cannot satisfy exact replay, and it costs runtime.

## Decision 3: the engine stores data as Parquet

Raw extracts, canonical tables (until Postgres arrives in Milestone 7) and feature snapshots are stored as Parquet, laid out as `{root}/{table}/**/*.parquet`. CSV and Excel remain input formats only: they are parsed and typed once at ingestion.

**Why.**

- **Types are part of the file.** CSV is text, so every read re-guesses dates, decimals and identifiers (a SKU `00123` becoming the number 123). Parquet fixes each column's type at ingestion.
- **Reads can skip data.** Parquet is columnar and keeps min/max statistics per chunk, so a query for a few columns up to a date does not read the rest. This is what makes Decision 1 cheap: polars pushes the as-of filter into the file scan.
- **Size and speed.** M5 in long form is roughly 59 million sales rows; Parquet is several times smaller than the CSV and loads much faster.

**Alternatives considered.**

- *Read CSV directly.* Rejected for the reasons above; it also re-parses on every run.
- *Postgres from the start.* Deferred, not rejected: the spec keeps modelling code free of a database, and Milestones 1–6 need no server. Postgres holds the canonical tables from Milestone 7.

## Consequences

- Backtest and calibration numbers can be trusted to use only data available at the time, provided all fact reads go through `AsOfStore`. Reading Parquet files directly from modelling code bypasses the guarantee and should be treated as a bug.
- The ingestion-time filter is off by default (`asof.ingest_grace_days: null`). Backfilled history, such as M5 or a merchant's first upload, is all ingested on one day, so enforcing the filter would hide it for any historical `as_of`. It should be set per tenant once data arrives nightly. Until then, later corrections to old rows are visible in backtests.
- The as-of filter works on rows, not columns. A purchase order is kept if its `order_date` is on or before `as_of`, but its `actual_receipt_date` and `status` may describe later events. This must be handled when `purchase_order` is first used (Milestone 4).
- The seed is not part of `config_hash`; the two are stored separately, so the same configuration with a different seed has the same hash.
- Exact replay holds for the same code and library versions. A change in NumPy's or polars' numerics could alter results, which is why scores also record the simulator and model versions.
- Parquet files are not human-readable; inspecting data needs polars or a similar tool, not a text editor.
