# Bugs

Defects found in code or tests: what broke, why, and how it was fixed. Newest first. See [the observations index](README.md) for the entry format.

## Feature property test compared against a later as_of

- **Date:** 2026-10-25
- **Where:** `thp/tests/unit/test_demand_features.py`, `test_rows_do_not_change_when_later_data_arrives`
- **What happened:** the test builds the panel at an "earlier" as_of and compares it with the full panel cut at that date. Its hypothesis range allowed week 20 + 6 days, a Sunday *after* the fixture's `AS_OF` that completes week 20. That build legitimately has an extra week (81 rows against 77), so the test failed. It failed on a clean checkout too: the bug predates the simulator work, and hypothesis only now generated the corner case.
- **Impact:** test only; the feature code was right.
- **Resolution:** fixed with `assume(as_of <= AS_OF)`.

## Upper demand tail inaccurate near u = 1 in float32

- **Date:** 2026-10-18
- **Where:** `thp/src/thp/simulate/quantiles.py`, `U_EPS`
- **What happened:** draws were clipped to [1e-6, 1 − 1e-6]. In float32, `1 − u` is only resolved to about 6e-8 near 1, so at 1e-6 its relative error is several percent, and the extreme upper tail came out about 0.1% low. `test_extreme_u_stays_finite` caught it.
- **Impact:** small, and only for the most extreme draws, but systematic.
- **Resolution:** `U_EPS` raised to 1e-4. This truncates the tail at about 9 halvings of tail probability past q95 (see [tradeoffs](tradeoffs.md)).
