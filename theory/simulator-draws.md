# Cash-conversion simulator: the draw logic

Which random numbers the simulator draws, how they turn into demand, and how they are seeded. It complements [the simulator overview](cash-conversion-simulator.md); the spec section is "Cash-conversion simulator" in the [technical spec](../docs/specs/inventory-risk-engine.md).

The simulator draws everything **once, up front, for all 2,000 paths together**. After that, every step is deterministic arithmetic on those draws. A path is one coherent future, and every SKU in it lives in that same future.

## What gets drawn

For one lot with N paths, S SKUs, C categories and W weeks:

| Draw | Shape | Distribution | Represents |
|---|---|---|---|
| zm, merchant shock | `[N, W]` | N(0,1), AR(1) over weeks | how good or bad this future is for the whole merchant |
| zc, category shocks | `[N, C, W]` | N(0,1), AR(1) over weeks | how good or bad it is per category |
| zs, SKU level | `[N, S, W]` | N(0,1), AR(1) over weeks, slowly decaying | how far this SKU's forecast level is off in this future |
| ε, idiosyncratic noise | `[N, S, W]` | N(0,1), independent | everything specific to one SKU-week |
| receipt week | `[N]` | lead-time distribution around the PO's expected receipt | when the goods arrive; one per path, since the PO arrives as one delivery |
| receivable delay | `[N]` | per the B2B payment-delay distribution | how late B2B customers pay (not built yet: needs receivables data, Phase 3+) |

**Not drawn, but computed from the path's state:**

- **Markdowns:** triggered by stock age and cover, which depend on what sold.
- **Sales:** min(demand, stock).
- **Returns:** applied as an expected rate, (1 − return rate).

## From draws to demand

**1. Shocks get persistence.** For each path, week 1 is a fresh N(0,1). Each later week keeps 70% of the previous value and adds new noise:

```
zm[n,t] = 0.7 · zm[n,t−1] + √(1 − 0.7²) · η[n,t]
```

The √(1 − 0.7²) ≈ 0.71 keeps the variance at exactly 1 every week. zc works the same way, and zs too but with φs = 0.9, so a SKU whose level is off stays off for most of the 13 weeks.

**2. Mix into a latent normal per SKU-week:**

```
x[n,i,t] = √0.15 · zm[n,t] + √0.10 · zc[n,c(i),t] + √0.20 · zs[n,i,t] + √0.55 · ε[n,i,t]
         = 0.387 · zm     + 0.316 · zc          + 0.447 · zs        + 0.742 · ε
```

The defaults ρm = 0.15, ρc = 0.10, ρs = 0.20 and the persistences 0.7 and 0.9 are in `simulate` in `thp/configs/default.yaml`. ρs and φs are assumptions until they can be estimated from PIT values (see [the copula](copula.md)).

**3. Map to a uniform:** u = Φ(x), the standard normal CDF.

**4. Map to units:** demand = `inverse_cdf(knots[i,t], u)` (`thp/src/thp/simulate/quantiles.py`).

## One SKU-week, two paths

The SKU's forecast for week 1 is q05…q95 = 2, 4, 7, 10, 14, 19, 23.

**Path A, a bad season:** zm = −1.2, zc = −0.3, zs = +0.2, ε = +0.4 (this SKU itself is doing fine).

```
x = 0.387·(−1.2) + 0.316·(−0.3) + 0.447·0.2 + 0.742·0.4 = −0.465 − 0.095 + 0.089 + 0.297 = −0.174
u = Φ(−0.174) = 0.431
demand = 7 + (0.431 − 0.25)/0.25 · (10 − 7) = 9.2 units   (below the median of 10)
```

**Path B, a good season, with the same zs and ε:** zm = +1.0, zc = +0.5.

```
x = 0.387 + 0.158 + 0.089 + 0.297 = 0.932  →  u = 0.824  →  demand = 14 + (0.824 − 0.75)/0.15 · 5 = 16.5 units
```

The SKU's own luck is identical in both paths. The difference comes entirely from the shared future. In path A, all the other SKUs are also pulled down by the same zm = −1.2, so the whole lot sells less together. That is what the shocks exist for: without them, bad and good draws across SKUs cancel out and the lot's cash ratio looks far more certain than it is. See [why shocks](why-shocks.md).

Across all 2,000 paths, each SKU-week's demand still follows its forecast exactly, since u is uniform whatever ρ is. The shocks only decide which futures the bad draws land in together.

## The rest of a path

With demand `[N, S, W]` in hand, each path n runs week by week:

1. Lot stock is 0 before `receipt_week[n]`, and the full lot quantity from then on. Goods arriving during a week sell from the next full week.
2. Sales = min(demand, available). Older stock sells first, then lot stock.
3. Markdown check: if lot stock is older than 12 weeks or cover exceeds 26 weeks, the price drops by 20% of list price, at most once per 4 weeks and to at most 50% off, and that week's demand is multiplied by (price ratio)^−1.5.
4. Cash = lot units sold × price × (1 − return rate). The B2B share moves later by `delay[n]`.

At the end, cumulative cash over the T days from disbursement ÷ lot cost gives one cash_ratio per path and tenor ([the tenor clock](tenor-clock.md)). That makes 2,000 numbers per tenor, and their quantiles are the output.

## Seeding: repeatable and order-independent

Every draw type gets its own substream from `make_rng(seed, *stream)` ([ADR 0001](../docs/adr/0001-as-of-access-seeding-and-parquet.md)):

| Draw | Substream key |
|---|---|
| zm | `(seed, MERCHANT)` |
| zc | `(seed, CATEGORY, key(category))` |
| zs | `(seed, SKU_LEVEL, key(sku, location))` |
| ε | `(seed, NOISE, key(sku, location))` |
| receipt | `(seed, RECEIPT)` |

What this buys:

- **Same seed, same result:** a score can be replayed exactly.
- **Chunk-safe:** a 50,000-SKU stock position is simulated in blocks. Because each SKU's zs and ε are keyed by the SKU, not its position, results don't depend on block size or line order. The merchant and category shocks come from their own streams, so every block sees the same shared future, which is essential. If each block drew its own zm, the blocks would become independent futures, and the diversification problem would come back.
- **Adding a draw type later** (e.g. the per-path factor of a t-copula) doesn't change the existing draws, so old golden-file results stay stable.
- **Common random numbers:** two lots of the same merchant scored with the same seed share the same merchant shocks. Comparisons between them, such as "what if the PO were 20% smaller?", reflect the change, not random noise.

The key that gives a category or SKU a stable integer is a 64-bit BLAKE2b hash of its id (`stream_key` in `thp/src/thp/simulate/copula.py`). Python's `hash()` would change between processes, and a 32-bit hash such as CRC32 would give some of 50,000 SKUs the same draws. The stream kinds are numbered in `Stream` and must never be renumbered.
