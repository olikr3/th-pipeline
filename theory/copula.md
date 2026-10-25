# Cash-conversion simulator: the copula

What the copula in the simulator does, why it is a Gaussian copula, how its parameters are estimated, and its known weakness. See [the simulator overview](cash-conversion-simulator.md) for where it sits in the algorithm, [the draw logic](simulator-draws.md) for how its inputs are drawn, and [why shocks](why-shocks.md) for why correlation is needed at all.

## The problem a copula solves

The forecast gives **marginals**: one distribution per SKU-week, such as "SKU A, week 5: 2–23 units at 90%". The simulator needs the **joint** distribution: how all 2,600 SKU-weeks of a lot move together. The marginals don't contain that information.

A copula is the piece that joins them. A result called Sklar's theorem says any joint distribution can be split into two separate parts:

1. **the marginals:** how each variable is distributed on its own;
2. **the copula:** how the variables depend on each other, stripped of their units and scales.

That split fits this engine exactly. The forecast source (baseline, LightGBM or a merchant's planner) owns the marginals. The simulator owns the dependence. Either side can change without touching the other: switching forecast sources doesn't change the correlation model, and changing ρ doesn't distort any SKU's forecast.

## How the Gaussian copula works

Formally, a copula is a joint distribution of uniform variables. The Gaussian one is built in three moves:

```
correlated normals   →   correlated uniforms     →   correlated demands
x ~ N(0, Σ)             u = Φ(x)                    d = Q_i,t(u)
```

1. **Correlated normals.** Draw x with the desired correlation. Normals are used because correlating them is easy: add shared factors.
2. **Uniforms.** Applying Φ, the standard normal CDF, to a standard normal gives exactly Uniform(0,1). This is the probability integral transform. Each u is uniform on its own, but the u's stay correlated: if x₁ and x₂ tend to be low together, so do u₁ and u₂.
3. **Demands.** The inverse CDF of a uniform has exactly the target distribution, which is what `inverse_cdf` in `thp/src/thp/simulate/quantiles.py` does. Each SKU-week gets its own forecast shape, and the dependence carries through from step 1.

So the correlation is created on the normal scale, where it is easy, and then carried over to any marginal shape: intermittent, skewed, with point masses at zero.

## Our correlation structure

x doesn't use a full correlation matrix Σ but a factor structure:

```
x[i,t] = √ρm · zm[t] + √ρc · zc[c(i),t] + √(1−ρm−ρc) · ε[i,t]
```

The correlation matrix this implies:

| Pair of SKU-weeks | Latent correlation |
|---|---|
| two SKUs, same category, same week | ρm + ρc = 0.25 |
| two SKUs, different categories, same week | ρm = 0.15 |
| any two SKUs, k weeks apart | the above × φᵏ (shocks decay at 0.7 per week) |
| same SKU, k weeks apart | (ρm + ρc) · φᵏ, only through the shocks (gap 1 in the overview: no SKU-level persistence yet) |

## What ρ means, and what it doesn't

ρ is the correlation of the **latent normals**, not of demand in units. Demand correlation depends on the marginals: two intermittent SKUs that are mostly zero show a much lower unit correlation even with the same ρ.

What carries over unchanged is **rank correlation** (Spearman), since every step above preserves order:

```
Spearman = (6/π) · arcsin(ρ/2)   →   ρ = 0.25 gives 0.239
```

So "ρ = 0.25" reads as: rank the two SKUs' weekly outcomes, and those rankings correlate at about 0.24, whatever the forecast shapes.

## Estimating ρ: running the copula backwards

Run the three moves in reverse on history:

1. **Actual sales → u:** u = F(actual) under the forecast that was issued then. This is the PIT value: where the outcome fell in the forecast distribution. For discrete demand, draw u uniformly between F(y−1) and F(y) (a randomised PIT), otherwise the many zeros clump.
2. **u → normal scores:** Φ⁻¹(u).
3. **Correlate the scores:**
   - the average correlation between SKUs within one merchant-week estimates ρm;
   - the average within a category-week estimates ρm + ρc, so subtracting ρm gives ρc;
   - the autocorrelation within one SKU over weeks estimates φ and the per-SKU persistence.

The same PIT values give a calibration check for free: if the forecasts are calibrated, the u's are uniform. That is one more reason to keep the history of external forecasts ([ADR 0002](../docs/adr/0002-baseline-and-external-forecasts.md)).

## The known weakness: weak joint tails

The Gaussian copula makes **extreme outcomes rarely coincide**. Two SKUs with ρ = 0.25: given SKU A has a bad week, how likely is SKU B to have one too? (Simulated with 4 million draws.)

| "Bad week" = below its | Independent | Gaussian copula | t-copula (ν = 4) |
|---|---|---|---|
| q10 | 10% | 19% | 26% |
| q05 | 5% | 12% | 22% |
| q01 | 1% | 4% | 17% |

Under the Gaussian copula, the further into the tail, the closer the SKUs get to independent. In real crises, losses cluster *more* in the extremes, not less. This is the flaw that made the Gaussian copula infamous in 2008 credit-derivative pricing, where it was used for exactly this kind of joint-default question.

**How much it matters here:**

- Underwriting uses q05 of the lot's cash ratio, a moderate tail, not a 1-in-1,000 event.
- The merchant shock is persistent and shared by every SKU in a path, so the whole lot already moves together in bad paths.

The risk is real but bounded, and the spec's backtest checks it directly: realised cash below q05 should happen in 3–7% of lots, *including within each quarter*. Bad quarters with far more than 7% are the symptom of weak joint tails.

**The fix, if needed:** a **t-copula**, a small change. Draw one extra number per path, w = √(χ²_ν / ν), and divide every x in that path by it, then use the t distribution's CDF instead of Φ. A small w scales up every SKU's deviation in that path together, which creates joint extremes. It adds one parameter (ν, the degrees of freedom; lower means fatter joint tails) and a few lines in `copula.py`. The marginals stay exact, as before.

## Why not something else

- **A full correlation matrix:** too many parameters to estimate (see [why shocks](why-shocks.md)).
- **Resampling historical error vectors** (an empirical copula): this needs long, aligned history for the same set of SKUs, which a new merchant or a new lot doesn't have.
- **Vine or other copula families:** more flexible but much harder to estimate and explain, with no evidence yet that they are needed.

The Gaussian factor copula is the simplest choice that is honest about correlation. The t-copula is the planned upgrade if the backtest shows tail clustering.
