# Cash-conversion simulator: why shocks

Why the simulator draws shared random factors ("shocks") instead of sampling every SKU-week independently. See [the simulator overview](cash-conversion-simulator.md) for where they sit in the algorithm, and [the draw logic](simulator-draws.md) for how they are drawn and combined.

The forecast table only describes each SKU-week **on its own**. It says "SKU A sells 2–23 units in week 5 with 90% probability", but nothing about whether SKU A and SKU B tend to have bad weeks *together*. The simulator has to assume something about that, and the assumption matters a lot.

## What happens without shocks

If every SKU-week is drawn independently, the bad draws and the good draws cancel out. A 200-line lot over 13 weeks sums 2,600 draws. Say each weekly draw has about ±50% relative spread. The spread of the lot's total then shrinks by √2600 ≈ 51, to about **±1%**. The simulator would say cash_ratio_90 is almost certain: q05 sits barely below the median.

Reality doesn't work like that. When a merchant has a bad season (a cold summer, a lost marketplace listing, a weak economy, a competitor's promotion), most of its SKUs sell below forecast **at the same time**. These forecast errors are correlated, so they don't cancel.

## What a shock is

A shock is just a random number shared by many SKUs in one simulated future. Each path draws:

- **zm:** one number for the merchant. A low value means this future is a bad season for everything.
- **zc:** one number per category. A low value means this category is having a bad time.

Each SKU's draw mixes these shared numbers with its own noise ε, which is where the correlation comes from. "Shock" is just the usual name; "common factor" means the same thing.

With ρm + ρc = 0.25, a quarter of each SKU's variance is shared. That part doesn't diversify away, however many SKUs the lot has. In the example above, the lot total keeps roughly √0.25 × 50% ≈ **±25%** of spread, not ±1%. These numbers are rough, but they show the scale of the difference.

## Why it matters for the money

Underwriting advances against **q05 of cash_ratio**:

- **Independent draws:** q05 ≈ median, so it would advance nearly the full expected cash. The first bad season then hits every lot at once, and every loan is short at the same time.
- **With shocks:** q05 sits well below the median, so it advances less. That reflects the real risk.

So the shocks are what make the advance conservative in the right way. Without them, the simulator over-advances most on big, diversified lots, which look the safest.

## Why factors rather than a correlation matrix

Correlation could be modelled directly with a 200 × 200 correlation matrix per lot. That's about 20,000 numbers that can't be estimated from the data, plus an expensive decomposition per lot. Two shared factors give the right structure (same merchant → correlated, same category → more correlated) with just two parameters, ρm and ρc. Those can be estimated from past forecast errors and checked in the backtest.

## And over time

Persisting the shocks across weeks (AR(1)) applies the same idea along the time axis. A bad season lasts several weeks rather than flipping each week, so a bad path stays bad long enough to matter by day 90.

That is also the reason for the proposed per-SKU persistent component (gap 1 in [the overview](cash-conversion-simulator.md)). Within one SKU, nothing currently ties week 1 to week 2 except these shared shocks. If a SKU's forecast is simply too high, it stays too high every week, and that error doesn't cancel out across weeks either. It's the same argument, applied to a single SKU over time.
