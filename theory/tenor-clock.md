# Cash-conversion simulator: the tenor clock

When the T days of a lot's tenor start counting, why the spec is inconsistent about it, and the proposed resolution. This is gap 2 in [the simulator overview](cash-conversion-simulator.md), and an open decision in the [technical spec](../docs/specs/inventory-risk-engine.md).

## What the decision is

The **tenor** T is the loan's term: the lender advances money, and repayment is due T days later (30/60/90). cash_ratio(T) is the cash the lot has produced by the due date, divided by its cost. The question is which day the T days count from:

- **Clock A, from disbursement:** day 0 is when the lender pays out. For PO financing, that's the order date (paying the supplier).
- **Clock B, from receipt:** day 0 is when the goods arrive in the warehouse.

The spec mixes the two. The pseudo-loan labels count from receipt ("Realized label for a lot with tenor T, received at r"). The simulator runs from `as_of` = order date, draws the receipt week, and uses a 13-week forecast. Validation compares simulated against realised cash_ratio, so both must use the same clock, or the calibration numbers compare different things.

## What the difference looks like

A lot costs 100k. Once it arrives, it produces 10k of cash per week. Lead time is 6 weeks.

| Tenor | Clock A (from order) | Clock B (from receipt) |
|---|---|---|
| 30 days (4.3 weeks) | 0.00: goods not there yet | 0.43 |
| 60 days (8.6 weeks) | 0.26: 2.6 weeks of selling | 0.86 |
| 90 days (12.9 weeks) | 0.69: 6.9 weeks of selling | 1.29 |

Same lot, same demand, very different numbers. The advance is q05(C_T) × V / (1+f), so under clock A this lot gets a much smaller advance.

## Arguments for each

**Clock A (from disbursement):**

- **It's what the contract says.** A loan is due T days after the money goes out, not after the goods arrive. The lender's risk is exactly "how much cash exists on the due date", and lead time eats into it.
- **Lead-time risk is counted.** A late delivery means less cash by the due date. Under clock A the simulator's receipt draw matters, which is correct; under clock B it barely does.
- **It fits the forecast.** From `as_of` = order date, 90 days is 12.9 weeks, inside the 13-week forecast.
- **It's honest at scoring time.** The due date is known when the loan is made. Under clock B it would depend on a receipt date nobody knows yet.

**Clock B (from receipt):**

- **It measures sell-through quality:** how fast this stock turns into cash, independent of the supplier. That's comparable across lots and useful in the trapped-cash report.
- **But it breaks the forecast horizon.** Day 90 after a 6-week lead time is week 19+ from `as_of`. The forecasts stop at 13 weeks, and external sources often stop earlier. Extending them means forecasting further out with worse accuracy.
- **And it doesn't match any repayment date**, unless the money is paid out at receipt.

## The way out: the clock is disbursement, and the product decides when that is

The real question is *when the lender pays*, which depends on the product:

| Product | Disbursement | `as_of` | Receipt draw | Clock |
|---|---|---|---|---|
| **PO financing:** pay the supplier at order | order date | order date | yes, lead-time risk counts | A |
| **Stock financing:** advance against goods on hand or just received | at or after receipt | disbursement date | no, goods are there | A, which then equals B |

So clock B is just clock A for stock financing. "From receipt" is right only when the money goes out at receipt. The engine should always count from disbursement and take `disbursement_date` as input. Whether a receipt delay applies follows from whether the goods are on hand.

## Proposal

1. **The simulator counts T from the disbursement date.** For a PO lot, disbursement = `as_of` = order date, and the receipt week is drawn. For a stock position, there's no receipt draw.
2. **Change the pseudo-loan label to match:** cash collected from lot units between disbursement and disbursement + T. For synthetic M5 POs, generate both variants (PO financing from the order date, stock financing from receipt). Both get backtested, and each product gets its own calibration.
3. **Expect cash_ratio_30 ≈ 0 for PO lots with 6-week lead times.** That's the honest answer: a 30-day PO loan doesn't work for that supplier. Underwriting can then decline it or offer a longer tenor, rather than the simulator pretending otherwise.
4. **Map days to weeks precisely (gap 4 in the overview).** Horizon 1 starts after `as_of`'s last complete week, up to 6 days before `as_of`. So the tenor end date (disbursement + T) has to be converted into fractional forecast weeks, interpolating cash within the week.

## What it means for the engine

The input is a lot with `as_of`, `disbursement_date` and a flag (or an empty receipt distribution) for goods on hand. Tenor end dates are computed from disbursement. A 90-day tenor from a disbursement late in the week reaches into a 14th forecast week, so the default horizon is 14 weeks, not the spec's 13 (see `observations/wrong-assumptions.md`).

**Status:** decided 2026-10-25: the tenor counts from disbursement. Implemented in `thp/src/thp/simulate/engine.py` (`Lot.disbursement_date`; `expected_receipt_date=None` for goods on hand). The spec's pseudo-loan label was changed to match.
