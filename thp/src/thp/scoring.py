"""Score a lot end to end: forecast, assemble, simulate, advance."""

from pathlib import Path
from typing import Any

from thp.core.asof import AsOfStore
from thp.core.config import Settings
from thp.demand.baseline import seasonal_naive
from thp.demand.external import external_forecast
from thp.simulate.assemble import LotRequest, assemble_lot
from thp.simulate.engine import simulate_lot
from thp.underwrite.advance import advance

BASELINE = "baseline"


def score_lot(
    canonical: Path,
    tenant_id: str,
    request: LotRequest,
    settings: Settings,
    source: str = BASELINE,
) -> dict[str, Any]:
    """The lot's score as a JSON-ready dict.

    ``settings`` should already include the request's ``n_paths`` override, so its
    config hash describes the run.
    """
    store = AsOfStore(
        root=canonical,
        tenant_id=tenant_id,
        as_of=request.as_of,
        ingest_grace=settings.asof.ingest_grace,
    )
    forecast = seasonal_naive(store, settings.demand)
    if source != BASELINE:
        forecast = external_forecast(store, source, forecast, settings.demand).forecast
    lot = assemble_lot(store, request, forecast)
    simulation = simulate_lot(lot, request.options.seed, settings.simulate)
    offer = advance(
        simulation,
        request.tenor_days,
        request.fee_rate,
        settings.underwrite,
        request.options.confidence,
    )
    summary = simulation.summary()
    unsold_value = float((simulation.unsold_units * lot.unit_cost).sum(axis=1).mean())
    return {
        "name": request.name,
        "tenant_id": tenant_id,
        "as_of": request.as_of.isoformat(),
        "disbursement_date": request.disbursed.isoformat(),
        "lot_cost": round(lot.cost, 2),
        "advance_amount": round(offer.amount, 2),
        "advance_rate": round(offer.rate, 4),
        "advance_capped": offer.capped,
        "tenor_days": offer.tenor_days,
        "confidence": offer.confidence,
        "cash_ratio": summary["cash_ratio"],
        "p_cash_ratio_below_1": summary["p_cash_ratio_below_1"],
        "expected_unsold_value": round(unsold_value, 2),
        "versions": {
            "forecast": str(forecast["model_version"].first()),
            "config_hash": settings.config_hash,
            "seed": request.options.seed,
        },
    }
