import argparse
from datetime import date
from pathlib import Path

from thp.core.asof import AsOfStore
from thp.core.config import load_settings
from thp.demand.baseline import seasonal_naive
from thp.demand.external import external_forecast
from thp.demand.forecast import write_forecast
from thp.features.demand import tenant_week_start, write_demand_features
from thp.ingest.forecasts import read_external_forecasts, write_external_forecasts
from thp.ingest.m5 import build_m5

BASELINE = "baseline"


def main() -> None:
    parser = argparse.ArgumentParser(prog="thp")
    commands = parser.add_subparsers(dest="command", required=True)

    m5 = commands.add_parser("ingest-m5", help="convert the M5 CSVs into canonical Parquet")
    m5.add_argument("--raw", type=Path, default=Path("data/raw/m5"))
    m5.add_argument("--out", type=Path, default=Path("data/canonical"))
    m5.add_argument("--stores", nargs="+", help="only these store ids, e.g. CA_1")
    m5.add_argument("--config", type=Path, action="append", default=[], help="YAML layer(s)")

    features = commands.add_parser("build-features", help="write the weekly demand feature panel")
    features.add_argument("--tenant", required=True)
    features.add_argument("--as-of", type=date.fromisoformat, required=True)
    features.add_argument("--canonical", type=Path, default=Path("data/canonical"))
    features.add_argument("--out", type=Path, default=Path("data/features"))
    features.add_argument("--latest-only", action="store_true", help="only the last week")
    features.add_argument("--config", type=Path, action="append", default=[], help="YAML layer(s)")

    ingest_fc = commands.add_parser(
        "ingest-forecasts", help="load forecasts from another source into canonical Parquet"
    )
    ingest_fc.add_argument("--tenant", required=True)
    ingest_fc.add_argument("--source", required=True, help="e.g. planner; lowercase, digits, _")
    ingest_fc.add_argument("--file", type=Path, required=True, help="CSV or Parquet")
    ingest_fc.add_argument(
        "--issued", type=date.fromisoformat, help="issue date, if the file has no issued_date"
    )
    ingest_fc.add_argument("--canonical", type=Path, default=Path("data/canonical"))
    ingest_fc.add_argument("--config", type=Path, action="append", default=[], help="YAML layer(s)")

    forecast = commands.add_parser("forecast", help="write demand quantiles for the next weeks")
    forecast.add_argument("--tenant", required=True)
    forecast.add_argument("--as-of", type=date.fromisoformat, required=True)
    forecast.add_argument(
        "--source",
        default=BASELINE,
        help=f"{BASELINE!r} (seasonal naive) or the name of an ingested external source",
    )
    forecast.add_argument("--canonical", type=Path, default=Path("data/canonical"))
    forecast.add_argument("--out", type=Path, default=Path("data/forecasts"))
    forecast.add_argument("--config", type=Path, action="append", default=[], help="YAML layer(s)")

    args = parser.parse_args()
    settings = load_settings(*args.config)
    if args.command == "ingest-m5":
        counts = build_m5(args.raw, args.out, settings.m5, stores=args.stores)
        for table, rows in counts.items():
            print(f"{table:>10}: {rows:,} rows")
    elif args.command == "build-features":
        store = AsOfStore(
            root=args.canonical,
            tenant_id=args.tenant,
            as_of=args.as_of,
            ingest_grace=settings.asof.ingest_grace,
        )
        print(write_demand_features(store, args.out, latest_only=args.latest_only))
    elif args.command == "ingest-forecasts":
        # Reference tables only; as_of plays no part.
        tenant = AsOfStore(root=args.canonical, tenant_id=args.tenant, as_of=date.today())
        frame = read_external_forecasts(
            args.file,
            tenant_id=args.tenant,
            source=args.source,
            week_start=tenant_week_start(tenant),
            issued_date=args.issued,
        )
        path = write_external_forecasts(frame, args.canonical)
        print(f"{frame.height:,} rows -> {path}")
    elif args.command == "forecast":
        store = AsOfStore(
            root=args.canonical,
            tenant_id=args.tenant,
            as_of=args.as_of,
            ingest_grace=settings.asof.ingest_grace,
        )
        result = seasonal_naive(store, settings.demand)
        if args.source != BASELINE:
            external = external_forecast(store, args.source, result, settings.demand)
            result = external.forecast
            if external.skipped.height:
                n = external.skipped.select("sku_id", "location_id").n_unique()
                print(f"skipped {n:,} series without sales history to derive missing quantiles")
        series = result.select("sku_id", "location_id").n_unique()
        print(f"{series:,} series, {result.height:,} rows -> {write_forecast(result, args.out)}")
