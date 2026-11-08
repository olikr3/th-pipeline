import argparse
import json
from datetime import date
from pathlib import Path

from thp.core.asof import AsOfStore
from thp.core.config import load_settings
from thp.demand.baseline import seasonal_naive
from thp.demand.external import external_forecast
from thp.demand.forecast import write_forecast
from thp.features.demand import tenant_week_start, write_demand_features
from thp.ingest.demo import DemoSpec, build_demo, demo_lots
from thp.ingest.forecasts import read_external_forecasts, write_external_forecasts
from thp.ingest.m5 import build_m5
from thp.scoring import BASELINE, score_lot
from thp.simulate.assemble import LotRequest


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

    demo = commands.add_parser("make-demo", help="generate the synthetic demo tenant and lots")
    demo.add_argument("--canonical", type=Path, default=Path("data/canonical"))
    demo.add_argument("--lots", type=Path, default=Path("data/demo/lots"))
    demo.add_argument("--skus", type=int, default=DemoSpec.n_skus)
    demo.add_argument("--weeks", type=int, default=DemoSpec.weeks, help="weeks of history")
    demo.add_argument("--seed", type=int, default=DemoSpec.seed)
    demo.add_argument("--config", type=Path, action="append", default=[], help="YAML layer(s)")

    score = commands.add_parser("score-lot", help="score a lot request (JSON) and print the result")
    score.add_argument("lot", type=Path, help="lot request JSON, as written by make-demo")
    score.add_argument("--tenant", required=True)
    score.add_argument(
        "--source",
        default=BASELINE,
        help=f"{BASELINE!r} (seasonal naive) or the name of an ingested external source",
    )
    score.add_argument("--canonical", type=Path, default=Path("data/canonical"))
    score.add_argument("--config", type=Path, action="append", default=[], help="YAML layer(s)")

    serve = commands.add_parser("mcp", help="serve the engine's commands as an MCP server")
    serve.add_argument(
        "--transport", choices=["stdio", "streamable-http"], default="stdio", help="default: stdio"
    )
    serve.add_argument("--host", default="127.0.0.1", help="streamable-http only")
    serve.add_argument("--port", type=int, default=8000, help="streamable-http only")
    serve.add_argument("--canonical", type=Path, default=Path("data/canonical"))
    serve.add_argument("--features", type=Path, default=Path("data/features"))
    serve.add_argument("--forecasts", type=Path, default=Path("data/forecasts"))
    serve.add_argument("--lots", type=Path, default=Path("data/demo/lots"))
    serve.add_argument("--config", type=Path, action="append", default=[], help="YAML layer(s)")

    args = parser.parse_args()
    if args.command == "mcp":
        _serve_mcp(args)
        return
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
    elif args.command == "make-demo":
        spec = DemoSpec(n_skus=args.skus, weeks=args.weeks, seed=args.seed)
        for table, rows in build_demo(args.canonical, spec).items():
            print(f"{table:>15}: {rows:,} rows")
        args.lots.mkdir(parents=True, exist_ok=True)
        for name, lot in demo_lots(args.canonical, spec).items():
            path = args.lots / f"{name}.json"
            path.write_text(json.dumps(lot, indent=2) + "\n")
            print(f"{'lot':>15}: {path}")
    elif args.command == "score-lot":
        request = LotRequest.model_validate_json(args.lot.read_text())
        n_paths = request.options.n_paths
        overrides = {"simulate": {"n_paths": n_paths}} if n_paths else {}
        run_settings = load_settings(*args.config, overrides)
        scored = score_lot(args.canonical, args.tenant, request, run_settings, args.source)
        print(json.dumps(scored, indent=2))


def _serve_mcp(args: argparse.Namespace) -> None:
    from thp.mcp_server import build_server  # the MCP SDK is only needed here

    server = build_server(
        canonical=args.canonical.resolve(),
        features=args.features.resolve(),
        forecasts=args.forecasts.resolve(),
        lots=args.lots.resolve(),
        config=tuple(args.config),
    )
    if args.transport == "stdio":
        server.run("stdio")
    else:
        server.run("streamable-http", host=args.host, port=args.port)
