import argparse
from datetime import date
from pathlib import Path

from thp.core.asof import AsOfStore
from thp.core.config import load_settings
from thp.features.demand import write_demand_features
from thp.ingest.m5 import build_m5


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
