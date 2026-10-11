import argparse
from pathlib import Path

from thp.core.config import load_settings
from thp.ingest.m5 import build_m5


def main() -> None:
    parser = argparse.ArgumentParser(prog="thp")
    commands = parser.add_subparsers(dest="command", required=True)

    m5 = commands.add_parser("ingest-m5", help="convert the M5 CSVs into canonical Parquet")
    m5.add_argument("--raw", type=Path, default=Path("data/raw/m5"))
    m5.add_argument("--out", type=Path, default=Path("data/canonical"))
    m5.add_argument("--stores", nargs="+", help="only these store ids, e.g. CA_1")
    m5.add_argument("--config", type=Path, action="append", default=[], help="YAML layer(s)")

    args = parser.parse_args()
    settings = load_settings(*args.config)
    counts = build_m5(args.raw, args.out, settings.m5, stores=args.stores)
    for table, rows in counts.items():
        print(f"{table:>10}: {rows:,} rows")
