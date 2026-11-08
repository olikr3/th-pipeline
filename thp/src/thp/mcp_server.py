"""The engine as an MCP server: the CLI's commands as tools an MCP client can call.

Run it with ``thp mcp`` (stdio, for clients that launch the server) or
``thp mcp --transport streamable-http`` (a long-running server at ``/mcp``).
Data paths are fixed when the server starts, so clients name tenants and dates,
never files on the server, except ``ingest_forecasts``, which reads a file the
server can see.
"""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from thp.core.asof import AsOfStore
from thp.core.config import Settings, load_settings
from thp.core.paths import data_root
from thp.demand.baseline import seasonal_naive
from thp.demand.external import external_forecast
from thp.demand.forecast import write_forecast
from thp.features.demand import tenant_week_start, write_demand_features
from thp.ingest.demo import DemoSpec, build_demo, demo_lots
from thp.ingest.forecasts import read_external_forecasts, write_external_forecasts
from thp.scoring import BASELINE, score_lot
from thp.simulate.assemble import LotRequest

INSTRUCTIONS = """\
The thp inventory risk engine. It scores inventory lots (a purchase order or stock \
on hand) for inventory financing: it forecasts demand, simulates how fast the lot \
turns into cash, and proposes an advance.

Typical flow: `list_tenants`, then `score_lot` with a lot request. With no data yet, \
`make_demo` creates the synthetic tenant `demo` and three example lots \
(`list_demo_lots`, `get_demo_lot`). `as_of` is the date a computation pretends to \
be at: only data dated on or before it is used."""

Source = Annotated[
    str,
    Field(description=f"{BASELINE!r} (seasonal naive) or the name of an ingested external source"),
]
Tenant = Annotated[str, Field(description="tenant id, e.g. 'demo' or 'm5'")]
AsOf = Annotated[date, Field(description="only data dated on or before this day is used")]

_WRITES = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True)
_READS = ToolAnnotations(readOnlyHint=True)


def build_server(
    *,
    canonical: Path | None = None,
    features: Path | None = None,
    forecasts: Path | None = None,
    lots: Path | None = None,
    config: tuple[Path, ...] = (),
) -> MCPServer:
    """An MCP server over the data under these paths, with ``config`` YAML layers.

    Paths left out default to the data root (see ``thp.core.paths``).
    """
    data = data_root()
    canonical = canonical or data / "canonical"
    features = features or data / "features"
    forecasts = forecasts or data / "forecasts"
    lots = lots or data / "demo" / "lots"
    settings = load_settings(*config)
    server = MCPServer("thp", instructions=INSTRUCTIONS)

    def store(tenant_id: str, as_of: date) -> AsOfStore:
        return AsOfStore(
            root=canonical,
            tenant_id=tenant_id,
            as_of=as_of,
            ingest_grace=settings.asof.ingest_grace,
        )

    @server.tool(annotations=_READS)
    def list_tenants() -> list[str]:
        """The tenants with canonical data."""
        root = canonical / "tenant"
        return sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []

    @server.tool(name="score_lot", annotations=_READS)
    def score_lot_tool(
        lot: LotRequest, tenant_id: Tenant, source: Source = BASELINE
    ) -> dict[str, Any]:
        """Score a lot request: forecast, simulate cash conversion, propose an advance.

        Returns the lot cost, cash-ratio quantiles at 30/60/90 days and the tenor,
        P(cash ratio < 1), expected unsold value and the advance amount (q05 of the
        cash ratio at the tenor x lot cost / (1 + fee), capped at 80% of cost).
        Set ``lot.options`` to change the number of paths, the seed or the confidence.
        """
        n_paths = lot.options.n_paths
        overrides = {"simulate": {"n_paths": n_paths}} if n_paths else {}
        run_settings = load_settings(*config, overrides)
        with _tool_errors():
            return score_lot(canonical, tenant_id, lot, run_settings, source)

    @server.tool(name="forecast", annotations=_WRITES)
    def forecast_tool(tenant_id: Tenant, as_of: AsOf, source: Source = BASELINE) -> dict[str, Any]:
        """Write demand quantiles (q05...q95) for the next weeks per SKU x location."""
        with _tool_errors():
            return _forecast(store(tenant_id, as_of), settings, source, forecasts)

    @server.tool(annotations=_WRITES)
    def build_features(tenant_id: Tenant, as_of: AsOf, latest_only: bool = False) -> str:
        """Write the weekly demand feature panel (or only its last week); returns its path."""
        with _tool_errors():
            path = write_demand_features(store(tenant_id, as_of), features, latest_only=latest_only)
        return str(path)

    @server.tool(annotations=_WRITES)
    def ingest_forecasts(
        tenant_id: Tenant,
        source: Annotated[str, Field(description="source name: lowercase letters, digits, _")],
        file: Annotated[str, Field(description="path of a CSV or Parquet file on the server")],
        issued: Annotated[
            date | None, Field(description="issue date, if the file has no issued_date column")
        ] = None,
    ) -> dict[str, Any]:
        """Load forecasts from another source (a planning tool, an ERP) for ``forecast``
        and ``score_lot`` to use with ``source``. Re-uploads replace earlier values."""
        with _tool_errors():
            # Reference tables only; as_of plays no part.
            tenant = AsOfStore(root=canonical, tenant_id=tenant_id, as_of=date.today())
            frame = read_external_forecasts(
                Path(file),
                tenant_id=tenant_id,
                source=source,
                week_start=tenant_week_start(tenant),
                issued_date=issued,
            )
            path = write_external_forecasts(frame, canonical)
        return {"rows": frame.height, "path": str(path)}

    @server.tool(annotations=_WRITES)
    def make_demo(
        skus: int = DemoSpec.n_skus, weeks: int = DemoSpec.weeks, seed: int = DemoSpec.seed
    ) -> dict[str, Any]:
        """Generate the synthetic tenant ``demo`` and its example lots, replacing earlier ones."""
        spec = DemoSpec(n_skus=skus, weeks=weeks, seed=seed)
        with _tool_errors():
            rows = build_demo(canonical, spec)
            lots.mkdir(parents=True, exist_ok=True)
            names = []
            for name, lot in demo_lots(canonical, spec).items():
                (lots / f"{name}.json").write_text(json.dumps(lot, indent=2) + "\n")
                names.append(name)
        return {"tenant_id": "demo", "rows": rows, "lots": names}

    @server.tool(annotations=_READS)
    def list_demo_lots() -> list[str]:
        """The names of the example lot requests written by ``make_demo``."""
        return sorted(p.stem for p in lots.glob("*.json")) if lots.is_dir() else []

    @server.tool(annotations=_READS)
    def get_demo_lot(name: str) -> dict[str, Any]:
        """An example lot request, ready to pass to ``score_lot`` (tenant ``demo``)."""
        path = lots / f"{name}.json"
        if path.parent != lots or not path.is_file():
            raise ToolError(f"no demo lot {name!r}; see list_demo_lots")
        result: dict[str, Any] = json.loads(path.read_text())
        return result

    return server


def _forecast(store: AsOfStore, settings: Settings, source: str, out: Path) -> dict[str, Any]:
    result = seasonal_naive(store, settings.demand)
    skipped = 0
    if source != BASELINE:
        external = external_forecast(store, source, result, settings.demand)
        result = external.forecast
        skipped = external.skipped.select("sku_id", "location_id").n_unique()
    return {
        "series": result.select("sku_id", "location_id").n_unique(),
        "rows": result.height,
        "skipped_series": skipped,
        "path": str(write_forecast(result, out)),
    }


@contextmanager
def _tool_errors() -> Iterator[None]:
    """Turn the engine's input errors into tool errors the client sees in full."""
    try:
        yield
    except (ValueError, KeyError, FileNotFoundError) as exc:
        raise ToolError(str(exc)) from exc
