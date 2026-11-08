from pathlib import Path
from typing import Any

import anyio
import pytest
from mcp import Client
from mcp.server.mcpserver import MCPServer

from thp.ingest.demo import DemoSpec
from thp.mcp_server import build_server

SPEC = DemoSpec(n_skus=12, weeks=80, seed=3)


@pytest.fixture(scope="module")
def server(tmp_path_factory: pytest.TempPathFactory) -> MCPServer:
    root = tmp_path_factory.mktemp("mcp")
    server = build_server(
        canonical=root / "canonical",
        features=root / "features",
        forecasts=root / "forecasts",
        lots=root / "lots",
    )
    result = _call(server, "make_demo", {"skus": SPEC.n_skus, "weeks": SPEC.weeks, "seed": 3})
    assert result["tenant_id"] == "demo"
    return server


def _call(server: MCPServer, tool: str, arguments: dict[str, Any] | None = None) -> Any:
    async def call() -> Any:
        async with Client(server) as client:
            result = await client.call_tool(tool, arguments or {})
        if result.is_error:
            raise RuntimeError(result.content[0].text)  # type: ignore[union-attr]
        content = result.structured_content
        # Non-object results are wrapped as {"result": ...}.
        return content["result"] if set(content) == {"result"} else content

    return anyio.run(call)


def test_lists_the_tools(server: MCPServer) -> None:
    async def names() -> set[str]:
        async with Client(server) as client:
            return {tool.name for tool in (await client.list_tools()).tools}

    assert anyio.run(names) == {
        "list_tenants",
        "score_lot",
        "forecast",
        "build_features",
        "ingest_forecasts",
        "make_demo",
        "list_demo_lots",
        "get_demo_lot",
    }


def test_scores_a_demo_lot(server: MCPServer) -> None:
    assert _call(server, "list_tenants") == ["demo"]
    names = _call(server, "list_demo_lots")
    assert "po_disciplined" in names
    lot = _call(server, "get_demo_lot", {"name": "po_disciplined"})
    lot["options"] = {"n_paths": 200, "seed": 1}
    scored = _call(server, "score_lot", {"lot": lot, "tenant_id": "demo"})
    assert scored["tenant_id"] == "demo"
    assert 0 < scored["advance_amount"] <= 0.8 * scored["lot_cost"]
    again = _call(server, "score_lot", {"lot": lot, "tenant_id": "demo"})
    assert again == scored


def test_forecast_and_features_write_files(server: MCPServer) -> None:
    lot = _call(server, "get_demo_lot", {"name": "po_disciplined"})
    out = _call(server, "forecast", {"tenant_id": "demo", "as_of": lot["as_of"]})
    assert out["series"] > 0
    assert Path(out["path"]).is_file()
    path = _call(
        server, "build_features", {"tenant_id": "demo", "as_of": lot["as_of"], "latest_only": True}
    )
    assert Path(path).exists()


def test_engine_errors_reach_the_client(server: MCPServer) -> None:
    lot = _call(server, "get_demo_lot", {"name": "po_disciplined"})
    lot["lines"][0]["sku_id"] = "NO-SUCH-SKU"
    with pytest.raises(RuntimeError, match="no list price"):
        _call(server, "score_lot", {"lot": lot, "tenant_id": "demo"})
    with pytest.raises(RuntimeError, match="no demo lot"):
        _call(server, "get_demo_lot", {"name": "../canonical"})
