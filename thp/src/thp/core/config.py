"""Layered configuration: default YAML -> tenant YAML -> request options."""

import hashlib
import json
from collections.abc import Mapping
from datetime import timedelta
from pathlib import Path
from typing import Annotated, Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Section(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class AsOfSettings(_Section):
    ingest_grace_days: int | None = Field(default=None, ge=0)

    @property
    def ingest_grace(self) -> timedelta | None:
        if self.ingest_grace_days is None:
            return None
        return timedelta(days=self.ingest_grace_days)


class DemandSettings(_Section):
    horizon_weeks: int = Field(default=13, gt=0, le=52)
    # Trailing weeks of baseline forecast errors that set the quantile spread.
    residual_weeks: int = Field(default=52, gt=0)
    # External forecasts issued longer ago than this before as_of are ignored.
    external_max_age_days: int = Field(default=35, ge=0)


class SimulateSettings(_Section):
    n_paths: int = Field(default=2000, gt=0)
    # Gaussian copula (theory/copula.md): shares of each SKU-week's latent variance
    # from the merchant shock, the category shock and the SKU's persistent level.
    # The rest is noise per SKU-week, so they must sum to less than 1.
    rho_merchant: float = Field(default=0.15, ge=0, lt=1)
    rho_category: float = Field(default=0.10, ge=0, lt=1)
    rho_sku: float = Field(default=0.20, ge=0, lt=1)
    # AR(1) week-to-week persistence of the shared shocks and of the SKU level.
    shock_persistence: float = Field(default=0.7, ge=0, lt=1)
    sku_persistence: float = Field(default=0.9, ge=0, lt=1)

    @model_validator(mode="after")
    def _noise_share_positive(self) -> Self:
        if self.rho_merchant + self.rho_category + self.rho_sku >= 1:
            raise ValueError("rho_merchant + rho_category + rho_sku must be below 1")
        return self


class M5Settings(_Section):
    tenant_id: str = "m5"
    # M5 has no costs: unit_cost = sell_price * (1 - gross margin). Assumed values.
    gross_margin_by_category: dict[str, Annotated[float, Field(gt=0, lt=1)]] = {
        "FOODS": 0.25,
        "HOUSEHOLD": 0.35,
        "HOBBIES": 0.40,
    }
    default_gross_margin: float = Field(default=0.30, gt=0, lt=1)


class Settings(_Section):
    asof: AsOfSettings = AsOfSettings()
    demand: DemandSettings = DemandSettings()
    simulate: SimulateSettings = SimulateSettings()
    m5: M5Settings = M5Settings()

    @property
    def config_hash(self) -> str:
        """SHA-256 of the canonical JSON form; stored with every result."""
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


def load_settings(*layers: Path | Mapping[str, Any]) -> Settings:
    """Merge layers left to right (later wins) and validate.

    A layer is a YAML file path or an already-parsed mapping.
    """
    merged: dict[str, Any] = {}
    for layer in layers:
        if isinstance(layer, Path):
            layer = yaml.safe_load(layer.read_text()) or {}
        merged = _deep_merge(merged, layer)
    return Settings.model_validate(merged)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out
