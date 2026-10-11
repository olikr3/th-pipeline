"""Layered configuration: default YAML -> tenant YAML -> request options."""

import hashlib
import json
from collections.abc import Mapping
from datetime import timedelta
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Section(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class AsOfSettings(_Section):
    ingest_grace_days: int | None = Field(default=None, ge=0)

    @property
    def ingest_grace(self) -> timedelta | None:
        if self.ingest_grace_days is None:
            return None
        return timedelta(days=self.ingest_grace_days)


class SimulateSettings(_Section):
    n_paths: int = Field(default=2000, gt=0)


class Settings(_Section):
    asof: AsOfSettings = AsOfSettings()
    simulate: SimulateSettings = SimulateSettings()

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
