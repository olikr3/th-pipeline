from datetime import timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from thp.core.config import Settings, load_settings

DEFAULT_YAML = Path(__file__).parents[2] / "configs" / "default.yaml"


def test_default_yaml_matches_code_defaults() -> None:
    assert load_settings(DEFAULT_YAML) == Settings()


def test_later_layers_override_earlier_ones(tmp_path: Path) -> None:
    tenant = tmp_path / "tenant.yaml"
    tenant.write_text("asof:\n  ingest_grace_days: 3\n")

    settings = load_settings(DEFAULT_YAML, tenant, {"simulate": {"n_paths": 500}})

    assert settings.asof.ingest_grace == timedelta(days=3)
    assert settings.simulate.n_paths == 500


def test_config_hash_tracks_content_not_layering() -> None:
    base = load_settings(DEFAULT_YAML)
    same = load_settings({"simulate": {"n_paths": 2000}})
    changed = load_settings({"simulate": {"n_paths": 2001}})

    assert base.config_hash == same.config_hash
    assert base.config_hash != changed.config_hash
    assert len(base.config_hash) == 64


def test_unknown_keys_are_rejected() -> None:
    with pytest.raises(ValidationError):
        load_settings({"simulate": {"n_path": 10}})


def test_settings_are_frozen() -> None:
    settings = Settings()
    with pytest.raises(ValidationError):
        settings.simulate.n_paths = 1  # type: ignore[misc]
