from pathlib import Path

import pytest

from thp.core.paths import DATA_ENV, data_root

REPO = Path(__file__).resolve().parents[3]


def test_checkout_uses_the_repo_root_from_anywhere(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(DATA_ENV, raising=False)
    for cwd in (REPO, REPO / "thp", tmp_path):
        monkeypatch.chdir(cwd)
        assert data_root() == REPO / "data"


def test_env_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(DATA_ENV, str(tmp_path / "elsewhere"))
    assert data_root() == tmp_path / "elsewhere"
