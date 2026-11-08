"""Where the engine's data lives when no path is given.

``THP_DATA`` wins when set. Otherwise, in a source checkout (``uv sync`` installs
``thp`` editable), it is ``data/`` at the repository root, so commands read and
write the same files whether they run from the root or from ``thp/``. An
installed package falls back to ``data/`` in the current directory.
"""

import os
from pathlib import Path

DATA_ENV = "THP_DATA"


def data_root() -> Path:
    if env := os.environ.get(DATA_ENV):
        return Path(env).expanduser().resolve()
    # <repo>/thp/src/thp/core/paths.py
    here = Path(__file__).resolve()
    project, src = here.parents[3], here.parents[2]
    if src.name == "src" and (project / "pyproject.toml").is_file():
        return project.parent / "data"
    return Path.cwd() / "data"
