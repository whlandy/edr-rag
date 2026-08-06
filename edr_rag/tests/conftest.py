from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def tmp_workdir(tmp_path: Path) -> Path:
    """Compatibility fixture for parser tests ported out of edr-wd."""
    return tmp_path
