"""Run the archived demo's real scenarios on development episode seeds."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from tests.support.development_scenarios import development_scenarios


@pytest.fixture(autouse=True)
def development_demo_seeds(monkeypatch):
    """Replace only the demo's loaded seed lists, preserving its scenario loader."""
    demo = importlib.import_module("examples._archived.classic_interactions_pygame")
    monkeypatch.setattr(demo, "load_classic_matrix", lambda path: development_scenarios(Path(path)))
