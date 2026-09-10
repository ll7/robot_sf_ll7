"""Tests for the advanced reproducibility example."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from robot_sf.benchmark.types import EpisodeRecord, MetricsBundle

if TYPE_CHECKING:
    from types import ModuleType

_EXAMPLE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "advanced"
    / "35_reproducible_episode_replay.py"
)


def _load_example() -> ModuleType:
    """Load the numbered example as a test module without executing ``main``."""
    spec = importlib.util.spec_from_file_location("reproducible_episode_replay", _EXAMPLE_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load {_EXAMPLE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(seed: int, duration: float) -> EpisodeRecord:
    """Build a small record fixture with runtime metadata variation."""
    return EpisodeRecord(
        version="v1",
        episode_id=f"quickstart_demo_crossing_basic--{seed}",
        scenario_id="quickstart_demo_crossing_basic",
        seed=seed,
        metrics=MetricsBundle(values={"steps": 4.0, "duration_s": duration}),
        algo="zero_action",
        horizon=4,
        timing={"wall_time_s": duration},
        raw={"duration_marker": duration},
    )


def test_report_distinguishes_same_and_changed_seed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The report uses the shared canonical policy and never assigns seed causality."""
    module = _load_example()
    records = {118: _record(118, 1.0), 119: _record(119, 2.0)}
    monkeypatch.setattr(module, "_run_record", lambda seed, max_steps: records[seed])

    report = module.build_replay_report(tmp_path, max_steps=4)

    assert report["status"] == "pass"
    assert report["comparisons"]["same_seed"]["status"] == "identical"
    changed_seed = report["comparisons"]["different_seed"]
    assert changed_seed["status"] == "different"
    assert changed_seed["first_identity"]["seed"] != changed_seed["second_identity"]["seed"]
    assert "causal" in changed_seed["interpretation"]
    assert report["canonical_policy"]["excluded_runtime_fields"] == ["raw", "timing"]
    assert (tmp_path / "same_seed_a.json").is_file()
    assert (tmp_path / "same_seed_b.json").is_file()
    assert (tmp_path / "different_seed.json").is_file()
    persisted = json.loads((tmp_path / "comparison.json").read_text(encoding="utf-8"))
    assert persisted == report
    assert module.main(["--output-dir", str(tmp_path / "cli-json"), "--format", "json"]) == 0
    assert module.main(["--output-dir", str(tmp_path / "cli-text"), "--format", "text"]) == 0
