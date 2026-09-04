"""Breach-vs-prerequisite outcome contract for perf suites (issue #8377).

A measured budget breach must fail on every lane; skips are reserved for
missing prerequisites (baseline/hardware) with grep-distinguishable reasons,
so a regression can never pass silently on a default run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import tests.perf.test_factory_creation_perf as factory_perf
import tests.perf.test_simulation_speed_perf as sim_speed_perf


def _perf_source(module_name: str) -> str:
    """Return the source text of one perf test module."""
    root = Path(__file__).resolve().parents[2]
    return (root / "tests" / "perf" / f"{module_name}.py").read_text(encoding="utf-8")


def test_perf_modules_never_skip_on_breach() -> None:
    """No bare breach-skip may remain in either perf suite."""
    for module_name in ("test_factory_creation_perf", "test_simulation_speed_perf"):
        assert "pytest.skip(" not in _perf_source(module_name), module_name


def test_factory_skipif_names_missing_baseline() -> None:
    """The missing-baseline skip keeps a prerequisite-naming reason."""
    marks = factory_perf.test_factory_creation_mean_within_budget.pytestmark
    reasons = [mark.kwargs.get("reason", "") for mark in marks if mark.name == "skipif"]
    assert reasons, "factory budget test must keep its baseline guard"
    assert any("Baseline file missing" in reason for reason in reasons)


def test_simulation_breach_fails_without_enforce(monkeypatch: pytest.MonkeyPatch) -> None:
    """A measured throughput breach fails even without ROBOT_SF_PERF_ENFORCE."""
    monkeypatch.setattr(sim_speed_perf, "_HARD_STEPS_PER_SEC", 1e12)
    monkeypatch.delenv("ROBOT_SF_PERF_ENFORCE", raising=False)
    with pytest.raises(pytest.fail.Exception, match="below hard threshold"):
        sim_speed_perf.test_simulation_step_throughput()


def test_simulation_soft_breach_fails_without_enforce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A soft-band breach fails once the hard floor is out of the way."""
    monkeypatch.setattr(sim_speed_perf, "_HARD_STEPS_PER_SEC", 0.0)
    monkeypatch.setattr(sim_speed_perf, "_SOFT_STEPS_PER_SEC", 1e12)
    monkeypatch.delenv("ROBOT_SF_PERF_ENFORCE", raising=False)
    with pytest.raises(pytest.fail.Exception, match="below soft threshold"):
        sim_speed_perf.test_simulation_step_throughput()


def test_factory_breach_fails_without_enforce(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A measured env-creation breach fails against a furnished baseline."""
    baseline = tmp_path / "factory_perf_baseline.json"
    baseline.write_text(
        '{"results": {"make_robot_env": {"mean_ms": 0.001}, '
        '"make_image_robot_env": {"mean_ms": 0.001}}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(factory_perf, "BASELINE_PATH", baseline)
    monkeypatch.delenv("ROBOT_SF_PERF_ENFORCE", raising=False)
    with pytest.raises(pytest.fail.Exception, match="exceeds hard budget"):
        # Pass through the real monkeypatch fixture: the test deletes
        # ROBOT_SF_FAST_DEMO itself before measuring.
        factory_perf.test_factory_creation_mean_within_budget(monkeypatch)
