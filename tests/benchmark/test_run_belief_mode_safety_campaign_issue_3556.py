"""Tests for issue #3556 belief-mode campaign decision screening."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "benchmark"
    / "run_belief_mode_safety_campaign_issue_3556.py"
)
_SPEC = importlib.util.spec_from_file_location("_issue_3556_campaign", _MODULE_PATH)
assert _SPEC is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
classify_screened_decision = _MODULE.classify_screened_decision
check_campaign_readiness = _MODULE.check_campaign_readiness
run_campaign = _MODULE.run_campaign
main = _MODULE.main


def test_run_mode_reports_malformed_jsonl_line(tmp_path, monkeypatch) -> None:
    """Malformed runner JSONL names the file and offending line."""

    def _write_bad_jsonl(*_args, **_kwargs) -> None:
        (tmp_path / "episodes_oracle.jsonl").write_text('{"ok": true}\n{bad\n', encoding="utf-8")

    monkeypatch.setattr(_MODULE, "run_map_batch", _write_bad_jsonl)

    with pytest.raises(ValueError, match="line 2"):
        _MODULE.run_mode(
            mode="oracle",
            scenarios=[],
            out_dir=tmp_path,
            fov_degrees=120.0,
            horizon=10,
            dt=0.1,
            workers=1,
        )


def test_aggregate_ignores_nonfinite_min_clearance() -> None:
    """Infinite clearances should not poison aggregate clearance summaries."""

    report = _MODULE.aggregate(
        [
            {"metrics": {"total_collision_count": 0, "near_misses": 0, "min_clearance": float("inf")}},
            {"metrics": {"total_collision_count": 0, "near_misses": 0, "min_clearance": 0.5}},
        ]
    )

    assert report["worst_min_clearance"] == 0.5


def _ready_kwargs() -> dict[str, float | int]:
    """Valid, strictly-positive campaign run geometry for readiness checks."""
    return {"fov_degrees": 120.0, "horizon": 300, "dt": 0.1, "workers": 4}


def _scenario_set(_tmp_path) -> Path:
    """Return the checked-in occlusion-bearing #3556 scenario set."""
    return _MODULE.REPO_ROOT / _MODULE.DEFAULT_SCENARIO_SET


def _mode(collision_rate: float, near_misses: int) -> dict[str, float | int]:
    """Build the minimal aggregate shape consumed by the decision classifier."""
    return {
        "episodes": 3,
        "collision_rate": collision_rate,
        "total_near_misses": near_misses,
    }


def test_oracle_unsafe_blocks_interpretation_even_without_mode_delta() -> None:
    """A collide-heavy oracle cannot support a dropped-vs-retained safety claim."""
    decision = classify_screened_decision(
        {
            "oracle": _mode(collision_rate=1.0, near_misses=3),
            "uncertain_retained": _mode(collision_rate=1.0, near_misses=3),
            "uncertain_dropped": _mode(collision_rate=1.0, near_misses=3),
        }
    )

    assert decision["decision"] == "inconclusive_oracle_unsafe"
    assert decision["screening_status"] == "oracle_unsafe"
    assert decision["oracle_near_safe"] is False
    assert decision["mode_is_discriminating"] is False


def test_near_safe_nondiscriminating_matrix_stays_inconclusive() -> None:
    """Near-safe oracle alone is not enough when dropped and retained match."""
    decision = classify_screened_decision(
        {
            "oracle": _mode(collision_rate=0.0, near_misses=0),
            "uncertain_retained": _mode(collision_rate=0.0, near_misses=0),
            "uncertain_dropped": _mode(collision_rate=0.0, near_misses=0),
        }
    )

    assert decision["decision"] == "inconclusive"
    assert decision["screening_status"] == "near_safe_nondiscriminating"
    assert decision["oracle_near_safe"] is True
    assert decision["mode_is_discriminating"] is False


def test_near_safe_discriminating_matrix_recommends_revise() -> None:
    """Dropped mode becoming less safe under a near-safe oracle is actionable."""
    decision = classify_screened_decision(
        {
            "oracle": _mode(collision_rate=0.0, near_misses=0),
            "uncertain_retained": _mode(collision_rate=0.0, near_misses=1),
            "uncertain_dropped": _mode(collision_rate=0.0, near_misses=4),
        }
    )

    assert decision["decision"] == "revise"
    assert decision["screening_status"] == "near_safe_discriminating"
    assert decision["oracle_near_safe"] is True
    assert decision["mode_is_discriminating"] is True


def test_campaign_readiness_ready_with_valid_inputs(tmp_path) -> None:
    """Valid pinned inputs pass every fail-closed readiness check for the real campaign."""
    report = check_campaign_readiness(_scenario_set(tmp_path), [101, 102, 103], **_ready_kwargs())

    assert report["ready"] is True
    assert report["failed_checks"] == []
    names = {c["name"] for c in report["checks"]}
    # The gate must actually exercise the oracle-near-safety + planner contracts, not just inputs.
    assert {"oracle_near_safety_contract", "uncertainty_planner_contract"} <= names


def test_campaign_readiness_missing_scenario_set(tmp_path) -> None:
    """A missing scenario set fails closed before any compute."""
    report = check_campaign_readiness(tmp_path / "absent.yaml", [101, 102], **_ready_kwargs())

    assert report["ready"] is False
    assert "scenario_set_exists" in report["failed_checks"]


def test_campaign_readiness_duplicate_seeds_block(tmp_path) -> None:
    """A seed matrix with duplicates is not a valid pinned matrix."""
    report = check_campaign_readiness(_scenario_set(tmp_path), [101, 101, 102], **_ready_kwargs())

    assert report["ready"] is False
    assert "seeds_pinned" in report["failed_checks"]


def test_campaign_readiness_nonpositive_geometry(tmp_path) -> None:
    """Zero/negative run geometry fails closed."""
    kwargs = _ready_kwargs()
    kwargs["dt"] = 0.0
    report = check_campaign_readiness(_scenario_set(tmp_path), [101], **kwargs)

    assert report["ready"] is False
    assert "run_geometry_positive" in report["failed_checks"]


def test_run_campaign_fails_closed_on_bad_inputs(tmp_path) -> None:
    """run_campaign aborts (rolling no episodes) when the readiness gate fails."""
    out_dir = tmp_path / "out"
    with pytest.raises(RuntimeError, match="readiness gate failed"):
        run_campaign(
            _scenario_set(tmp_path),
            [101, 101],  # duplicate -> not ready
            out_dir,
            **_ready_kwargs(),
        )
    # Fail-closed means nothing was written.
    assert not out_dir.exists()


def test_main_preflight_only_exit_codes(tmp_path) -> None:
    """--preflight-only returns 0 when ready and 1 when a check fails; rolls no episodes."""
    scenario_set = _scenario_set(tmp_path)
    ready_code = main(
        ["--preflight-only", "--scenario-set", str(scenario_set), "--seeds", "101", "102"]
    )
    assert ready_code == 0

    not_ready_code = main(
        [
            "--preflight-only",
            "--scenario-set",
            str(tmp_path / "absent.yaml"),
            "--seeds",
            "101",
            "102",
        ]
    )
    assert not_ready_code == 1
