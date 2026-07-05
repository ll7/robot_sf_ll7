"""Tests issue #4013 learned-prediction MPC comparison report."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import yaml

from scripts.analysis.compare_model_based_planning_issue_4013 import (
    EXPECTED_CLAIM_BOUNDARY,
    SCHEMA_VERSION,
    build_report_from_config,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_build_report_requires_three_paired_nonfallback_roles(tmp_path: Path) -> None:
    """Matched learned, CV, and model-free rows produce a diagnostic-ready report."""
    learned = _write_episodes(
        tmp_path / "learned.jsonl",
        [_episode("learned_prediction_mpc", "learned_prediction_mpc", success=True)],
    )
    cv = _write_episodes(
        tmp_path / "cv.jsonl",
        [_episode("cv_prediction_mpc", "cv_prediction_mpc", success=True)],
    )
    model_free = _write_episodes(
        tmp_path / "goal.jsonl",
        [_episode("model_free_baseline", "goal", success=False)],
    )
    config = _write_config(
        tmp_path,
        runs={
            "learned_prediction_mpc": learned,
            "cv_prediction_mpc": cv,
            "model_free_baseline": model_free,
        },
    )

    report = build_report_from_config(config)

    assert report["status"] == "diagnostic_ready"
    assert report["paired_seed_count"] == 1
    assert report["fallback_degraded_rows"] == {
        "excluded": 0,
        "included_as_non_evidence": 0,
    }
    assert report["roles"]["learned_prediction_mpc"]["success_rate"] == pytest.approx(1.0)
    assert report["roles"]["model_free_baseline"]["success_rate"] == pytest.approx(0.0)
    assert all(item["met"] for item in report["closure_criteria"])
    assert json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))["status"] == (
        "diagnostic_ready"
    )
    assert "not paper-grade benchmark evidence" in (tmp_path / "report.md").read_text(
        encoding="utf-8"
    )


def test_report_blocks_when_model_free_baseline_missing(tmp_path: Path) -> None:
    """The closure comparator cannot pass with only learned and CV prediction-MPC arms."""
    learned = _write_episodes(
        tmp_path / "learned.jsonl",
        [_episode("learned_prediction_mpc", "learned_prediction_mpc")],
    )
    cv = _write_episodes(
        tmp_path / "cv.jsonl",
        [_episode("cv_prediction_mpc", "cv_prediction_mpc")],
    )
    config = _write_config(
        tmp_path,
        runs={"learned_prediction_mpc": learned, "cv_prediction_mpc": cv},
    )

    report = build_report_from_config(config)

    assert report["status"] == "diagnostic_blocked"
    assert "missing role: model_free_baseline" in report["blockers"]
    assert report["paired_seed_count"] == 0


def test_report_excludes_fallback_rows_and_blocks_claim(tmp_path: Path) -> None:
    """Fallback or degraded rows are counted as non-evidence, not success."""
    learned = _write_episodes(
        tmp_path / "learned.jsonl",
        [
            _episode(
                "learned_prediction_mpc",
                "learned_prediction_mpc",
                metadata_status="fallback",
            )
        ],
    )
    cv = _write_episodes(
        tmp_path / "cv.jsonl",
        [_episode("cv_prediction_mpc", "cv_prediction_mpc")],
    )
    model_free = _write_episodes(
        tmp_path / "goal.jsonl",
        [_episode("model_free_baseline", "goal")],
    )
    config = _write_config(
        tmp_path,
        runs={
            "learned_prediction_mpc": learned,
            "cv_prediction_mpc": cv,
            "model_free_baseline": model_free,
        },
    )

    report = build_report_from_config(config)

    assert report["status"] == "diagnostic_blocked"
    assert report["fallback_degraded_rows"]["excluded"] == 1
    assert "learned_prediction_mpc has no non-fallback evidence rows" in report["blockers"]
    assert "learned_prediction_mpc has fallback/degraded rows excluded" in report["blockers"]
    assert report["roles"]["learned_prediction_mpc"]["evidence_episodes"] == 0


def test_report_fails_closed_on_episode_missing_scenario_or_seed(tmp_path: Path) -> None:
    """A record missing scenario_id or seed must raise, not silently key on 'None::None'."""
    bad = _episode("learned_prediction_mpc", "learned_prediction_mpc")
    del bad["seed"]
    learned = _write_episodes(tmp_path / "learned.jsonl", [bad])
    cv = _write_episodes(
        tmp_path / "cv.jsonl",
        [_episode("cv_prediction_mpc", "cv_prediction_mpc")],
    )
    model_free = _write_episodes(
        tmp_path / "goal.jsonl",
        [_episode("model_free_baseline", "goal")],
    )
    config = _write_config(
        tmp_path,
        runs={
            "learned_prediction_mpc": learned,
            "cv_prediction_mpc": cv,
            "model_free_baseline": model_free,
        },
    )

    with pytest.raises(ValueError, match="scenario_id"):
        build_report_from_config(config)


def _episode(
    role: str,
    algo: str,
    *,
    success: bool = True,
    seed: int = 4013,
    scenario_id: str = "francis2023_blind_corner",
    metadata_status: str = "ok",
) -> dict[str, object]:
    return {
        "version": "v1",
        "episode_id": f"{role}-{seed}",
        "scenario_id": scenario_id,
        "seed": seed,
        "algo": algo,
        "metrics": {
            "success": success,
            "collisions": 0,
            "near_misses": 0,
            "min_clearance_m": 0.6,
            "time_to_goal_s": 2.5,
        },
        "outcome": {"collision_event": False},
        "termination_reason": "goal_reached" if success else "timeout",
        "integrity": {"ok": True},
        "wall_time_sec": 0.25,
        "algorithm_metadata": {
            "algorithm": algo,
            "status": metadata_status,
            "policy_semantics": role,
        },
    }


def _write_episodes(path: Path, records: list[dict[str, object]]) -> Path:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def _write_config(tmp_path: Path, *, runs: dict[str, Path]) -> Path:
    config = {
        "schema_version": SCHEMA_VERSION,
        "issue": 4013,
        "evidence_tier": "diagnostic-only",
        "claim_boundary": EXPECTED_CLAIM_BOUNDARY,
        "output_json": "report.json",
        "output_markdown": "report.md",
        "runs": [
            {"role": role, "episodes_jsonl": str(path)} for role, path in sorted(runs.items())
        ],
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    return path
