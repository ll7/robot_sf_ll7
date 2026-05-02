"""Tests for camera-ready campaign comparison helper."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

import pytest

from scripts.tools.compare_camera_ready_campaigns import (
    _build_markdown,
    _resolve_safe_output_path,
    compare_campaigns,
    main,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_compare_campaigns_reports_prediction_success_gain(tmp_path: Path) -> None:
    """Comparison should capture metric deltas between base and candidate planner rows."""
    base_root = tmp_path / "base_campaign"
    candidate_root = tmp_path / "candidate_campaign"
    _write_summary(
        base_root / "reports" / "campaign_summary.json",
        {
            "campaign": {"campaign_id": "base"},
            "planner_rows": [
                {
                    "planner_key": "prediction_planner",
                    "status": "partial-failure",
                    "episodes": 0,
                    "success_mean": "0.0",
                    "collisions_mean": "0.0",
                }
            ],
        },
    )
    _write_summary(
        candidate_root / "reports" / "campaign_summary.json",
        {
            "campaign": {"campaign_id": "candidate"},
            "planner_rows": [
                {
                    "planner_key": "prediction_planner",
                    "status": "ok",
                    "episodes": 135,
                    "success_mean": "0.9778",
                    "collisions_mean": "0.0000",
                }
            ],
        },
    )

    payload = compare_campaigns(base_root, candidate_root)
    planner = payload["planner_deltas"][0]
    assert planner["planner_key"] == "prediction_planner"
    assert planner["base_status"] == "partial-failure"
    assert planner["candidate_status"] == "ok"
    assert planner["base_episodes"] == 0
    assert planner["candidate_episodes"] == 135
    assert planner["metrics"]["success_mean"]["delta"] > 0.9
    assert payload["reproducibility"]["status"] == "drift_detected"


def test_compare_campaigns_reports_missing_planners(tmp_path: Path) -> None:
    """Comparison should explicitly report planners present in only one campaign."""
    base_root = tmp_path / "base_campaign"
    candidate_root = tmp_path / "candidate_campaign"
    _write_summary(
        base_root / "reports" / "campaign_summary.json",
        {
            "campaign": {"campaign_id": "base"},
            "planner_rows": [
                {"planner_key": "goal", "status": "ok", "episodes": 10, "success_mean": "1.0"},
                {"planner_key": "orca", "status": "ok", "episodes": 10, "success_mean": "1.0"},
            ],
        },
    )
    _write_summary(
        candidate_root / "reports" / "campaign_summary.json",
        {
            "campaign": {"campaign_id": "candidate"},
            "planner_rows": [
                {"planner_key": "goal", "status": "ok", "episodes": 10, "success_mean": "1.0"},
                {
                    "planner_key": "prediction_planner",
                    "status": "ok",
                    "episodes": 10,
                    "success_mean": "1.0",
                },
            ],
        },
    )

    payload = compare_campaigns(base_root, candidate_root)
    assert payload["missing_in_base"] == ["prediction_planner"]
    assert payload["missing_in_candidate"] == ["orca"]


def test_resolve_safe_output_path_rejects_escape(tmp_path: Path) -> None:
    """Output path validation should reject writes outside safe root."""
    safe_root = tmp_path / "safe"
    safe_root.mkdir()
    inside = _resolve_safe_output_path(safe_root / "reports" / "ok.json", safe_root)
    assert inside.is_relative_to(safe_root)
    with pytest.raises(ValueError, match="Unsafe output path"):
        _resolve_safe_output_path(tmp_path / ".." / "outside.json", safe_root)


def test_compare_campaigns_filters_non_json_float_values(tmp_path: Path) -> None:
    """Infinite and NaN metrics should be excluded from planner metric deltas."""
    base_root = tmp_path / "base_campaign"
    candidate_root = tmp_path / "candidate_campaign"
    _write_summary(
        base_root / "reports" / "campaign_summary.json",
        {
            "campaign": {"campaign_id": "base"},
            "planner_rows": [
                {
                    "planner_key": "goal",
                    "status": "ok",
                    "episodes": 10,
                    "success_mean": "inf",
                    "collisions_mean": "nan",
                }
            ],
        },
    )
    _write_summary(
        candidate_root / "reports" / "campaign_summary.json",
        {
            "campaign": {"campaign_id": "candidate"},
            "planner_rows": [
                {
                    "planner_key": "goal",
                    "status": "ok",
                    "episodes": 10,
                    "success_mean": "0.9",
                    "collisions_mean": "0.1",
                }
            ],
        },
    )

    payload = compare_campaigns(base_root, candidate_root)
    assert payload["planner_deltas"][0]["metrics"] == {}


def test_build_markdown_keeps_planners_without_numeric_metrics() -> None:
    """Markdown should still render planners with empty metrics as explicit N/A rows."""
    payload = {
        "base_campaign_id": "base",
        "candidate_campaign_id": "candidate",
        "planner_deltas": [
            {
                "planner_key": "prediction_planner",
                "base_status": "partial-failure",
                "candidate_status": "ok",
                "base_episodes": 0,
                "candidate_episodes": 135,
                "metrics": {},
            }
        ],
        "missing_in_base": [],
        "missing_in_candidate": [],
    }
    markdown = _build_markdown(payload)
    assert "base_status" in markdown
    assert "candidate_status" in markdown
    assert (
        "| prediction_planner | partial-failure | ok | 0 | 135 | no | N/A | N/A | N/A | N/A |"
        in markdown
    )


def test_compare_campaigns_marks_exact_match_reproducible(tmp_path: Path) -> None:
    """Identical planner rows should yield an exact reproducibility verdict."""
    base_root = tmp_path / "base_campaign"
    candidate_root = tmp_path / "candidate_campaign"
    _write_summary(
        base_root / "reports" / "campaign_summary.json",
        {
            "campaign": {"campaign_id": "base"},
            "planner_rows": [
                {
                    "planner_key": "goal",
                    "status": "ok",
                    "episodes": 10,
                    "success_mean": "1.0",
                    "collisions_mean": "0.0",
                    "near_misses_mean": "0.0",
                    "snqi_mean": "-0.1",
                    "time_to_goal_norm_mean": "0.2",
                }
            ],
        },
    )
    _write_summary(
        candidate_root / "reports" / "campaign_summary.json",
        {
            "campaign": {"campaign_id": "candidate"},
            "planner_rows": [
                {
                    "planner_key": "goal",
                    "status": "ok",
                    "episodes": 10,
                    "success_mean": "1.0",
                    "collisions_mean": "0.0",
                    "near_misses_mean": "0.0",
                    "snqi_mean": "-0.1",
                    "time_to_goal_norm_mean": "0.2",
                }
            ],
        },
    )

    payload = compare_campaigns(base_root, candidate_root)
    assert payload["reproducibility"]["status"] == "reproduced"
    assert payload["reproducibility"]["exact_match_planners"] == ["goal"]
    assert payload["planner_deltas"][0]["exact_match"] is True
    markdown = _build_markdown(payload)
    assert "Reproducibility" in markdown
    assert "reproduced" in markdown


def test_main_require_identical_exits_nonzero_on_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI with --require-identical should return 1 when campaigns differ."""
    base_root = tmp_path / "base"
    candidate_root = tmp_path / "candidate"
    _write_summary(
        base_root / "reports" / "campaign_summary.json",
        {
            "campaign": {"campaign_id": "base"},
            "planner_rows": [
                {"planner_key": "goal", "status": "ok", "episodes": 10, "success_mean": "1.0"},
            ],
        },
    )
    _write_summary(
        candidate_root / "reports" / "campaign_summary.json",
        {
            "campaign": {"campaign_id": "candidate"},
            "planner_rows": [
                {"planner_key": "goal", "status": "ok", "episodes": 10, "success_mean": "0.8"},
            ],
        },
    )
    out_json = tmp_path / "out.json"
    out_md = tmp_path / "out.md"
    # Monkeypatch cwd so the safe-root check accepts paths under tmp_path
    monkeypatch.setattr("pathlib.Path.cwd", lambda: tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_camera_ready_campaigns.py",
            "--base-campaign-root",
            str(base_root),
            "--candidate-campaign-root",
            str(candidate_root),
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
            "--require-identical",
        ],
    )
    result = main()
    assert result == 1, "Expected exit code 1 when campaigns differ and --require-identical is set"


def test_main_require_identical_exits_zero_on_exact_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI with --require-identical should return 0 when campaigns are identical."""
    base_root = tmp_path / "base"
    candidate_root = tmp_path / "candidate"
    same_row = {"planner_key": "goal", "status": "ok", "episodes": 10, "success_mean": "1.0"}
    _write_summary(
        base_root / "reports" / "campaign_summary.json",
        {"campaign": {"campaign_id": "base"}, "planner_rows": [same_row]},
    )
    _write_summary(
        candidate_root / "reports" / "campaign_summary.json",
        {"campaign": {"campaign_id": "candidate"}, "planner_rows": [same_row]},
    )
    out_json = tmp_path / "out.json"
    out_md = tmp_path / "out.md"
    # Monkeypatch cwd so the safe-root check accepts paths under tmp_path
    monkeypatch.setattr("pathlib.Path.cwd", lambda: tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_camera_ready_campaigns.py",
            "--base-campaign-root",
            str(base_root),
            "--candidate-campaign-root",
            str(candidate_root),
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
            "--require-identical",
        ],
    )
    result = main()
    assert result == 0, "Expected exit code 0 when campaigns are identical"


def test_main_without_require_identical_exits_zero_on_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI without --require-identical should return 0 even when campaigns differ."""
    base_root = tmp_path / "base"
    candidate_root = tmp_path / "candidate"
    _write_summary(
        base_root / "reports" / "campaign_summary.json",
        {
            "campaign": {"campaign_id": "base"},
            "planner_rows": [
                {"planner_key": "goal", "status": "ok", "episodes": 10, "success_mean": "1.0"},
            ],
        },
    )
    _write_summary(
        candidate_root / "reports" / "campaign_summary.json",
        {
            "campaign": {"campaign_id": "candidate"},
            "planner_rows": [
                {"planner_key": "goal", "status": "ok", "episodes": 10, "success_mean": "0.5"},
            ],
        },
    )
    out_json = tmp_path / "out.json"
    out_md = tmp_path / "out.md"
    # Monkeypatch cwd so the safe-root check accepts paths under tmp_path
    monkeypatch.setattr("pathlib.Path.cwd", lambda: tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_camera_ready_campaigns.py",
            "--base-campaign-root",
            str(base_root),
            "--candidate-campaign-root",
            str(candidate_root),
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
        ],
    )
    result = main()
    assert result == 0, "Expected exit code 0 when --require-identical is not set"
