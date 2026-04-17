"""Tests for SNQI calibration analysis CLI."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from scripts.tools import analyze_snqi_calibration

if TYPE_CHECKING:
    from pathlib import Path


def _write_assets(tmp_path: Path) -> tuple[Path, Path]:
    weights = {
        "w_success": 0.20,
        "w_time": 0.10,
        "w_collisions": 0.15,
        "w_near": 0.25,
        "w_comfort": 0.15,
        "w_force_exceed": 0.10,
        "w_jerk": 0.05,
    }
    baseline = {
        "time_to_goal_norm": {"med": 0.5, "p95": 1.0},
        "collisions": {"med": 0.0, "p95": 1.0},
        "near_misses": {"med": 0.0, "p95": 4.0},
        "force_exceed_events": {"med": 0.0, "p95": 6.0},
        "jerk_mean": {"med": 0.05, "p95": 0.5},
    }
    weights_path = tmp_path / "weights.json"
    baseline_path = tmp_path / "baseline.json"
    weights_path.write_text(json.dumps(weights), encoding="utf-8")
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    return weights_path, baseline_path


def _episode(planner: str, success: float, near: float) -> dict[str, object]:
    return {
        "planner_key": planner,
        "kinematics": "differential_drive",
        "metrics": {
            "success": success,
            "time_to_goal_norm": 0.55 if success else 0.95,
            "collisions": 0.0 if success else 1.0,
            "near_misses": near,
            "comfort_exposure": 0.05 + near * 0.02,
            "force_exceed_events": near,
            "jerk_mean": 0.06 + near * 0.03,
        },
    }


def test_analyze_snqi_calibration_cli_writes_outputs(tmp_path: Path, capsys) -> None:
    """CLI should write reproducible JSON/Markdown/CSV calibration artifacts."""
    weights_path, baseline_path = _write_assets(tmp_path)
    episodes_path = tmp_path / "episodes.jsonl"
    episodes = [
        _episode("safe", 1.0, 0.0),
        _episode("safe", 1.0, 1.0),
        _episode("risky", 0.0, 3.0),
        _episode("risky", 1.0, 4.0),
    ]
    episodes_path.write_text(
        "\n".join(json.dumps(row) for row in episodes) + "\n",
        encoding="utf-8",
    )

    output_json = tmp_path / "out" / "calibration.json"
    output_md = tmp_path / "out" / "calibration.md"
    output_csv = tmp_path / "out" / "calibration.csv"
    exit_code = analyze_snqi_calibration.main(
        [
            "--episodes",
            str(episodes_path),
            "--weights",
            str(weights_path),
            "--baseline",
            str(baseline_path),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--output-csv",
            str(output_csv),
        ]
    )

    assert exit_code == 0
    stdout = json.loads(capsys.readouterr().out)
    assert stdout["snqi_calibration_json"] == str(output_json)
    assert stdout["recommendation"] in {
        "keep_v3_fixed",
        "demote_snqi_further",
        "propose_candidate_v4_contract",
    }
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "snqi-calibration-analysis.v1"
    assert payload["input"]["kind"] == "episodes_jsonl"
    assert len(payload["input"]["weights_sha256"]) == 64
    assert output_md.exists()
    assert (
        "planner_rank_correlation_vs_v3" in output_csv.read_text(encoding="utf-8").splitlines()[0]
    )


def test_analyze_snqi_calibration_cli_rejects_bad_epsilon(tmp_path: Path) -> None:
    """CLI should reject perturbation fractions outside the local-neighborhood range."""
    weights_path, baseline_path = _write_assets(tmp_path)
    episodes_path = tmp_path / "episodes.jsonl"
    episodes_path.write_text(json.dumps(_episode("safe", 1.0, 0.0)) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="epsilon"):
        analyze_snqi_calibration.main(
            [
                "--episodes",
                str(episodes_path),
                "--weights",
                str(weights_path),
                "--baseline",
                str(baseline_path),
                "--epsilon",
                "1.2",
                "--output-json",
                str(tmp_path / "out.json"),
                "--output-md",
                str(tmp_path / "out.md"),
                "--output-csv",
                str(tmp_path / "out.csv"),
            ]
        )
