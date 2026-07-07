"""Tests for issue #4016 measured benchmark-row summarization."""

from __future__ import annotations

import json

import pytest

from scripts.analysis.summarize_distributional_rl_issue_4016_benchmark_rows import (
    summarize_benchmark_rows,
)


def _row(
    *,
    seed: int = 4016,
    risk_objective: str = "mean",
    degraded: bool = False,
    success: bool = False,
    collision_count: float = 0.0,
    near_misses: float = 0.0,
    min_clearance: float = 1.0,
    path_efficiency: float = 0.5,
) -> dict[str, object]:
    return {
        "scenario_id": "classic_cross_trap_low",
        "seed": seed,
        "horizon": 5,
        "row_status": "degraded" if degraded else "native",
        "outcome": {
            "route_complete": success,
            "collision_event": collision_count > 0,
            "timeout_event": not success,
        },
        "metrics": {
            "success": success,
            "total_collision_count": collision_count,
            "near_misses": near_misses,
            "min_clearance": min_clearance,
            "path_efficiency": path_efficiency,
        },
        "algorithm_metadata": {
            "algorithm": "distributional_rl",
            "status": "ok",
            "fallback_or_degraded": degraded,
            "config": {
                "checkpoint_path": "output/models/distributional_rl/issue_4016/qr.pt",
                "risk_alpha": 0.2,
                "risk_objective": risk_objective,
            },
        },
    }


def _write_jsonl(path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_summarize_benchmark_rows_writes_measured_manifests(tmp_path) -> None:
    """Native benchmark rows become measured mean/CVaR comparison manifests."""
    mean_jsonl = tmp_path / "mean.jsonl"
    risk_jsonl = tmp_path / "risk.jsonl"
    _write_jsonl(mean_jsonl, [_row(risk_objective="mean", path_efficiency=0.8)])
    _write_jsonl(
        risk_jsonl,
        [
            _row(risk_objective="cvar_lower", degraded=True),
            _row(risk_objective="cvar_lower", min_clearance=1.2, path_efficiency=0.7),
        ],
    )

    summary = summarize_benchmark_rows(
        mean_jsonl=mean_jsonl,
        risk_jsonl=risk_jsonl,
        output_dir=tmp_path / "evidence",
        write_comparison=True,
    )

    assert summary["benchmark_runner_measured"] is True
    assert summary["included_rows"] == {"mean": 1, "risk": 1}
    assert summary["fallback_degraded_rows"] == {"mean": 0, "risk": 1}
    risk = json.loads((tmp_path / "evidence" / "qr_dqn_cvar_manifest.json").read_text())
    assert risk["benchmark_runner_measured"] is True
    assert risk["metrics"]["mean_min_clearance"] == pytest.approx(1.2)
    comparison = json.loads(
        (tmp_path / "evidence" / "distributional_rl_risk_comparison.json").read_text()
    )
    assert comparison["effect"]["comparison_status"] == "valid_diagnostic"


def test_summarize_benchmark_rows_fails_without_native_rows(tmp_path) -> None:
    """Fallback/degraded-only JSONL must not become closure evidence."""
    mean_jsonl = tmp_path / "mean.jsonl"
    risk_jsonl = tmp_path / "risk.jsonl"
    _write_jsonl(mean_jsonl, [_row(degraded=True)])
    _write_jsonl(risk_jsonl, [_row(risk_objective="cvar_lower")])

    with pytest.raises(ValueError, match="no non-fallback benchmark rows"):
        summarize_benchmark_rows(
            mean_jsonl=mean_jsonl,
            risk_jsonl=risk_jsonl,
            output_dir=tmp_path / "evidence",
        )
