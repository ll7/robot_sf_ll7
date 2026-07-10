"""Smoke tests for predictive planner evaluation CLI."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from scripts.validation import evaluate_predictive_planner as mod


def test_evaluate_predictive_planner_smoke(monkeypatch, tmp_path: Path) -> None:
    """Evaluation CLI should run with mocked map runner and produce summary artifacts."""
    checkpoint = tmp_path / "predictive_model.pt"
    torch.save({}, checkpoint)
    scenario_matrix = tmp_path / "scenarios.yaml"
    scenario_matrix.write_text("scenarios: []\n", encoding="utf-8")
    output_dir = tmp_path / "eval_out"

    def _fake_run_map_batch(*args, **kwargs):
        """Write one successful episode row instead of running map evaluation."""
        jsonl_path = Path(args[1])
        row = {
            "episode_id": "episode_001",
            "scenario_id": "scenario_a",
            "seed": 7,
            "status": "ok",
            "termination_reason": "goal_reached",
            "metrics": {
                "success_rate": 1.0,
                "min_distance": 0.9,
                "avg_speed": 0.4,
                "total_collision_count": 0.0,
            },
        }
        jsonl_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        return {"ok": True}

    monkeypatch.setattr(mod, "run_map_batch", _fake_run_map_batch)
    monkeypatch.setattr(
        mod,
        "parse_args",
        lambda: mod.argparse.Namespace(
            checkpoint=checkpoint,
            scenario_matrix=scenario_matrix,
            schema_path=Path("robot_sf/benchmark/schemas/episode.schema.v1.json"),
            horizon=10,
            dt=0.1,
            workers=1,
            min_success_rate=0.3,
            min_distance=0.25,
            output_dir=output_dir,
            tag="smoke",
            algo_config=None,
            seed_manifest=None,
            comparison_jsonl=None,
            observation_noise=None,
            bootstrap_samples=20,
            bootstrap_seed=123,
            confidence=0.95,
        ),
    )

    code = mod.main()
    assert code == 0
    summary_path = output_dir / "smoke_summary.json"
    assert summary_path.exists()
    assert (output_dir / "smoke.jsonl").exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["contract_version"] == "benchmark-reset-v2"
    assert payload["integrity"]["pass"] is True
    assert payload["collision_metric_status"]["status"] == "available"
    assert payload["uncertainty"]["success_rate_ci"][1] == 1.0


def test_evaluate_predictive_planner_fails_closed_without_collision_metric(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Evaluation quality gates should reject rows without explicit collision metrics."""
    checkpoint = tmp_path / "predictive_model.pt"
    torch.save({}, checkpoint)
    scenario_matrix = tmp_path / "scenarios.yaml"
    scenario_matrix.write_text("scenarios: []\n", encoding="utf-8")
    output_dir = tmp_path / "eval_out"

    def _fake_run_map_batch(*args, **kwargs):
        jsonl_path = Path(args[1])
        row = {
            "episode_id": "episode_001",
            "scenario_id": "scenario_a",
            "seed": 7,
            "status": "ok",
            "termination_reason": "goal_reached",
            "metrics": {
                "success_rate": 1.0,
                "min_distance": 0.9,
                "avg_speed": 0.4,
            },
        }
        jsonl_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        return {"ok": True}

    monkeypatch.setattr(mod, "run_map_batch", _fake_run_map_batch)
    monkeypatch.setattr(
        mod,
        "parse_args",
        lambda: mod.argparse.Namespace(
            checkpoint=checkpoint,
            scenario_matrix=scenario_matrix,
            schema_path=Path("robot_sf/benchmark/schemas/episode.schema.v1.json"),
            horizon=10,
            dt=0.1,
            workers=1,
            min_success_rate=0.3,
            min_distance=0.25,
            output_dir=output_dir,
            tag="smoke",
            algo_config=None,
            seed_manifest=None,
            comparison_jsonl=None,
            observation_noise=None,
            bootstrap_samples=20,
            bootstrap_seed=123,
            confidence=0.95,
        ),
    )

    code = mod.main()

    assert code == 2
    payload = json.loads((output_dir / "smoke_summary.json").read_text(encoding="utf-8"))
    assert payload["collision_metric_status"]["status"] == "not_available"
    assert payload["quality_gates"]["collision_metric_available"] is False
    assert payload["quality_gates"]["pass_all"] is False


def test_uncertainty_helpers_validate_confidence_and_filter_non_finite() -> None:
    """Uncertainty helpers should avoid silent confidence fallback and non-finite samples."""
    assert 1.28 < mod._normal_z(0.80) < 1.29

    try:
        mod._normal_z(1.0)
    except ValueError as exc:
        assert "confidence" in str(exc)
    else:  # pragma: no cover - explicit failure branch
        raise AssertionError("expected invalid confidence to fail")

    interval = mod._bootstrap_mean_interval(
        [1.0, float("nan"), 3.0, float("inf")],
        samples=20,
        confidence=0.95,
        seed=123,
    )
    assert interval[0] is not None
    assert interval[1] is not None
    assert 1.0 <= interval[0] <= interval[1] <= 3.0


def test_integrity_summary_groups_multiple_reasons_per_episode() -> None:
    """Integrity summary should emit one contradiction entry per episode id."""
    rows = [
        {
            "episode_id": "episode_001",
            "termination_reason": "collision",
            "metrics": {"success_rate": 1.0, "total_collision_count": 2.0},
        }
    ]

    summary = mod._integrity_summary(rows)
    assert summary["pass"] is False
    assert summary["contradiction_count"] == 1
    assert summary["contradictions"] == [
        {
            "episode_id": "episode_001",
            "reasons": ["collision_with_success", "success_with_collision_metric"],
        }
    ]
