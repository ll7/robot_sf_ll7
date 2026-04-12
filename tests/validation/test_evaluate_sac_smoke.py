"""Smoke tests for SAC evaluation CLI (issue #790)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.validation import evaluate_sac as mod


def _fake_run_map_batch(*args, **kwargs):
    """Write one success episode record and return a minimal summary."""
    jsonl_path = Path(args[1])
    row = {
        "episode_id": "ep_001",
        "scenario_id": "scenario_a",
        "seed": 42,
        "status": "ok",
        "termination_reason": "goal_reached",
        "metrics": {
            "success_rate": 1.0,
            "min_distance": 0.8,
            "avg_speed": 0.5,
        },
    }
    jsonl_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return {"ok": True}


def test_evaluate_sac_smoke(monkeypatch, tmp_path: Path) -> None:
    """Evaluation CLI should run with mocked map runner and produce summary artifacts."""
    checkpoint = tmp_path / "sac_model.zip"
    checkpoint.write_text("stub", encoding="utf-8")
    scenario_matrix = tmp_path / "scenarios.yaml"
    scenario_matrix.write_text("scenarios: []\n", encoding="utf-8")
    output_dir = tmp_path / "eval_out"

    # Write a minimal algo config template so the script can read it.
    algo_cfg = tmp_path / "algo.yaml"
    algo_cfg.write_text("model_path: stub\ndevice: auto\nobs_mode: dict\n", encoding="utf-8")

    monkeypatch.setattr(mod, "run_map_batch", _fake_run_map_batch)
    monkeypatch.setattr(
        mod,
        "parse_args",
        lambda: argparse.Namespace(
            checkpoint=checkpoint,
            scenario_matrix=scenario_matrix,
            algo_config=algo_cfg,
            schema_path=Path("robot_sf/benchmark/schemas/episode.schema.v1.json"),
            horizon=10,
            dt=0.1,
            workers=1,
            min_success_rate=0.3,
            output_dir=output_dir,
            tag="smoke",
            device="cpu",
        ),
    )

    code = mod.main()
    assert code == 0
    summary_path = output_dir / "smoke_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["success_rate"] == 1.0
    assert summary["total_episodes"] == 1
    assert summary["device"] == "cpu"


def test_evaluate_sac_quality_gate_failure(monkeypatch, tmp_path: Path) -> None:
    """Quality gate should return exit code 1 when success rate is below threshold."""
    checkpoint = tmp_path / "sac_model.zip"
    checkpoint.write_text("stub", encoding="utf-8")
    scenario_matrix = tmp_path / "scenarios.yaml"
    scenario_matrix.write_text("scenarios: []\n", encoding="utf-8")
    output_dir = tmp_path / "eval_out"
    algo_cfg = tmp_path / "algo.yaml"
    algo_cfg.write_text("model_path: stub\ndevice: auto\nobs_mode: dict\n", encoding="utf-8")

    def _fake_fail_batch(*args, **kwargs):
        jsonl_path = Path(args[1])
        row = {
            "episode_id": "ep_001",
            "scenario_id": "scenario_a",
            "seed": 1,
            "status": "collision",
            "termination_reason": "collision",
            "metrics": {"success_rate": 0.0, "min_distance": 0.0, "avg_speed": 0.2},
        }
        jsonl_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        return {"ok": True}

    monkeypatch.setattr(mod, "run_map_batch", _fake_fail_batch)
    monkeypatch.setattr(
        mod,
        "parse_args",
        lambda: argparse.Namespace(
            checkpoint=checkpoint,
            scenario_matrix=scenario_matrix,
            algo_config=algo_cfg,
            schema_path=Path("robot_sf/benchmark/schemas/episode.schema.v1.json"),
            horizon=10,
            dt=0.1,
            workers=1,
            min_success_rate=0.5,
            output_dir=output_dir,
            tag="smoke",
            device="cpu",
        ),
    )

    code = mod.main()
    assert code == 1


def test_evaluate_sac_arg_parser_requires_checkpoint() -> None:
    """CLI must require --checkpoint."""
    import sys as _sys

    saved = _sys.argv
    _sys.argv = ["evaluate_sac.py"]
    try:
        raised = False
        try:
            mod.parse_args()
        except SystemExit:
            raised = True
        assert raised, "--checkpoint should be required"
    finally:
        _sys.argv = saved
