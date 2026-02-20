"""Smoke tests for predictive planner evaluation CLI."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.validation import evaluate_predictive_planner as mod


def test_evaluate_predictive_planner_smoke(monkeypatch, tmp_path: Path) -> None:
    """Evaluation CLI should run with mocked map runner and produce summary artifacts."""
    checkpoint = tmp_path / "predictive_model.pt"
    checkpoint.write_text("stub", encoding="utf-8")
    scenario_matrix = tmp_path / "scenarios.yaml"
    scenario_matrix.write_text("scenarios: []\n", encoding="utf-8")
    output_dir = tmp_path / "eval_out"

    def _fake_run_map_batch(*args, **kwargs):
        jsonl_path = Path(args[1])
        row = {
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
        ),
    )

    code = mod.main()
    assert code == 0
    assert (output_dir / "smoke_summary.json").exists()
    assert (output_dir / "smoke.jsonl").exists()
