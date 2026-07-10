"""Tests for the training-run manifest backfill tool."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from robot_sf.common import TrainingRunStatus
from scripts.tools import backfill_training_run_manifest as backfill

if TYPE_CHECKING:
    from pathlib import Path


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def _make_run_tree(root: Path, run_id: str = "ppo_expert_demo_20260622T142053") -> Path:
    """Build a minimal retained-artefact tree mirroring a completed run."""
    bench = root / "benchmarks"
    imitation = bench / "ppo_imitation"

    _write(
        bench / "expert_policies" / "ppo_expert_demo.json",
        {
            "policy_id": "ppo_expert_demo",
            "seeds": [509],
            "scenario_profile": ["classic_doorway_low", "classic_doorway_high"],
            "metrics": {
                "success_rate": {"mean": 0.9, "median": 1.0, "p95": 1.0, "ci95": [0.85, 0.95]},
                "collision_rate": {"mean": 0.1, "median": 0.0, "p95": 1.0},
            },
        },
    )
    _write(
        imitation / "perf" / f"{run_id}.json",
        {"run_id": run_id, "total_wall_clock_sec": 58414.5},
    )
    _write(
        imitation / "eval_by_scenario" / f"{run_id}.json",
        [
            {"scenario_id": "classic_doorway_low", "eval_step": 524288, "episodes": 5},
            {"scenario_id": "classic_doorway_low", "eval_step": 1000000, "episodes": 7},
            {"scenario_id": "classic_doorway_high", "eval_step": 1000000, "episodes": 3},
        ],
    )
    _write(imitation / "eval_timeline" / f"{run_id}.json", [{"eval_step": 1000000}])
    episode_log = imitation / "episodes" / f"{run_id}.jsonl"
    episode_log.parent.mkdir(parents=True, exist_ok=True)
    episode_log.write_text('{"episode": 1}\n', encoding="utf-8")
    return bench


def test_backfill_rebuilds_manifest_from_artifacts(tmp_path: Path) -> None:
    """A completed run with no runs/ manifest is reconstructed from its artefacts."""
    run_id = "ppo_expert_demo_20260622T142053"
    _make_run_tree(tmp_path, run_id)

    written = backfill.backfill(run_dir=tmp_path)

    assert written == tmp_path / "benchmarks" / "ppo_imitation" / "runs" / f"{run_id}.json"
    payload = json.loads(written.read_text(encoding="utf-8"))

    assert payload["run_id"] == run_id
    assert payload["status"] == TrainingRunStatus.COMPLETED.value
    assert payload["seeds"] == [509]
    # Portable relative paths, independent of the running checkout.
    assert payload["episode_log_path"] == f"benchmarks/ppo_imitation/episodes/{run_id}.jsonl"
    assert payload["perf_summary_path"].startswith("benchmarks/ppo_imitation/perf/")
    # wall clock derived from total_wall_clock_sec (58414.5 / 3600).
    assert payload["wall_clock_hours"] == pytest.approx(16.2263, abs=1e-3)
    # Coverage counted at the final eval step only (7 + 3), not summed across checkpoints.
    assert payload["scenario_coverage"] == {"classic_doorway_low": 7, "classic_doorway_high": 3}
    assert payload["metrics"]["success_rate"]["mean"] == 0.9
    assert any("Backfilled" in note for note in payload["notes"])
    assert "/tmp" not in json.dumps(payload) or payload["episode_log_path"].startswith(
        "benchmarks/"
    )


def test_backfill_refuses_to_overwrite_without_force(tmp_path: Path) -> None:
    """An existing manifest is preserved unless --force is given."""
    _make_run_tree(tmp_path)
    backfill.backfill(run_dir=tmp_path)
    with pytest.raises(backfill.BackfillError, match="already exists"):
        backfill.backfill(run_dir=tmp_path)
    # Force overwrites cleanly.
    backfill.backfill(run_dir=tmp_path, force=True)


def test_backfill_dry_run_writes_nothing(tmp_path: Path) -> None:
    """Dry-run reports the target without creating a manifest."""
    run_id = "ppo_expert_demo_20260622T142053"
    _make_run_tree(tmp_path, run_id)
    target = backfill.backfill(run_dir=tmp_path, dry_run=True)
    assert not target.exists()


def test_backfill_rejects_object_eval_by_scenario_artifact(tmp_path: Path) -> None:
    """The retained evaluation artifact must preserve its row-array contract."""
    run_id = "ppo_expert_demo_20260622T142053"
    bench = _make_run_tree(tmp_path, run_id)
    _write(bench / "ppo_imitation" / "eval_by_scenario" / f"{run_id}.json", {})

    with pytest.raises(backfill.BackfillError, match="Expected JSON array"):
        backfill.backfill(run_dir=tmp_path)
