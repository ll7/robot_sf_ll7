"""TODO docstring. Document this module."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from robot_sf.benchmark.algorithm_metadata import enrich_algorithm_metadata
from robot_sf.benchmark.runner import run_batch

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA_PATH = "robot_sf/benchmark/schemas/episode.schema.v1.json"


def test_algorithm_metadata_present(tmp_path: Path):
    """Smoke-test that benchmark episode records include enriched algorithm metadata."""
    matrix = [
        {
            "id": "algo-meta-smoke",
            "density": "low",
            "flow": "uni",
            "obstacle": "open",
            "groups": 0.0,
            "speed_var": "low",
            "goal_topology": "point",
            "robot_context": "embedded",
            "repeats": 1,
        },
    ]
    out_jsonl = tmp_path / "episodes.jsonl"
    summary = run_batch(
        matrix,
        out_path=out_jsonl,
        schema_path=SCHEMA_PATH,
        base_seed=0,
        horizon=5,
        dt=0.1,
        record_forces=False,
        append=False,
    )
    assert summary["written"] == 1
    content = out_jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(content) == 1
    rec = json.loads(content[0])
    assert "algorithm_metadata" in rec
    algo_md = rec["algorithm_metadata"]
    assert isinstance(algo_md, dict)
    assert "algorithm" in algo_md
    assert algo_md["baseline_category"] == "classical"
    assert algo_md["policy_semantics"] == "deterministic_goal_seeking"
    planner_meta = algo_md.get("planner_kinematics")
    assert isinstance(planner_meta, dict)
    assert planner_meta.get("execution_mode") == "native"


def test_prediction_mpc_registry_declared_adapter_only():
    """Guard that prediction_mpc is registry-declared adapter-only by design.

    Issue #6828 (Option A) established that the #5579 "native execution"
    canary requirement for a registry-declared adapter-only algorithm is
    satisfied by a fail-closed ``canary.solver_execution`` contract, NOT by
    ``planner_kinematics.execution_mode == "native"``. ``execution_mode`` is a
    command-space concept (native robot commands vs adapter-projected), not a
    record of whether the MPC solver ran.

    The canonical ``prediction_mpc`` planner is adapter-only by design: it
    emits unicycle_vw commands through ``PredictionMPCPlannerAdapter`` and has
    no native command path. A gate bound to ``execution_mode == "native"`` is
    unsatisfiable by construction for this planner, so this test pins the
    registry declaration so a future reader cannot mistake it for a staging
    defect and re-raise it as a review blocker. See issue #6828 and the
    "Native Execution Criterion For The #5579 Canary" section of
    ``docs/code_review.md``.
    """
    enriched = enrich_algorithm_metadata(algo="prediction_mpc")
    planner_kinematics = enriched["planner_kinematics"]
    assert planner_kinematics["supports_native_commands"] is False
    assert planner_kinematics["execution_mode"] == "adapter"
    assert planner_kinematics["adapter_name"] == "PredictionMPCPlannerAdapter"
