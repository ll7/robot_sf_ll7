"""Tests for the issue #9545 narrow-doorway crash-vs-wait diagnostic."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from scripts.analysis.narrow_doorway_crash_vs_wait_issue_9545 import (
    FINAL_STAGE_WEIGHTS,
    _discounted_return,
    build_binding,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = (
    REPO_ROOT / "docs" / "context" / "evidence" / "issue_9545_narrow_doorway_crash_vs_wait"
)


def test_bound_weights_match_training_base_config() -> None:
    """Bound eval weights equal the training base final-stage weights."""
    base = yaml.safe_load(
        (
            REPO_ROOT
            / "configs/training/ppo/ablations/expert_ppo_issue_791_reward_curriculum_promotion_10m_env22_eval_aligned_large_capacity_base.yaml"
        ).read_text(encoding="utf-8")
    )
    final_stage = base["env_factory_kwargs"]["reward_curriculum"]["stages"][-1]["reward_kwargs"][
        "weights"
    ]
    assert dict(final_stage) == dict(FINAL_STAGE_WEIGHTS)
    assert base["env_factory_kwargs"]["reward_name"] == "route_completion_v3"


def test_scenario_cap_is_400_not_600() -> None:
    """The canonical scenario declares a 400-step cap."""
    scenarios = yaml.safe_load(
        (REPO_ROOT / "configs/scenarios/single/francis2023_narrow_doorway.yaml").read_text(
            encoding="utf-8"
        )
    )["scenarios"]
    scenario = next(s for s in scenarios if s["name"] == "francis2023_narrow_doorway")
    assert int(scenario["simulation_config"]["max_episode_steps"]) == 400
    assert sorted(scenario["seeds"]) == [225, 226, 227]


def test_discounted_return_orders_crash_below_wait() -> None:
    """Crash-vs-wait arithmetic is monotone in the collision penalty."""
    wait = [-0.015] * 11
    crash = [0.05] * 10 + [-15.0]
    assert _discounted_return(crash, 0.99) < _discounted_return(wait, 0.99)


def test_binding_records_gamma_provenance_gap(tmp_path: Path) -> None:
    """Binding states the training-gamma provenance explicitly."""
    binding = build_binding(tmp_path, 0.99)
    assert binding["scenario_cap_steps"] == 400
    assert binding["checkpoint_gamma_embedded"] == 0.99
    assert binding["gamma_training_config_declared"] is None
    assert "not proof of the training-time objective" in binding["gamma_provenance_note"]


def test_committed_return_table_prefers_wait() -> None:
    """Committed counterfactual table shows wait dominating crash on all seeds."""
    with (EVIDENCE_DIR / "return_table.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 6
    for row in rows:
        if row["mode"] == "hold_stop":
            assert row["ends_in_contact"] == "False"
    crashes = {r["seed"]: float(r["discounted_return"]) for r in rows if "policy" in r["mode"]}
    waits = {r["seed"]: float(r["discounted_return"]) for r in rows if "hold" in r["mode"]}
    assert set(crashes) == {"225", "226", "227"} == set(waits)
    for seed in crashes:
        assert crashes[seed] < waits[seed]


def test_committed_sensitivity_never_prefers_crash() -> None:
    """Committed sensitivity sweep has no crash-preferring cell."""
    with (EVIDENCE_DIR / "sensitivity.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 27
    assert all(row["prefers_crash"] == "False" for row in rows)


def test_committed_traces_end_in_wall_contact() -> None:
    """Committed per-step traces record obstacle contact on all seeds."""
    for seed in (225, 226, 227):
        rows = [
            json.loads(line)
            for line in (EVIDENCE_DIR / f"trace_seed{seed}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert rows[-1]["is_obstacle_collision"] is True
        assert rows[-1]["terminated"] is True
        assert len(rows) < 400
        contact_terms = rows[-1]["reward_terms"]
        assert contact_terms["collision"] == -15.0
