"""Tests for the reactivity-vs-replay rank-study preflight (issue #3637).

Covers the pure checker, the packet loader, the canonical shipped launch packet (regression guard),
and the thin CLI. No benchmark execution, no rank-stability interpretation.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from robot_sf.benchmark.reactivity_ablation import REACTIVITY_ARMS, REPLAY_LIMITATION
from robot_sf.benchmark.reactivity_replay_preflight import (
    DIAGNOSTIC_SEED_COUNT,
    MIN_RANK_STABILITY_SEEDS,
    PREFLIGHT_SCHEMA,
    ReactivityReplayRunPlan,
    build_preflight_manifest,
    check_run_plan,
    run_plan_from_packet,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SHIPPED_PACKET = (
    REPO_ROOT / "configs/benchmarks/reactivity_replay_rank_study_issue_3637_launch_packet.yaml"
)

# scripts/benchmark/ has no package __init__; load the CLI module by path (repo convention).
_CLI_PATH = (
    REPO_ROOT / "scripts" / "benchmark" / "preflight_reactivity_replay_rank_study_issue_3637.py"
)
_SPEC = importlib.util.spec_from_file_location("_issue_3637_preflight_cli", _CLI_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_CLI = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CLI)
cli_main = _CLI.main


def _ready_seeds() -> tuple[int, ...]:
    """A paired seed set that meets the rank-stability floor."""
    return tuple(range(101, 101 + MIN_RANK_STABILITY_SEEDS))


def _ready_plan(**overrides) -> ReactivityReplayRunPlan:
    """A plan that passes every precondition unless overridden."""
    seeds = _ready_seeds()
    base = {
        "planners": ("goal", "orca", "social_force"),
        "arm_seeds": dict.fromkeys(REACTIVITY_ARMS, seeds),
        "scenario_set": "configs/scenarios/sets/classic_crossing_subset.yaml",
        "horizon": 300,
    }
    base.update(overrides)
    return ReactivityReplayRunPlan(**base)


def _checks_by_name(plan: ReactivityReplayRunPlan) -> dict[str, bool]:
    return {c.name: c.passed for c in check_run_plan(plan)}


def test_ready_plan_passes_and_manifest_is_ready():
    """A well-formed plan passes all checks and yields a ready manifest."""
    manifest = build_preflight_manifest(_ready_plan())
    assert manifest["status"] == "ready"
    assert manifest["schema_version"] == PREFLIGHT_SCHEMA
    assert manifest["issue"] == 3637
    assert manifest["blocking_issues"] == []
    assert all(c["passed"] for c in manifest["checks"])


def test_manifest_always_carries_replay_limitation():
    """The replay limitation travels with the manifest, even when blocked."""
    manifest = build_preflight_manifest(_ready_plan(planners=("goal",)))
    assert manifest["status"] == "blocked"
    limitation = manifest["replay_limitation"]
    assert limitation["is_trajectory_playback"] is False
    assert "not pre-recorded trajectory playback" in limitation["note"].lower()
    assert "rank stability" in manifest["claim_boundary"].lower()


def test_too_few_planners_blocks():
    """Fewer than three planners fails the planner-count check."""
    assert _checks_by_name(_ready_plan(planners=("goal", "orca")))["planner_count"] is False


def test_duplicate_planners_do_not_inflate_count():
    """Distinct-count, not raw length, drives the planner check."""
    plan = _ready_plan(planners=("goal", "goal", "goal"))
    assert _checks_by_name(plan)["planner_count"] is False


def test_unpaired_seeds_block():
    """Differing seed sets across arms fail the paired-seeds check."""
    seeds = _ready_seeds()
    plan = _ready_plan(
        arm_seeds={REACTIVITY_ARMS[0]: seeds, REACTIVITY_ARMS[1]: seeds[:-1] + (999,)}
    )
    checks = _checks_by_name(plan)
    assert checks["paired_seeds"] is False


def test_missing_arm_blocks_arms_and_pairing():
    """A plan missing the replay arm fails both the arms and pairing checks."""
    plan = _ready_plan(arm_seeds={REACTIVITY_ARMS[0]: _ready_seeds()})
    checks = _checks_by_name(plan)
    assert checks["arms_present"] is False
    assert checks["paired_seeds"] is False


def test_seed_budget_floor_enforced():
    """A paired seed set below the floor (but above the diagnostic) still blocks on budget."""
    small = tuple(range(101, 101 + DIAGNOSTIC_SEED_COUNT + 1))  # 5 seeds: > diagnostic, < floor
    plan = _ready_plan(arm_seeds=dict.fromkeys(REACTIVITY_ARMS, small))
    checks = _checks_by_name(plan)
    assert checks["paired_seeds"] is True
    assert checks["seed_budget"] is False


def test_trajectory_playback_blocks():
    """Declaring replay as trajectory playback is a hard block (mislabeled mechanism)."""
    plan = _ready_plan(replay_is_trajectory_playback=True)
    assert _checks_by_name(plan)["replay_limitation"] is False


def test_missing_limitation_note_blocks():
    """An empty limitation note blocks even when the playback flag is correct."""
    plan = _ready_plan(replay_limitation="   ")
    assert _checks_by_name(plan)["replay_limitation"] is False


def test_short_horizon_blocks():
    """A horizon below the contrast-registration floor blocks."""
    assert _checks_by_name(_ready_plan(horizon=50))["horizon"] is False


def test_run_plan_from_packet_shared_seeds_are_paired():
    """A shared 'seeds' key fans out to identical paired arms."""
    packet = {
        "planners": ["goal", "orca", "social_force"],
        "scenario_set": "configs/scenarios/sets/classic_crossing_subset.yaml",
        "horizon": 300,
        "seeds": list(range(101, 121)),
    }
    plan = run_plan_from_packet(packet)
    assert set(plan.arm_seeds) == set(REACTIVITY_ARMS)
    assert plan.arm_seeds[REACTIVITY_ARMS[0]] == plan.arm_seeds[REACTIVITY_ARMS[1]]
    assert plan.replay_limitation == REPLAY_LIMITATION
    assert build_preflight_manifest(plan)["status"] == "ready"


def test_run_plan_from_packet_explicit_arm_seeds():
    """Explicit per-arm seeds are honored verbatim."""
    seeds = list(range(101, 121))
    packet = {
        "planners": ["goal", "orca", "social_force"],
        "scenario_set": "configs/scenarios/sets/classic_crossing_subset.yaml",
        "horizon": 300,
        "arm_seeds": {"reactive": seeds, "replay": seeds},
    }
    plan = run_plan_from_packet(packet)
    assert plan.arm_seeds["reactive"] == tuple(seeds)


@pytest.mark.parametrize(
    "packet",
    [
        {"scenario_set": "x", "horizon": 1, "seeds": [1]},  # missing planners
        {"planners": ["a"], "horizon": 1, "seeds": [1]},  # missing scenario_set
        {"planners": ["a"], "scenario_set": "x", "seeds": [1]},  # missing horizon
        {"planners": ["a"], "scenario_set": "x", "horizon": 1},  # missing seeds/arm_seeds
        {"planners": ["a"], "scenario_set": "x", "horizon": 1, "seeds": ["bad"]},  # bad seeds
        {"planners": ["a"], "scenario_set": "x", "horizon": True, "seeds": [1]},  # bool horizon
    ],
)
def test_run_plan_from_packet_rejects_malformed(packet):
    """Malformed packets raise ValueError rather than silently passing."""
    with pytest.raises(ValueError):
        run_plan_from_packet(packet)


def test_shipped_packet_preflights_ready():
    """Regression guard: the canonical shipped launch packet must preflight as ready."""
    packet = yaml.safe_load(SHIPPED_PACKET.read_text(encoding="utf-8"))
    manifest = build_preflight_manifest(run_plan_from_packet(packet))
    assert manifest["status"] == "ready", manifest["blocking_issues"]


def test_scenario_set_checksum_mismatch_blocks(tmp_path):
    """A supplied scenario-set checksum fails closed when the file drifts."""
    scenario_set = tmp_path / "scenario_set.yaml"
    scenario_set.write_text("scenarios: []\n", encoding="utf-8")

    manifest = build_preflight_manifest(
        _ready_plan(scenario_set=str(scenario_set), scenario_set_sha256="0" * 64)
    )

    assert manifest["status"] == "blocked"
    assert any(issue.startswith("scenario_set_sha256:") for issue in manifest["blocking_issues"])


def test_scenario_set_checksum_match_passes(tmp_path):
    """A supplied scenario-set checksum passes when it matches the named file."""
    scenario_set = tmp_path / "scenario_set.yaml"
    scenario_set.write_text("scenarios: []\n", encoding="utf-8")
    digest = hashlib.sha256(scenario_set.read_bytes()).hexdigest()

    checks = check_run_plan(_ready_plan(scenario_set=str(scenario_set), scenario_set_sha256=digest))

    assert {c.name: c.passed for c in checks}["scenario_set_sha256"] is True


def test_shipped_packet_checksum_matches_scenario_set():
    """The packet's recorded scenario-set sha256 must match the file (drift guard)."""
    import hashlib

    packet = yaml.safe_load(SHIPPED_PACKET.read_text(encoding="utf-8"))
    scenario_path = REPO_ROOT / packet["scenario_set"]
    actual = hashlib.sha256(scenario_path.read_bytes()).hexdigest()
    assert actual == packet["scenario_set_sha256"]


def test_cli_ready_packet_exit_zero(tmp_path):
    """The CLI returns 0 and writes a manifest for the ready shipped packet."""
    out = tmp_path / "preflight.json"
    code = cli_main(["--packet", str(SHIPPED_PACKET), "--output-json", str(out), "--json"])
    assert code == 0
    manifest = json.loads(out.read_text(encoding="utf-8"))
    assert manifest["status"] == "ready"
    assert manifest["provenance"]["packet"] == str(SHIPPED_PACKET)


def test_cli_blocked_packet_exit_one(tmp_path):
    """A packet that violates a precondition makes the CLI exit 1."""
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        yaml.safe_dump(
            {
                "planners": ["goal", "orca"],  # only 2 planners
                "scenario_set": "configs/scenarios/sets/classic_crossing_subset.yaml",
                "horizon": 300,
                "seeds": list(range(101, 121)),
            }
        ),
        encoding="utf-8",
    )
    assert cli_main(["--packet", str(bad)]) == 1


def test_cli_unreadable_packet_exit_two(tmp_path):
    """A missing packet path makes the CLI exit 2 (usage/IO error)."""
    assert cli_main(["--packet", str(tmp_path / "does_not_exist.yaml")]) == 2
