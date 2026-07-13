"""Tests for frozen-state counterfactual replay (issue #5442).

Covers the acceptance criteria: RNG+actor snapshot/restore determinism, baseline
reproduction, versioned output config, computed ``t_inevitable``/``t_uca`` for the
preventable-late-braking, already-unavoidable, and two-action-interaction
fixtures, fail-closed ``unknown`` on nondeterministic baseline or missing feasible
action set, schema conformance, and preservation of every branch result.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from robot_sf.benchmark import last_avoidable_fixtures as fx
from robot_sf.benchmark.last_avoidable_replay import (
    LAST_AVOIDABLE_REPLAY_SCHEMA,
    SUBSTITUTION_HOLD,
    VERDICT_ALREADY_UNAVOIDABLE,
    VERDICT_AVOIDABLE,
    VERDICT_UNKNOWN,
    ReplayConfig,
    locate_last_avoidable,
)

_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "robot_sf"
    / "benchmark"
    / "schemas"
    / "last_avoidable_replay.v1.json"
)
_SCHEMA = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))


def _run(scenario, *, determinism_replays: int = 20):
    """Drive the replay engine over a fixture scenario and return the report."""
    contact_step = fx.find_contact_step(scenario)
    assert contact_step is not None and contact_step >= 1, "baseline must collide"
    horizon = contact_step + 6
    config = ReplayConfig(
        t_danger=0,
        t_contact=contact_step,
        horizon=horizon,
        substitution_mode=SUBSTITUTION_HOLD,
        determinism_replays=determinism_replays,
        action_set_id="decel_lattice",
        feasibility_filter="all_admissible_decel",
        collision_predicate="euclidean_distance<=collision_radius",
        pedestrian_response=scenario.pedestrian_response,
    )
    model = fx.KinematicCollisionModel(scenario)
    baseline = fx.maintain_baseline_actions(contact_step + horizon + 2)
    return locate_last_avoidable(model, baseline, config)


# -- config validation ------------------------------------------------------
def test_config_rejects_non_positive_window() -> None:
    """t_contact must be strictly greater than t_danger."""
    with pytest.raises(ValueError, match="t_contact"):
        ReplayConfig(t_danger=5, t_contact=5, horizon=3)


def test_config_rejects_bad_horizon_and_mode() -> None:
    """Horizon must be >= 1 and substitution_mode must be known."""
    with pytest.raises(ValueError, match="horizon"):
        ReplayConfig(t_danger=0, t_contact=3, horizon=0)
    with pytest.raises(ValueError, match="substitution_mode"):
        ReplayConfig(t_danger=0, t_contact=3, horizon=2, substitution_mode="teleport")


# -- acceptance criterion: RNG + actor state snapshot/restore ---------------
def test_snapshot_includes_rng_and_actor_state() -> None:
    """Restoring a snapshot (RNG captured) reproduces jittered steps bit-for-bit."""
    scenario = fx.KinematicScenario(
        robot_x0=0.0,
        robot_speed0=5.0,
        ped_pos0=(5.0, -1.2),
        ped_vel0=(0.0, 1.0),
        rng_jitter_std=0.5,
        include_rng_in_snapshot=True,
        seed=3,
    )
    model = fx.KinematicCollisionModel(scenario)
    for _ in range(2):
        model.step(0.0)
    snap = model.snapshot()

    def _roll(n: int) -> list[np.ndarray]:
        positions = []
        for _ in range(n):
            model.step(0.0)
            positions.append(model.ped_pos.copy())
        return positions

    first = _roll(4)
    model.restore(snap)
    second = _roll(4)
    for a, b in zip(first, second, strict=True):
        assert np.allclose(a, b), "RNG+actor snapshot/restore must be deterministic"


def test_snapshot_without_rng_diverges() -> None:
    """Omitting the RNG from the snapshot makes a jittered replay nondeterministic."""
    scenario = fx.KinematicScenario(
        robot_x0=0.0,
        robot_speed0=5.0,
        ped_pos0=(5.0, -1.2),
        ped_vel0=(0.0, 1.0),
        rng_jitter_std=0.5,
        include_rng_in_snapshot=False,
        seed=3,
    )
    model = fx.KinematicCollisionModel(scenario)
    snap = model.snapshot()
    model.step(0.0)
    first = model.ped_pos.copy()
    model.restore(snap)
    model.step(0.0)
    second = model.ped_pos.copy()
    assert not np.allclose(first, second), "without RNG capture the replay should diverge"


# -- acceptance criterion: preventable late braking -------------------------
def test_preventable_late_braking_is_avoidable() -> None:
    """Early braking avoids contact; t_uca precedes the point of no return."""
    report = _run(fx.preventable_late_braking_scenario())
    assert report.verdict == VERDICT_AVOIDABLE
    assert report.determinism.deterministic is True
    assert report.t_uca is not None and report.t_inevitable is not None
    assert report.t_uca < report.t_inevitable <= report.config.t_contact
    assert report.minimal_sufficient_interventions, "must record preventing interventions"
    # every branch step in the window is preserved
    assert len(report.branches) == report.config.t_contact - report.config.t_danger


def test_avoidable_records_exact_no_return_point() -> None:
    """The engine's computed t_uca / t_inevitable are stable for the fixture."""
    report = _run(fx.preventable_late_braking_scenario())
    assert report.t_uca == 0
    assert report.t_inevitable == 7


# -- acceptance criterion: already-unavoidable contact ----------------------
def test_already_unavoidable_contact() -> None:
    """Full coverage with no preventing action yields already_unavoidable, not unknown."""
    report = _run(fx.already_unavoidable_scenario())
    assert report.verdict == VERDICT_ALREADY_UNAVOIDABLE
    assert report.determinism.deterministic is True
    assert report.feasible_coverage == pytest.approx(1.0)
    assert report.t_uca is None
    assert report.t_inevitable == report.config.t_danger
    assert report.abstain_reason is None


# -- acceptance criterion: two-action interaction ---------------------------
def test_two_action_interaction_closed_loop_avoidable() -> None:
    """A closed-loop (reactive) pedestrian is a genuine two-body interaction case."""
    scenario = fx.two_action_interaction_scenario()
    assert scenario.pedestrian_response == fx.PED_RESPONSE_CLOSED_LOOP
    report = _run(scenario)
    assert report.verdict == VERDICT_AVOIDABLE
    assert report.config.pedestrian_response == fx.PED_RESPONSE_CLOSED_LOOP
    assert report.t_uca is not None and report.t_inevitable is not None


# -- acceptance criterion: fail-closed unknown, never unavoidable -----------
def test_nondeterministic_baseline_returns_unknown() -> None:
    """A nondeterministic baseline abstains to unknown, never 'unavoidable'."""
    report = _run(fx.nondeterministic_baseline_scenario())
    assert report.verdict == VERDICT_UNKNOWN
    assert report.abstained is True
    assert report.abstain_reason == "nondeterministic_baseline"
    assert report.verdict != "unavoidable"


def test_missing_feasible_action_returns_unknown() -> None:
    """A missing feasible action set abstains to unknown (coverage gap)."""
    report = _run(fx.missing_feasible_action_scenario())
    assert report.verdict == VERDICT_UNKNOWN
    assert report.abstained is True
    assert report.abstain_reason == "incomplete_feasible_action_coverage"
    assert report.feasible_coverage < 1.0
    assert report.verdict != "unavoidable"


# -- acceptance criterion: output contract ----------------------------------
def test_report_conforms_to_schema_and_records_provenance() -> None:
    """Every determination emits a schema-valid, provenance-complete report."""
    for builder in (
        fx.preventable_late_braking_scenario,
        fx.already_unavoidable_scenario,
        fx.two_action_interaction_scenario,
        fx.nondeterministic_baseline_scenario,
        fx.missing_feasible_action_scenario,
    ):
        report = _run(builder())
        payload = report.to_dict()
        jsonschema.validate(payload, _SCHEMA)
        assert payload["schema_version"] == LAST_AVOIDABLE_REPLAY_SCHEMA
        assert payload["normative_fault"] == "not_assessed"
        # versioned analysis config is preserved in output
        cfg = payload["config"]
        assert cfg["action_set_id"] == "decel_lattice"
        assert cfg["collision_predicate"]
        assert cfg["horizon"] >= 1
        assert cfg["pedestrian_response"] in {"replayed", "closed_loop"}


def test_runner_smoke(tmp_path) -> None:
    """The offline CLI runs a fixture and writes a schema-valid report."""
    from scripts.analysis.run_last_avoidable_replay_issue_5442 import main

    exit_code = main(
        [
            "--out-dir",
            str(tmp_path),
            "--fixtures",
            "preventable_late_braking",
            "--determinism-replays",
            "5",
        ]
    )
    assert exit_code == 0
    written = json.loads((tmp_path / "preventable_late_braking.json").read_text())
    jsonschema.validate(written, _SCHEMA)
    assert written["verdict"] == VERDICT_AVOIDABLE
    assert written["runtime_s"] is not None
