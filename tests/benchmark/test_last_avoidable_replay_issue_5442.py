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
    SUBSTITUTION_SINGLE_STEP,
    VERDICT_ALREADY_UNAVOIDABLE,
    VERDICT_AVOIDABLE,
    VERDICT_UNKNOWN,
    ReplayConfig,
    _action_prevents_contact,
    _branch_over_window,
    _capture_window_snapshots,
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
        source_kind="synthetic_fixture",
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


def test_initial_contact_abstains_before_branching() -> None:
    """Contact at the initial snapshot cannot be attributed to a later action."""
    scenario = fx.KinematicScenario(
        robot_x0=0.0,
        robot_speed0=1.0,
        ped_pos0=(0.0, 0.0),
        ped_vel0=(0.0, 0.0),
    )
    report = locate_last_avoidable(
        fx.KinematicCollisionModel(scenario),
        [0.0] * 8,
        ReplayConfig(t_danger=0, t_contact=3, horizon=2, substitution_mode=SUBSTITUTION_HOLD),
    )

    assert report.verdict == VERDICT_UNKNOWN
    assert report.abstained is True
    assert report.abstain_reason == "baseline_initial_contact"
    assert report.t_uca is None


@pytest.mark.parametrize(
    ("substitution_mode", "extra_actions"),
    ((SUBSTITUTION_HOLD, 0), (SUBSTITUTION_SINGLE_STEP, 5 - 1)),
)
def test_minimal_recorded_continuation_is_accepted(substitution_mode, extra_actions) -> None:
    """The engine accepts the shortest suffix required by each substitution mode."""
    scenario = fx.preventable_late_braking_scenario()
    contact_step = fx.find_contact_step(scenario)
    assert contact_step is not None
    horizon = 5
    report = locate_last_avoidable(
        fx.KinematicCollisionModel(scenario),
        fx.maintain_baseline_actions(contact_step + extra_actions),
        ReplayConfig(
            t_danger=0,
            t_contact=contact_step,
            horizon=horizon,
            substitution_mode=substitution_mode,
        ),
    )

    assert report.verdict == VERDICT_AVOIDABLE


def test_hold_requires_inclusive_contact_prefix() -> None:
    """A hold replay must record the action whose result is the declared contact."""
    scenario = fx.preventable_late_braking_scenario()
    contact_step = fx.find_contact_step(scenario)
    assert contact_step is not None
    report = locate_last_avoidable(
        fx.KinematicCollisionModel(scenario),
        fx.maintain_baseline_actions(contact_step - 1),
        ReplayConfig(
            t_danger=0,
            t_contact=contact_step,
            horizon=1,
            substitution_mode=SUBSTITUTION_HOLD,
        ),
    )

    assert report.verdict == VERDICT_UNKNOWN
    assert report.abstain_reason == "insufficient_baseline_actions"


def test_contact_tick_matches_applied_action_count_at_boundary() -> None:
    """The observed contact state tick matches the inclusive contact config tick."""
    scenario = fx.preventable_late_braking_scenario()
    contact_tick = fx.find_contact_step(scenario)
    assert contact_tick is not None
    report = locate_last_avoidable(
        fx.KinematicCollisionModel(scenario),
        fx.maintain_baseline_actions(contact_tick),
        ReplayConfig(
            t_danger=0,
            t_contact=contact_tick,
            horizon=1,
            substitution_mode=SUBSTITUTION_HOLD,
        ),
    )

    assert report.determinism.observed_contact_steps == (contact_tick,) * 5


def test_declared_contact_tick_mismatch_abstains_before_branching() -> None:
    """A validly shaped but incorrect t_contact cannot certify avoidability."""
    scenario = fx.preventable_late_braking_scenario()
    actual_contact_tick = fx.find_contact_step(scenario)
    assert actual_contact_tick == 10
    config = ReplayConfig(
        t_danger=0,
        t_contact=12,
        horizon=18,
        substitution_mode=SUBSTITUTION_HOLD,
        determinism_replays=5,
    )

    report = locate_last_avoidable(
        fx.KinematicCollisionModel(scenario),
        fx.maintain_baseline_actions(config.t_contact + config.horizon + 2),
        config,
    )

    assert report.verdict == VERDICT_UNKNOWN
    assert report.abstained is True
    assert report.abstain_reason == "baseline_contact_tick_mismatch"
    assert report.determinism.observed_contact_steps == (actual_contact_tick,) * 5
    assert report.branches == ()
    assert report.t_uca is None
    assert report.t_inevitable is None
    assert report.minimal_sufficient_interventions == ()
    jsonschema.validate(report.to_dict(), _SCHEMA)


def test_post_contact_snapshots_are_coverage_gaps_for_branching() -> None:
    """Branching never tests interventions from snapshots already in contact."""
    scenario = fx.preventable_late_braking_scenario()
    actual_contact_tick = fx.find_contact_step(scenario)
    assert actual_contact_tick == 10
    config = ReplayConfig(
        t_danger=0,
        t_contact=12,
        horizon=2,
        substitution_mode=SUBSTITUTION_HOLD,
    )
    model = fx.KinematicCollisionModel(scenario)
    initial_snapshot = model.snapshot()
    baseline_actions = fx.maintain_baseline_actions(config.t_contact + config.horizon)
    snapshots = _capture_window_snapshots(model, initial_snapshot, baseline_actions, config)

    branches, interventions = _branch_over_window(model, snapshots, baseline_actions, config)
    post_contact_branches = [branch for branch in branches if branch.step >= actual_contact_tick]

    assert len(post_contact_branches) == config.t_contact - actual_contact_tick
    assert all(branch.feasible_count == 0 for branch in post_contact_branches)
    assert all(not branch.any_prevented for branch in post_contact_branches)
    assert all(intervention["step"] < actual_contact_tick for intervention in interventions)


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


def test_latent_trace_drift_is_nondeterministic_even_when_contact_matches() -> None:
    """Equal collision/contact outcomes do not hide a snapshot restore omission."""

    class _LatentDriftModel:
        """Model whose restore forgets one state field on purpose."""

        def __init__(self) -> None:
            self.step_index = 0
            self.latent = 0

        def snapshot(self):
            return {"step": self.step_index, "latent": self.latent}

        def restore(self, snapshot) -> None:
            self.step_index = snapshot["step"]

        def step(self, action) -> None:
            del action
            self.step_index += 1
            self.latent += 1

        def collision(self) -> bool:
            return self.step_index >= 3

        def feasible_actions(self):
            return (0.0,)

        def action_label(self, action) -> str:
            return f"action={action:g}"

    config = ReplayConfig(
        t_danger=0,
        t_contact=3,
        horizon=1,
        substitution_mode=SUBSTITUTION_HOLD,
        determinism_replays=2,
        source_kind="synthetic_fixture",
    )
    report = locate_last_avoidable(_LatentDriftModel(), [0.0, 0.0, 0.0], config)

    assert report.verdict == VERDICT_UNKNOWN
    assert report.abstain_reason == "nondeterministic_baseline"
    assert report.determinism.collision_stable is True
    assert report.determinism.contact_step_stable is True
    assert report.determinism.trace_stable is False
    assert report.determinism.first_divergence_step == 0
    assert report.determinism.first_divergence_field == "step.state.latent"


def test_incomplete_snapshot_state_returns_unknown_before_replay() -> None:
    """A model that omits mutable state cannot pass on an accidentally stable horizon."""
    model = fx.KinematicCollisionModel(fx.preventable_late_braking_scenario())
    model.replay_state_complete = False
    config = ReplayConfig(
        t_danger=0,
        t_contact=fx.find_contact_step(model.scenario) or 1,
        horizon=5,
        substitution_mode=SUBSTITUTION_HOLD,
    )
    report = locate_last_avoidable(
        model,
        fx.maintain_baseline_actions(config.t_contact + config.horizon + 2),
        config,
    )
    assert report.verdict == VERDICT_UNKNOWN
    assert report.abstained is True
    assert report.abstain_reason == "incomplete_snapshot_state"


def test_missing_feasible_action_returns_unknown() -> None:
    """A missing feasible action set abstains to unknown (coverage gap)."""
    report = _run(fx.missing_feasible_action_scenario())
    assert report.verdict == VERDICT_UNKNOWN
    assert report.abstained is True
    assert report.abstain_reason == "incomplete_feasible_action_coverage"
    assert report.feasible_coverage < 1.0
    assert report.verdict != "unavoidable"


def test_truncated_single_step_horizon_is_not_a_prevention() -> None:
    """A delayed contact beyond recorded data must fail closed, not return True."""
    scenario = fx.preventable_late_braking_scenario()
    model = fx.KinematicCollisionModel(scenario)
    snapshot = model.snapshot()
    config = ReplayConfig(
        t_danger=0,
        t_contact=3,
        horizon=5,
        substitution_mode="single_step",
    )
    with pytest.raises(ValueError, match="recorded baseline suffix"):
        _action_prevents_contact(model, snapshot, 0.0, 0, [0.0], config)

    report = locate_last_avoidable(model, [0.0], config)
    assert report.verdict == VERDICT_UNKNOWN
    assert report.abstain_reason == "insufficient_baseline_actions"


def test_late_coverage_gap_with_witness_does_not_certify_no_return() -> None:
    """A finite witness cannot certify avoidability when coverage is incomplete."""
    scenario = fx.preventable_late_braking_scenario()
    contact_step = fx.find_contact_step(scenario)
    assert contact_step is not None
    model = fx.KinematicCollisionModel(scenario)
    original_feasible = model.feasible_actions
    model.feasible_actions = lambda: () if model.step_index >= 7 else original_feasible()
    config = ReplayConfig(
        t_danger=0,
        t_contact=contact_step,
        horizon=contact_step + 6,
        substitution_mode=SUBSTITUTION_HOLD,
    )
    report = locate_last_avoidable(
        model,
        fx.maintain_baseline_actions(config.t_contact + config.horizon + 2),
        config,
    )
    assert report.verdict == VERDICT_UNKNOWN
    assert report.abstained is True
    assert report.abstain_reason == "incomplete_feasible_action_coverage"
    assert report.t_uca is None
    assert report.t_inevitable is None
    assert any("finite avoidance witness" in note for note in report.notes)


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


def test_runner_smoke_joins_causal_report(tmp_path) -> None:
    """With --join-causal-report the CLI also emits a valid collision_causal_report.v1."""
    from robot_sf.benchmark.collision.collision_causal_report import (
        COLLISION_CAUSAL_REPORT_SCHEMA_VERSION,
        validate_collision_causal_report,
    )
    from scripts.analysis.run_last_avoidable_replay_issue_5442 import main

    exit_code = main(
        [
            "--out-dir",
            str(tmp_path),
            "--fixtures",
            "preventable_late_braking",
            "--determinism-replays",
            "5",
            "--join-causal-report",
        ]
    )
    assert exit_code == 0
    causal_path = tmp_path / "preventable_late_braking__causal_report.json"
    causal = json.loads(causal_path.read_text())
    assert causal["schema_version"] == COLLISION_CAUSAL_REPORT_SCHEMA_VERSION
    # The joined report validates against the additive causal-report contract.
    validate_collision_causal_report(causal)
    assert causal["causal_contribution"]["verdict"] == "avoidable"
