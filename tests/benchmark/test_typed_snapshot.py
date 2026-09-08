"""Focused proof for the preparation-only typed continuation snapshot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from robot_sf.benchmark.simulator_counterfactual_adapter import SimulatorCounterfactualModel
from robot_sf.benchmark.typed_snapshot import (
    NoOpStep,
    SnapshotCompatibility,
    SnapshotCompatibilityError,
    SnapshotPayloadError,
    capture_typed_snapshot,
    compare_continuation_traces,
    read_typed_snapshot,
    restore_typed_snapshot,
    state_inventory_payload,
    write_typed_snapshot,
)
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.sim.sim_config import SimulationSettings
from robot_sf.sim.simulator import init_simulators

_FIXTURE_MAP = Path(__file__).resolve().parents[2] / "maps/svg_maps/classic_doorway.svg"


@pytest.fixture(autouse=True)
def _preserve_global_rng():
    """Keep snapshot tests from leaking random-stream changes."""
    state = np.random.get_state()
    yield
    np.random.set_state(state)


def _build_model() -> SimulatorCounterfactualModel:
    """Build the small native doorway simulator used by adapter seam tests."""
    np.random.seed(25)
    map_def = convert_map(str(_FIXTURE_MAP))
    sim_config = SimulationSettings(
        difficulty=0,
        ped_density_by_difficulty=[0.06],
        route_spawn_seed=21,
    )
    config = RobotSimulationConfig(sim_config=sim_config)
    simulator = init_simulators(config, map_def, num_robots=1, random_start_pos=False)[0]
    return SimulatorCounterfactualModel(simulator, collision_radius=0.5)


def _compatibility(model: SimulatorCounterfactualModel) -> SnapshotCompatibility:
    """Return deterministic identity metadata for the native fixture."""
    digest = hashlib.sha256(b"typed-snapshot-fixture").hexdigest()
    return SnapshotCompatibility(
        map_sha256=digest,
        config_sha256=hashlib.sha256(b"config").hexdigest(),
        code_revision=hashlib.sha256(b"revision").hexdigest(),
        dt_s=float(model.sim.config.time_per_step_in_secs),
        planner_id="fixture.planner.stateless",
    )


def _rollout(model: SimulatorCounterfactualModel, steps: int) -> list[tuple[object, ...]]:
    """Record exact native state needed by the continuation negative control."""
    route_behavior = next(
        behavior for behavior in model.sim.peds_behaviors if hasattr(behavior, "navigators")
    )
    result: list[tuple[object, ...]] = []
    for _ in range(steps):
        result.append(
            (
                np.asarray(model.sim.pysf_state.pysf_states()).copy(),
                tuple(model.sim.robots[0].pose[0]),
                tuple(model.sim.robots[0].state.velocity),
                int(model.sim.robot_navs[0].waypoint_id),
                tuple(
                    (int(key), int(nav.waypoint_id), bool(nav.reached_waypoint))
                    for key, nav in sorted(route_behavior.navigators.items())
                ),
            )
        )
        model.step((0.2, 0.0))
    return result


def test_typed_snapshot_round_trip_preserves_exact_continuation(tmp_path: Path) -> None:
    """Capture, durable round-trip, and restore reproduce a native suffix exactly."""
    model = _build_model()
    compatibility = _compatibility(model)
    typed = capture_typed_snapshot(model, compatibility)
    route_behavior = next(
        behavior for behavior in model.sim.peds_behaviors if hasattr(behavior, "navigators")
    )
    original_route_id = route_behavior.navigators[0].waypoint_id

    artifact = write_typed_snapshot(typed, tmp_path / "continuation.json")
    loaded = read_typed_snapshot(artifact.metadata_path)
    assert artifact.total_bytes > artifact.metadata_bytes > 0
    assert artifact.payload_bytes > 0
    assert loaded.boundary.absolute_time_s == pytest.approx(0.0)
    assert loaded.boundary.remaining_budget_steps == 2000

    baseline = _rollout(model, 4)
    model.sim.robot_navs[0].waypoint_id = len(model.sim.robot_navs[0].waypoints) - 1
    route_behavior.navigators[0].waypoint_id = 0
    restore_typed_snapshot(model, loaded, compatibility)
    assert route_behavior.navigators[0].waypoint_id == original_route_id
    replay = _rollout(model, 4)

    for expected, actual in zip(baseline, replay, strict=True):
        assert np.array_equal(expected[0], actual[0])
        assert expected[1:] == actual[1:]


def test_incompatible_snapshot_is_rejected_before_destination_mutation() -> None:
    """A digest mismatch fails before ``model.restore`` can mutate the simulator."""
    model = _build_model()
    compatibility = _compatibility(model)
    typed = capture_typed_snapshot(model, compatibility)
    before = model.sim.robots[0].pose
    destination = SnapshotCompatibility(
        map_sha256=hashlib.sha256(b"different-map").hexdigest(),
        config_sha256=compatibility.config_sha256,
        code_revision=compatibility.code_revision,
        dt_s=compatibility.dt_s,
        planner_id=compatibility.planner_id,
    )
    with pytest.raises(SnapshotCompatibilityError):
        restore_typed_snapshot(model, typed, destination)
    assert model.sim.robots[0].pose == before


def test_corrupt_or_unknown_snapshot_metadata_fails_closed(tmp_path: Path) -> None:
    """Unknown schema markers and payload digest changes never load silently."""
    model = _build_model()
    path = tmp_path / "continuation.json"
    artifact = write_typed_snapshot(capture_typed_snapshot(model, _compatibility(model)), path)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["schema_version"] = "simulator_typed_snapshot.unknown"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(SnapshotPayloadError, match="unsupported snapshot schema"):
        read_typed_snapshot(path)

    metadata["schema_version"] = "simulator_typed_snapshot.v1"
    metadata["payload_sha256"] = "0" * 64
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(SnapshotPayloadError, match="payload digest"):
        read_typed_snapshot(path)
    assert artifact.payload_path.is_file()


def test_noop_comparator_reports_latent_state_not_collision_only() -> None:
    """A same-terminal trace still fails when one declared nested state differs."""
    expected = [
        NoOpStep(
            step=0,
            state={"route": {"waypoint_id": 2}, "rng": {"draw": 4}},
            terminal=False,
            events=[],
        ),
        NoOpStep(step=1, state={"route": {"waypoint_id": 3}}, terminal=True),
    ]
    actual = [
        NoOpStep(
            step=0,
            state={"route": {"waypoint_id": 2}, "rng": {"draw": 5}},
            terminal=False,
            events=[],
        ),
        NoOpStep(step=1, state={"route": {"waypoint_id": 3}}, terminal=True),
    ]
    comparison = compare_continuation_traces(expected, actual)
    assert comparison.equivalent is False
    assert comparison.first_divergence_step == 0
    assert comparison.first_divergence_field == "step.state.rng.draw"
    assert comparison.expected == 4
    assert comparison.actual == 5


def test_state_inventory_is_machine_readable_and_marks_gaps() -> None:
    """The inventory exposes owners, units, tests, and unsupported state explicitly."""
    payload = state_inventory_payload()
    assert payload["schema_version"] == "simulator_state_inventory.v1"
    assert len(payload["entries"]) >= 16
    assert {entry["status"] for entry in payload["entries"]} >= {"supported", "unsupported"}
    assert all("owner" in entry and "test" in entry for entry in payload["entries"])
    assert any(entry["path"] == "controller.planner_memory" for entry in payload["entries"])
