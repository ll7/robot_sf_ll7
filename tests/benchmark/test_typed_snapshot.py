"""Focused proof for the preparation-only typed continuation snapshot."""

from __future__ import annotations

import hashlib
import json
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

import robot_sf.benchmark.typed_snapshot as typed_snapshot_module
from robot_sf.benchmark.simulator_counterfactual_adapter import SimulatorCounterfactualModel
from robot_sf.benchmark.typed_snapshot import (
    NoOpStep,
    SnapshotBoundary,
    SnapshotCompatibility,
    SnapshotCompatibilityError,
    SnapshotContractError,
    SnapshotPayloadError,
    TypedSimulatorSnapshot,
    capture_typed_snapshot,
    compare_continuation_traces,
    read_typed_snapshot,
    restore_typed_snapshot,
    state_inventory_payload,
    write_state_inventory,
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


def _build_model(*, capture_rng: bool = True) -> SimulatorCounterfactualModel:
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
    return SimulatorCounterfactualModel(simulator, collision_radius=0.5, capture_rng=capture_rng)


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


def _synthetic_compatibility() -> SnapshotCompatibility:
    """Return small deterministic identity metadata for pure contract tests."""
    return SnapshotCompatibility(
        map_sha256=hashlib.sha256(b"map").hexdigest(),
        config_sha256=hashlib.sha256(b"config").hexdigest(),
        code_revision=hashlib.sha256(b"revision").hexdigest(),
        dt_s=0.1,
        planner_id="synthetic.planner",
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
    expected_groups = deepcopy(model.sim.groups.groups)
    expected_group_by_ped = deepcopy(model.sim.groups.group_by_ped_id)
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
    model.sim.groups.groups = {}
    model.sim.groups.group_by_ped_id = {}
    model.sim.groups._invalidate_groups_as_lists_cache()
    model.sim.pysf_sim.peds.groups = []
    restore_typed_snapshot(model, loaded, compatibility)
    assert route_behavior.navigators[0].waypoint_id == original_route_id
    assert model.sim.groups.groups == expected_groups
    assert model.sim.groups.group_by_ped_id == expected_group_by_ped
    assert model.sim.pysf_sim.peds.groups == model.sim.groups.groups_as_lists
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


@pytest.mark.parametrize(
    "array_name",
    ("pysf_state", "ped_headings", "ped_angular_velocities", "ped_max_speeds"),
)
def test_array_shape_drift_is_rejected_before_destination_mutation(array_name: str) -> None:
    """Every restored numeric array must match the live destination before mutation."""
    model = _build_model()
    compatibility = _compatibility(model)
    typed = capture_typed_snapshot(model, compatibility)
    source = typed.arrays[array_name]
    wrong_shape = (
        (*source.shape[:-1], source.shape[-1] + 1) if source.ndim > 1 else (source.size + 1,)
    )
    arrays = dict(typed.arrays)
    arrays[array_name] = np.zeros(wrong_shape, dtype=source.dtype)
    malformed = TypedSimulatorSnapshot(
        compatibility=typed.compatibility,
        boundary=typed.boundary,
        state=typed.state,
        arrays=arrays,
    )
    before_step = model._step_index
    before_state = model.sim.pysf_state.pysf_states().copy()
    with pytest.raises(SnapshotCompatibilityError, match=f"{array_name} shape mismatch"):
        restore_typed_snapshot(model, malformed, compatibility)
    assert model._step_index == before_step
    np.testing.assert_array_equal(model.sim.pysf_state.pysf_states(), before_state)


@pytest.mark.parametrize(
    "array_name",
    ("pysf_state", "ped_headings", "ped_angular_velocities", "ped_max_speeds"),
)
def test_array_dtype_drift_is_rejected_before_destination_mutation(array_name: str) -> None:
    """Every restored numeric array must preserve the destination dtype exactly."""
    model = _build_model()
    compatibility = _compatibility(model)
    typed = capture_typed_snapshot(model, compatibility)
    source = typed.arrays[array_name]
    wrong_dtype = np.float32 if source.dtype != np.dtype(np.float32) else np.float64
    arrays = dict(typed.arrays)
    arrays[array_name] = np.asarray(source, dtype=wrong_dtype)
    malformed = TypedSimulatorSnapshot(
        compatibility=typed.compatibility,
        boundary=typed.boundary,
        state=typed.state,
        arrays=arrays,
    )
    before_step = model._step_index
    before_state = model.sim.pysf_state.pysf_states().copy()
    with pytest.raises(SnapshotCompatibilityError, match=f"{array_name} dtype mismatch"):
        restore_typed_snapshot(model, malformed, compatibility)
    assert model._step_index == before_step
    np.testing.assert_array_equal(model.sim.pysf_state.pysf_states(), before_state)


def test_rng_incomplete_snapshot_fails_closed_before_destination_mutation() -> None:
    """A capture without both RNG streams cannot claim an exact continuation."""
    model = _build_model(capture_rng=False)
    compatibility = _compatibility(model)
    typed = capture_typed_snapshot(model, compatibility)
    assert typed.state["rng_capture_complete"] is False
    before_step = model._step_index
    before_state = model.sim.pysf_state.pysf_states().copy()
    with pytest.raises(SnapshotCompatibilityError, match="complete RNG state"):
        restore_typed_snapshot(model, typed, compatibility)
    assert model._step_index == before_step
    np.testing.assert_array_equal(model.sim.pysf_state.pysf_states(), before_state)


def test_rng_disabled_destination_is_rejected_before_destination_mutation() -> None:
    """Atomic rollback requires a destination that snapshots both RNG streams."""
    source_model = _build_model()
    typed = capture_typed_snapshot(source_model, _compatibility(source_model))
    destination = _build_model(capture_rng=False)
    compatibility = _compatibility(destination)
    before_step = destination._step_index
    before_state = destination.sim.pysf_state.pysf_states().copy()
    with pytest.raises(SnapshotCompatibilityError, match="must capture RNG"):
        restore_typed_snapshot(destination, typed, compatibility)
    assert destination._step_index == before_step
    np.testing.assert_array_equal(destination.sim.pysf_state.pysf_states(), before_state)


def _assert_numpy_rng_state_equal(expected: tuple[object, ...], actual: tuple[object, ...]) -> None:
    """Compare legacy NumPy RNG state tuples without relying on array equality semantics."""
    assert expected[0] == actual[0]
    np.testing.assert_array_equal(expected[1], actual[1])
    assert expected[2:] == actual[2:]


def test_failed_python_rng_restore_rolls_back_numpy_and_python_streams() -> None:
    """A post-mutation RNG failure restores both streams from the rollback snapshot."""
    model = _build_model()
    compatibility = _compatibility(model)
    typed = capture_typed_snapshot(model, compatibility)
    state = deepcopy(typed.state)
    state["python_random_state"] = ("invalid-python-random-state",)
    malformed = TypedSimulatorSnapshot(
        compatibility=typed.compatibility,
        boundary=typed.boundary,
        state=state,
        arrays=typed.arrays,
    )
    np.random.random()
    random.random()
    before_numpy = np.random.get_state()
    before_python = random.getstate()
    with pytest.raises(SnapshotContractError, match="destination state was rolled back"):
        restore_typed_snapshot(model, malformed, compatibility)
    _assert_numpy_rng_state_equal(before_numpy, np.random.get_state())
    assert random.getstate() == before_python


def test_restore_failure_rolls_back_partial_adapter_mutation() -> None:
    """A restore failure cannot leave the destination at an intermediate state."""
    model = _build_model()
    compatibility = _compatibility(model)
    typed = capture_typed_snapshot(model, compatibility)
    before = model.snapshot()

    class _FailOnceModel:
        """Delegate the adapter while injecting one post-mutation failure."""

        def __init__(self, delegate: SimulatorCounterfactualModel) -> None:
            self.delegate = delegate
            self.sim = delegate.sim
            self.calls = 0

        def snapshot(self):
            return self.delegate.snapshot()

        def restore(self, snapshot):
            self.calls += 1
            if self.calls == 1:
                self.delegate._step_index = 999
                self.sim.pysf_state.pysf_states()[0, 0] = 123.0
                raise RuntimeError("synthetic restore failure")
            self.delegate.restore(snapshot)

    failing_model = _FailOnceModel(model)
    with pytest.raises(SnapshotContractError, match="destination state was rolled back"):
        restore_typed_snapshot(failing_model, typed, compatibility)

    assert failing_model.calls == 2
    assert model._step_index == before.step_index
    np.testing.assert_array_equal(model.sim.pysf_state.pysf_states(), before.pysf_state)


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
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(SnapshotPayloadError, match="unsupported snapshot schema"):
        read_typed_snapshot(path)

    metadata["schema_version"] = "simulator_typed_snapshot.v2"
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


def test_compatibility_and_boundary_round_trip_reject_invalid_metadata() -> None:
    """Compatibility and boundary parsers preserve valid values and reject malformed input."""
    compatibility = _synthetic_compatibility()
    assert SnapshotCompatibility.from_dict(compatibility.to_dict()) == compatibility
    with pytest.raises(SnapshotPayloadError, match="missing fields"):
        SnapshotCompatibility.from_dict({})
    invalid_compatibility = compatibility.to_dict()
    invalid_compatibility["dt_s"] = "not-a-number"
    with pytest.raises(SnapshotPayloadError, match="invalid compatibility"):
        SnapshotCompatibility.from_dict(invalid_compatibility)
    with pytest.raises(SnapshotContractError, match="checkpoint_sha256"):
        SnapshotCompatibility(
            map_sha256=compatibility.map_sha256,
            config_sha256=compatibility.config_sha256,
            code_revision=compatibility.code_revision,
            dt_s=compatibility.dt_s,
            checkpoint_sha256="bad",
        )

    boundary = SnapshotBoundary(
        step_index=2,
        absolute_time_s=0.2,
        remaining_budget_steps=4,
        next_observation_ready=True,
    )
    assert SnapshotBoundary.from_dict(boundary.to_dict()) == boundary
    with pytest.raises(SnapshotPayloadError, match="invalid snapshot boundary"):
        SnapshotBoundary.from_dict({})
    invalid_boundary = boundary.to_dict()
    invalid_boundary["phase"] = "post_step"
    with pytest.raises(SnapshotPayloadError, match="invalid snapshot boundary"):
        SnapshotBoundary.from_dict(invalid_boundary)
    invalid_boundary["phase"] = "pre_step"
    invalid_boundary["next_observation_ready"] = "yes"
    with pytest.raises(SnapshotPayloadError, match="next_observation_ready"):
        SnapshotBoundary.from_dict(invalid_boundary)


def test_typed_metadata_round_trip_encodes_nested_scalars_and_duplicate_array_names(
    tmp_path: Path,
) -> None:
    """The durable representation handles nested JSON values and collision-safe array names."""
    snapshot = TypedSimulatorSnapshot(
        compatibility=_synthetic_compatibility(),
        boundary=SnapshotBoundary(0, 0.0, None),
        state={
            "a.b": np.asarray([1.0]),
            "a_b": np.asarray([2.0]),
            "scalar": np.float64(3.0),
            "tuple": (True, None),
            "list": [1, {"nested": "value"}],
        },
        arrays={"seed": np.asarray([4.0])},
    )
    metadata = snapshot.to_metadata_dict()
    assert {"state_a_b", "state_a_b_2", "seed"} <= set(metadata["arrays"])
    artifact = write_typed_snapshot(snapshot, tmp_path / "synthetic.json")
    loaded = read_typed_snapshot(artifact.metadata_path)
    assert loaded.state["scalar"] == pytest.approx(3.0)
    assert loaded.state["tuple"] == (True, None)
    assert loaded.state["list"] == [1, {"nested": "value"}]
    assert np.array_equal(loaded.state["a.b"], np.asarray([1.0]))
    assert np.array_equal(loaded.state["a_b"], np.asarray([2.0]))


def test_nested_encoding_and_payload_validation_fail_closed() -> None:
    """Unsafe arrays, values, and references are rejected before durable use."""
    with pytest.raises(SnapshotContractError, match="unsafe object dtype"):
        typed_snapshot_module._array_copy(np.asarray([object()], dtype=object), "object")
    with pytest.raises(SnapshotContractError, match="numeric dtype"):
        typed_snapshot_module._array_copy(np.asarray(["text"]), "text")
    with pytest.raises(SnapshotContractError, match="non-finite values"):
        typed_snapshot_module._array_copy(np.asarray([np.inf]), "infinite")
    with pytest.raises(SnapshotContractError, match="unsupported value type"):
        typed_snapshot_module._encode_value(object(), {}, "state")
    with pytest.raises(SnapshotContractError, match="non-finite float"):
        typed_snapshot_module._encode_value(float("inf"), {}, "state")
    with pytest.raises(SnapshotPayloadError, match="missing array"):
        typed_snapshot_module._decode_value({"$array": "missing"}, {}, "state")
    with pytest.raises(SnapshotPayloadError, match="malformed tuple"):
        typed_snapshot_module._decode_value({"$tuple": "not-a-list"}, {}, "state")


def test_noop_comparison_covers_equal_nested_values_lengths_and_shapes() -> None:
    """No-op comparison distinguishes equal arrays, nested mismatches, and trace-length drift."""
    equal = compare_continuation_traces(
        [NoOpStep(step=0, state={"array": np.asarray([1, 2])})],
        [NoOpStep(step=0, state={"array": np.asarray([1, 2])})],
    )
    assert equal.equivalent is True
    shape_mismatch = compare_continuation_traces(
        [NoOpStep(step=0, state={"array": np.asarray([1, 2])})],
        [NoOpStep(step=0, state={"array": np.asarray([[1, 2]])})],
    )
    assert shape_mismatch.first_divergence_field == "step.state.array"
    missing_mapping_key = compare_continuation_traces(
        [NoOpStep(step=0, state={"present": 1})],
        [NoOpStep(step=0, state={})],
    )
    assert missing_mapping_key.first_divergence_field == "step.state.present"
    sequence_length = compare_continuation_traces(
        [NoOpStep(step=0, state={"items": [1]})],
        [NoOpStep(step=0, state={"items": [1, 2]})],
    )
    assert sequence_length.first_divergence_field == "step.state.items"
    trace_length = compare_continuation_traces(
        [NoOpStep(step=0, state={})],
        [NoOpStep(step=0, state={}), NoOpStep(step=1, state={})],
    )
    assert trace_length.first_divergence_field == "trace_length"


def test_state_inventory_writer_emits_deterministic_json(tmp_path: Path) -> None:
    """The inventory writer creates a readable artifact with the public inventory payload."""
    target = write_state_inventory(tmp_path / "inventory.json")
    assert target.is_file()
    assert json.loads(target.read_text(encoding="utf-8")) == state_inventory_payload()
