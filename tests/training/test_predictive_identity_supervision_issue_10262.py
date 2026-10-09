"""Focused deterministic regressions for predictive supervision actor identity (#10262)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from robot_sf.sensor.socnav_observation import SocNavObservationFusion
from robot_sf.training.predictive_supervision import (
    compatible_mixed_supervision,
    identity_match_indices,
    observation_episode_ids,
    supervision_metadata,
    validate_supervision_metadata,
)
from scripts.training import build_predictive_mixed_dataset as mixed_builder
from scripts.training import collect_predictive_hardcase_data as hardcase
from scripts.training import collect_predictive_planner_data as base
from scripts.training import train_predictive_planner as trainer

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(params=[base, hardcase], ids=["base", "hardcase"])
def collector(request: pytest.FixtureRequest):
    """Both independent collectors must satisfy the same identity contract."""
    return request.param


def _frame(
    collector,
    positions: list[tuple[float, float]],
    ids: tuple[str, ...],
    *,
    robot_heading: float = 0.0,
    pedestrian_velocity_ego: tuple[float, float] = (0.0, 0.0),
):
    """Produce a real collector Frame from the supported nested SOCNAV payload."""
    obs = {
        "robot": {
            "position": [0.0, 0.0],
            "heading": [robot_heading],
            "speed": [0.0, 0.0],
            "velocity_xy": [0.0, 0.0],
        },
        "goal": {"current": [5.0, 0.0]},
        "pedestrians": {
            "positions": np.asarray(positions, dtype=np.float32).reshape(-1, 2),
            "velocities": np.broadcast_to(
                np.asarray(pedestrian_velocity_ego, dtype=np.float32),
                (len(positions), 2),
            ).copy(),
            "count": np.asarray([len(positions)]),
        },
    }
    return collector._extract_frame(obs, max_agents=3, ped_ids=ids)


def _sample(collector, frames, *, horizon: int = 1):
    return collector._frames_to_samples(
        frames, max_agents=3, horizon_steps=horizon, ego_conditioning=False
    )


def test_crossing_witness_keeps_identity_in_both_collectors(collector) -> None:
    """A at 0 moves to 0.7; B at 0.8 moves to 0.1 (NN would swap both)."""
    a, b = "reset-1:A", "reset-1:B"
    frames = [
        _frame(collector, [(0.0, 0.0), (0.8, 0.0)], (a, b)),
        _frame(collector, [(0.7, 0.0), (0.1, 0.0)], (a, b)),
    ]
    _, target, mask, target_mask = _sample(collector, frames)
    np.testing.assert_allclose(target[0, :2, 0, 0], [0.7, 0.1])
    np.testing.assert_array_equal(mask[0], [1.0, 1.0, 0.0])
    np.testing.assert_array_equal(target_mask[0, :2, 0], [1.0, 1.0])


def test_observation_reordering_does_not_permute_targets(collector) -> None:
    """Source and future presentation orders can both change without relabeling."""
    a, b = "reset-1:A", "reset-1:B"
    frames = [
        _frame(collector, [(0.0, 0.0), (0.8, 0.0)], (a, b)),
        _frame(collector, [(0.1, 0.0), (0.7, 0.0)], (b, a)),
    ]
    _, target, _, target_mask = _sample(collector, frames)
    np.testing.assert_allclose(target[0, :2, 0, 0], [0.7, 0.1])
    np.testing.assert_array_equal(target_mask[0, :2, 0], [1.0, 1.0])


def test_pedestrian_velocity_is_already_ego_rotated(collector) -> None:
    """The source velocity feature must not be rotated twice at nonzero robot heading."""
    identity = "reset-1:A"
    frames = [
        _frame(
            collector,
            [(0.0, 1.0)],
            (identity,),
            robot_heading=float(np.pi / 2),
            pedestrian_velocity_ego=(0.3, -0.2),
        ),
        _frame(collector, [(0.0, 2.0)], (identity,)),
    ]
    state, target, _, target_mask = _sample(collector, frames)
    np.testing.assert_allclose(state[0, 0, 2:4], [0.3, -0.2])
    np.testing.assert_allclose(target[0, 0, 0], [2.0, 0.0], atol=1e-6)
    assert target_mask[0, 0, 0] == 1.0


def test_missing_and_reappearing_identity_is_masked_not_stolen(collector) -> None:
    """An absent actor has no target at that step but can reappear at a later horizon."""
    a, b = "reset-1:A", "reset-1:B"
    frames = [
        _frame(collector, [(0.0, 0.0), (0.8, 0.0)], (a, b)),
        _frame(collector, [(0.2, 0.0)], (b,)),
        _frame(collector, [(0.7, 0.0), (0.4, 0.0)], (a, b)),
    ]
    _, target, _, target_mask = _sample(collector, frames, horizon=2)
    np.testing.assert_array_equal(target_mask[0, :2], [[0.0, 1.0], [1.0, 1.0]])
    np.testing.assert_array_equal(target[0, 0, 0], [0.0, 0.0])
    np.testing.assert_allclose(target[0, 0, 1], [0.7, 0.0])
    np.testing.assert_allclose(target[0, 1, :, 0], [0.2, 0.4])


def test_zero_count_is_not_inferred_from_padded_positions(collector) -> None:
    """An explicit SOCNAV count=0 must not turn padded zeros into actors."""
    empty = _frame(collector, [(0.0, 0.0)], ("reset-1:A",))
    empty.ped_count = 0
    empty.ped_positions_world = np.zeros((0, 2), dtype=np.float32)
    empty.ped_velocities_ego = np.zeros((0, 2), dtype=np.float32)
    empty.ped_ids = ()
    next_frame = _frame(collector, [(0.1, 0.0)], ("reset-1:A",))
    _, _, mask, target_mask = _sample(collector, [empty, next_frame])
    assert not np.any(mask)
    assert not np.any(target_mask)


def test_reset_boundary_never_reuses_another_episode_identity(collector) -> None:
    first = _frame(collector, [(0.0, 0.0)], ("episode-1:simulator-slot-0",))
    second = _frame(collector, [(0.7, 0.0)], ("episode-2:simulator-slot-0",))
    _, target, _, target_mask = _sample(collector, [first, second])
    assert target_mask[0, 0, 0] == 0.0
    np.testing.assert_array_equal(target[0, 0, 0], [0.0, 0.0])


def test_missing_duplicate_or_unsupported_identity_rejected(collector) -> None:
    with pytest.raises(ValueError, match="observation-aligned"):
        collector._extract_frame(
            {
                "robot": {"position": [0.0, 0.0]},
                "goal": {"current": [0.0, 0.0]},
                "pedestrians": {"positions": [[0.0, 0.0]], "count": [1]},
            },
            max_agents=2,
        )
    with pytest.raises(ValueError, match="Duplicate"):
        identity_match_indices(("e:A", "e:A"), ("e:A",))
    frame = _frame(collector, [(0.0, 0.0)], ("e:A",))
    frame.ped_ids = ()
    with pytest.raises(ValueError, match="missing pedestrian identities"):
        _sample(collector, [frame, _frame(collector, [(0.1, 0.0)], ("e:A",))])


def test_sensor_source_sidechannel_uses_sorted_original_indices() -> None:
    """Exercise real production observation filtering/ordering, not synthetic index repair."""
    fusion = SocNavObservationFusion.__new__(SocNavObservationFusion)
    fusion.env_config = SimpleNamespace(
        observation_visibility=SimpleNamespace(
            enabled=False,
            memory_for_lost_pedestrians=False,
            ordering_tie_break="legacy",
        ),
    )
    fusion.simulator = SimpleNamespace(config=SimpleNamespace(time_per_step_in_secs=0.1))
    fusion.max_pedestrians = 2
    fusion._pedestrian_tracker = None
    fusion._current_tracking_result = None
    fusion._current_source_indices = ()
    fusion._lost_pedestrian_memory = {}
    positions = np.asarray([(0.8, 0.0), (0.0, 0.0), (1.5, 0.0)], dtype=np.float32)
    rows, _, _ = fusion._prepare_pedestrian_rows(
        ped_positions=positions,
        ped_velocities=np.zeros_like(positions),
        robot_pos=np.asarray((0.0, 0.0), dtype=np.float32),
        heading=0.0,
    )
    np.testing.assert_array_equal(rows[:, 0], [0.0, 0.8])
    assert fusion.current_source_indices == (1, 0)
    fusion.simulator.ped_pos = positions
    env = SimpleNamespace(
        state=SimpleNamespace(sensors=SimpleNamespace(wrapped_adapter=fusion)),
        simulator=fusion.simulator,
    )
    assert observation_episode_ids(env, episode_id="reset-1") == (
        "reset-1:simulator-slot-1", "reset-1:simulator-slot-0",
    )
    assert observation_episode_ids(env, episode_id="reset-2")[0] != (
        observation_episode_ids(env, episode_id="reset-1")[0]
    )


def test_missing_or_duplicate_sensor_source_index_fails_closed() -> None:
    sensor = SimpleNamespace(
        current_source_indices=(0, 0),
        env_config=SimpleNamespace(observation_visibility=SimpleNamespace()),
    )
    env = SimpleNamespace(
        state=SimpleNamespace(sensors=sensor),
        simulator=SimpleNamespace(ped_pos=np.zeros((2, 2))),
    )
    with pytest.raises(ValueError, match="Duplicate"):
        observation_episode_ids(env, episode_id="episode-1")
    del sensor.current_source_indices
    with pytest.raises(ValueError, match="requires SocNavObservationFusion"):
        observation_episode_ids(env, episode_id="episode-1")


def _npz(path: Path, *, marker: dict | None) -> None:
    payload = {
        "state": np.zeros((3, 2, 4), dtype=np.float32),
        "target": np.zeros((3, 2, 2, 2), dtype=np.float32),
        "mask": np.ones((3, 2), dtype=np.float32),
        "target_mask": np.ones((3, 2, 2), dtype=np.float32),
    }
    if marker is not None:
        payload["supervision_metadata_json"] = json.dumps(marker, sort_keys=True)
    np.savez_compressed(path, **payload)


def _mix(monkeypatch, base_path: Path, hard_path: Path, output: Path, *, allow_legacy: bool):
    monkeypatch.setattr(
        mixed_builder,
        "parse_args",
        lambda: mixed_builder.argparse.Namespace(
            base_dataset=base_path,
            hardcase_dataset=hard_path,
            output=output,
            hardcase_repeat=1,
            shuffle_seed=7,
            weighting_spec=None,
            allow_legacy_supervision=allow_legacy,
        ),
    )
    return mixed_builder.main()


@pytest.mark.parametrize("allow_legacy", [False, True])
def test_mixer_rejects_corrected_and_unmarked_mixture(monkeypatch, tmp_path, allow_legacy):
    a, b, out = (tmp_path / name for name in ("base.npz", "hard.npz", "mixed.npz"))
    _npz(a, marker=supervision_metadata("base"))
    _npz(b, marker=None)
    with pytest.raises(ValueError, match="supervision"):
        _mix(monkeypatch, a, b, out, allow_legacy=allow_legacy)
    assert not out.exists()


def test_mixer_writes_v2_metadata_in_npz_sidecar_and_manifest(monkeypatch, tmp_path) -> None:
    a, b, out = (tmp_path / name for name in ("base.npz", "hard.npz", "mixed.npz"))
    _npz(a, marker=supervision_metadata("base"))
    _npz(b, marker=supervision_metadata("hardcase"))
    assert _mix(monkeypatch, a, b, out, allow_legacy=False) == 0
    with np.load(out) as raw:
        metadata = validate_supervision_metadata(raw, path=out)
        assert metadata["collector_id"] == "mixed"
        assert metadata["source_collectors"] == ["base", "hardcase"]
        assert metadata["velocity_coordinate_frame"] == "robot_ego_xy_rotation_only"
    summary = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    manifest = json.loads(
        out.with_suffix(out.suffix + ".manifest.json").read_text(encoding="utf-8")
    )
    assert summary["supervision_metadata"] == metadata
    assert manifest["supervision_metadata"] == metadata


def test_mixer_rejects_velocity_frame_mismatch(monkeypatch, tmp_path) -> None:
    a, b, out = (tmp_path / name for name in ("base.npz", "hard.npz", "mixed.npz"))
    _npz(a, marker=supervision_metadata("base"))
    incompatible = supervision_metadata("hardcase")
    incompatible["velocity_coordinate_frame"] = "world_xy"
    _npz(b, marker=incompatible)
    with pytest.raises(ValueError, match="velocity_coordinate_frame"):
        _mix(monkeypatch, a, b, out, allow_legacy=False)


def test_unmarked_legacy_requires_explicit_training_optin(tmp_path) -> None:
    dataset = tmp_path / "old_unmarked.npz"
    _npz(dataset, marker=None)
    with np.load(dataset) as raw:
        with pytest.raises(ValueError, match="no supervision_metadata_json"):
            validate_supervision_metadata(raw, path=dataset)
        assert validate_supervision_metadata(raw, path=dataset, allow_legacy=True) is None
    with pytest.raises(ValueError, match="no supervision_metadata_json"):
        trainer.main(["--dataset", str(dataset), "--output-dir", str(tmp_path / "out")])


def test_identity_corrected_dataset_requires_explicit_target_mask(tmp_path) -> None:
    """The legacy mask-to-target_mask fallback must not mislabel absent v2 targets."""
    path = tmp_path / "incomplete_identity_v2.npz"
    _npz(path, marker=supervision_metadata("base"))
    with np.load(path) as raw:
        payload = {name: raw[name] for name in raw.files if name != "target_mask"}
    np.savez_compressed(path, **payload)
    with np.load(path) as raw:
        with pytest.raises(ValueError, match="target_mask"):
            validate_supervision_metadata(raw, path=path)


def test_explicit_zero_count_rejects_padded_phantom_actors(collector) -> None:
    obs = {
        "robot": {"position": [0.0, 0.0]},
        "goal": {"current": [2.0, 0.0]},
        "pedestrians": {
            "positions": [[0.0, 0.0], [0.0, 0.0]],
            "velocities": [[0.0, 0.0], [0.0, 0.0]],
            "count": [0],
        },
    }
    frame = collector._extract_frame(obs, max_agents=2, ped_ids=())
    assert frame.ped_count == 0
    assert frame.ped_ids == ()


def test_manifest_mismatch_is_not_silently_trusted(tmp_path) -> None:
    dataset = tmp_path / "corrected.npz"
    _npz(dataset, marker=supervision_metadata("base"))
    dataset.with_suffix(dataset.suffix + ".manifest.json").write_text(
        json.dumps({"supervision_metadata": supervision_metadata("hardcase")}),
        encoding="utf-8",
    )
    with np.load(dataset) as raw:
        with pytest.raises(ValueError, match="differs from embedded NPZ"):
            validate_supervision_metadata(raw, path=dataset)


def test_legacy_and_corrected_contracts_must_never_coalesce() -> None:
    with pytest.raises(ValueError, match="Incompatible"):
        compatible_mixed_supervision(None, supervision_metadata("hardcase"), allow_legacy=True)
    assert compatible_mixed_supervision(None, None, allow_legacy=True) is None
