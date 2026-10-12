"""Persisted predictive supervision must retain auditable row identities (#10262)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.training import collect_predictive_hardcase_data as hardcase
from scripts.training import collect_predictive_planner_data as base


@pytest.mark.parametrize("collector", [base, hardcase], ids=["base", "hardcase"])
def test_collection_persists_sample_identity(collector, monkeypatch, tmp_path: Path) -> None:
    """Exercise the real simulator and NPZ writer; identity-v2 drops source IDs."""
    output = tmp_path / "collected.npz"
    scenario = tmp_path / "scenario.yaml"
    scenario.write_text(
        f"scenarios:\n  - name: provenance_dev\n    map_file: {Path('maps/svg_maps/classic_bottleneck.svg').resolve()}\n"
        "    simulation_config:\n      ped_density: 0.04\n",
        encoding="utf-8",
    )
    seeds = tmp_path / "seeds.yaml"
    seeds.write_text("provenance_dev: [1001]\n", encoding="utf-8")
    args = argparse.Namespace(
        episodes=1,
        max_steps=12,
        max_agents=16,
        horizon_steps=4,
        ego_conditioning=False,
        model_family="predictive_legacy_v1",
        seed=1001,
        episode_seeds=[1001],
        max_speed=1.0,
        min_goal_distance=0.0,
        max_reset_attempts=1,
        difficulty=0,
        max_peds_per_group=1,
        use_global_routing=False,
        output=output,
        scenario_matrix=scenario,
        seed_manifest=seeds,
    )
    monkeypatch.setattr(collector, "parse_args", lambda: args)
    if collector is base:
        original = base._reset_with_min_goal

        class DevSeed:
            def integers(self, *args, **kwargs):
                return 1001

        def reset(**kwargs):
            kwargs["rng"] = DevSeed()
            return original(**kwargs)

        monkeypatch.setattr(base, "_reset_with_min_goal", reset)
    assert collector.main() == 0
    with np.load(output, allow_pickle=False) as raw:
        required = {
            "sample_scenario_id",
            "sample_episode_id",
            "sample_seed",
            "sample_step",
            "sample_pedestrian_id",
            "sample_target_pedestrian_id",
            "sample_target_step",
        }
        assert required <= set(raw.files), "NPZ discarded checkable predictive sample identities"
        assert raw["sample_seed"].tolist() == [1001] * len(raw["state"])
        assert len(set(raw["sample_episode_id"])) == 1
        mask = raw["mask"].astype(bool)
        target_mask = raw["target_mask"].astype(bool)
        assert mask.any(), "collection must include real pedestrians"
        ids = raw["sample_pedestrian_id"]
        assert np.all(ids[mask] != "")
        expected = np.broadcast_to(ids[..., None], raw["target_mask"].shape)
        np.testing.assert_array_equal(
            raw["sample_target_pedestrian_id"][target_mask], expected[target_mask]
        )
        metadata = json.loads(raw["supervision_metadata_json"].item())
        assert metadata["sample_identity_schema"] == "predictive_sample_identity_v1"


def _identity_payload(collector_id="base", *, episode="collection:base-episode-0"):
    """A compact source/target fixture with independently enumerated expected IDs."""
    from robot_sf.training import predictive_supervision as supervision

    ids = [f"{episode}:simulator-slot-0", f"{episode}:simulator-slot-1"]
    return {
        "state": np.zeros((2, 3, 4), dtype=np.float32),
        "target": np.zeros((2, 3, 2, 2), dtype=np.float32),
        "mask": np.asarray([[1, 1, 0]] * 2, dtype=np.float32),
        "target_mask": np.asarray([[[1, 1], [1, 1], [0, 0]]] * 2, dtype=np.float32),
        "sample_scenario_id": np.asarray(["scenario-a"] * 2),
        "sample_episode_id": np.asarray([episode] * 2),
        "sample_seed": np.asarray([1001, 1001]),
        "sample_step": np.asarray([0, 1]),
        "sample_target_step": np.asarray([[1, 2], [2, 3]]),
        "sample_pedestrian_id": np.asarray([[*ids, ""]] * 2),
        "sample_target_pedestrian_id": np.asarray([[[ids[0]] * 2, [ids[1]] * 2, [""] * 2]] * 2),
        "supervision_metadata_json": json.dumps(supervision.collection_metadata(collector_id)),
    }


@pytest.mark.parametrize(
    "corruption",
    ["borrow_target", "borrow_episode", "seed", "step", "duplicate", "padding", "missing", "mask"],
)
def test_persisted_identity_corruption_is_refused(tmp_path, corruption):
    """Labels cannot be borrowed across actors, episodes, seeds or padded rows."""
    from robot_sf.training import predictive_supervision as supervision

    payload = _identity_payload()
    if corruption == "borrow_target":
        payload["sample_target_pedestrian_id"][0, 0, 0] = payload["sample_pedestrian_id"][0, 1]
    elif corruption == "borrow_episode":
        payload["sample_pedestrian_id"][0, 0] = "another:simulator-slot-0"
    elif corruption == "seed":
        payload["sample_seed"][1] = 1002
    elif corruption == "step":
        payload["sample_target_step"][0, 0] = 2
    elif corruption == "duplicate":
        payload["sample_pedestrian_id"][0, 1] = payload["sample_pedestrian_id"][0, 0]
        payload["sample_target_pedestrian_id"][0, 1] = payload["sample_target_pedestrian_id"][0, 0]
    elif corruption == "padding":
        payload["sample_pedestrian_id"][0, 2] = "borrowed"
    elif corruption == "missing":
        del payload["sample_seed"]
    else:
        payload["mask"][0, 0] = 0.5
    dataset = tmp_path / "bad.npz"
    np.savez_compressed(dataset, **payload)
    with (
        np.load(dataset) as raw,
        pytest.raises(ValueError, match="identity|identities|steps|association|binary"),
    ):
        supervision.validate_supervision_metadata(raw, path=dataset)


def test_mixing_preserves_identity_through_repeat_and_shuffle(monkeypatch, tmp_path):
    """Feature rows and identities undergo the same permutation and oversampling."""
    from robot_sf.training import predictive_supervision as supervision
    from scripts.training import build_predictive_mixed_dataset as mixer

    paths = [tmp_path / name for name in ("base.npz", "hard.npz", "mix.npz")]
    lookup = {}
    for collector_id, path in zip(("base", "hardcase"), paths[:2], strict=True):
        payload = _identity_payload(collector_id, episode=f"collection:{collector_id}-episode-0")
        offset = 10 if collector_id == "base" else 20
        payload["state"][:, :, 0] = np.asarray([offset, offset + 1])[:, None]
        for i in range(2):
            lookup[offset + i] = (payload["sample_episode_id"][i], payload["sample_step"][i])
        np.savez_compressed(path, **payload)
    monkeypatch.setattr(
        mixer,
        "parse_args",
        lambda: argparse.Namespace(
            base_dataset=paths[0],
            hardcase_dataset=paths[1],
            output=paths[2],
            hardcase_repeat=3,
            shuffle_seed=1001,
            weighting_spec=None,
            allow_legacy_supervision=False,
        ),
    )
    assert mixer.main() == 0
    with np.load(paths[2]) as raw:
        supervision.validate_supervision_metadata(raw, path=paths[2])
        assert len(raw["state"]) == 8
        for i, state in enumerate(raw["state"]):
            assert (raw["sample_episode_id"][i], raw["sample_step"][i]) == lookup[state[0, 0]]


def test_new_and_historical_identity_provenance_never_mix():
    from robot_sf.training import predictive_supervision as supervision

    with pytest.raises(ValueError, match="provenance"):
        supervision.compatible_mixed_supervision(
            supervision.collection_metadata("base"), supervision.supervision_metadata("hardcase")
        )


def test_borrowed_observation_source_indices_are_refused():
    """A valid slot list alone does not certify a different observation's actors."""
    from types import SimpleNamespace

    from robot_sf.training import predictive_supervision as supervision

    sensor = SimpleNamespace(current_source_indices=(0, 1), _position_cap=lambda: 10.0)
    env = SimpleNamespace(
        state=SimpleNamespace(sensors=sensor),
        simulator=SimpleNamespace(
            ped_pos=np.asarray([[0.7, 0.0], [0.1, 0.0]], dtype=np.float32),
        ),
    )
    observation = {
        "pedestrians": {
            "positions": np.asarray([[0.1, 0.0], [0.7, 0.0]], dtype=np.float32),
            "count": [2],
        }
    }
    with pytest.raises(ValueError, match="Borrowed or stale"):
        supervision.observation_episode_ids(env, episode_id="e1", observation=observation)


def test_historical_identity_v2_is_flagged_not_relabelled(tmp_path):
    from robot_sf.training import predictive_supervision as supervision

    payload = _identity_payload()
    payload = {key: value for key, value in payload.items() if not key.startswith("sample_")}
    marker = supervision.supervision_metadata("base")
    payload["supervision_metadata_json"] = json.dumps(marker)
    dataset = tmp_path / "v2.npz"
    np.savez_compressed(dataset, **payload)
    with np.load(dataset) as raw, pytest.warns(UserWarning, match="not sample-identity verified"):
        assert supervision.validate_supervision_metadata(raw, path=dataset) == marker


def test_repeated_sample_key_cannot_borrow_another_row(tmp_path):
    from robot_sf.training import predictive_supervision as supervision

    payload = _identity_payload()
    payload["sample_step"][1] = 0
    payload["sample_target_step"][1] = [1, 2]
    payload["target"][1, 0, 0, 0] = 0.7
    dataset = tmp_path / "conflicting.npz"
    np.savez_compressed(dataset, **payload)
    with np.load(dataset) as raw, pytest.raises(ValueError, match="reused with different rows"):
        supervision.validate_supervision_metadata(raw, path=dataset)


def test_reset_records_actual_seed_after_rejected_attempt():
    from types import SimpleNamespace

    seeds = iter([100001, 100002])
    used = []

    class Seeds:
        def integers(self, low, high):
            assert (low, high) == (100000, 200000)
            return next(seeds)

    def reset(*, seed):
        used.append(seed)
        distance = 0.0 if len(used) == 1 else 5.0
        return {"robot_position": [0.0, 0.0], "goal_current": [distance, 0.0]}, {}

    obs, skipped, seed = base._reset_with_min_goal(
        env=SimpleNamespace(reset=reset),
        rng=Seeds(),
        min_goal_distance=2.0,
        max_reset_attempts=2,
    )
    assert obs is not None and skipped == 0 and seed == 100002
    assert used == [100001, 100002]


def test_preserved_dev_collection_hashes_and_target_oracle():
    """The durable bundle remains readable, checksum-bound and simulator-verifiable."""
    import hashlib
    import runpy

    root = Path("tests/fixtures/predictive_sample_identity")
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["seeds"] == [1001, 1002, 1003]
    for entry in manifest["artifacts"]:
        artifact = Path(entry["artifact_path"])
        assert not artifact.is_absolute()
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == entry["sha256"]
    verify = runpy.run_path(str(root / "reproduce.py"))["verify"]
    for collector_id in ("base", "hardcase"):
        frames = json.loads((root / f"{collector_id}-source-frames.json").read_text())
        result = verify(root / f"{collector_id}.npz", frames)
        assert result["episodes"] == 3 and result["samples"] == 24
        assert result["checked_target_points"] > 0
