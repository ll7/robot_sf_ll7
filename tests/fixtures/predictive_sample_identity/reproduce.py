"""Collect development-only row-identity diagnostics; never train a model."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

from robot_sf.training.predictive_supervision import validate_supervision_metadata
from scripts.training import collect_predictive_hardcase_data as hardcase
from scripts.training import collect_predictive_planner_data as base

ROOT = Path("tests/fixtures/predictive_sample_identity")


def collect(collector, argv: list[str]) -> list[dict]:
    """Capture a simulator oracle independently of the frame/target production helpers.

    Returns:
        Raw actor world coordinates and source robot poses before each collected frame.
    """
    original = collector.observation_episode_ids
    original_argv = sys.argv
    frames = []
    steps = {}

    def identities(env, *, episode_id, observation):
        ids = original(env, episode_id=episode_id, observation=observation)
        epochs = {}
        for behavior in env.simulator.peds_behaviors:
            offset = int(getattr(behavior, "global_ped_offset", 0) or 0)
            for local_id, count in getattr(behavior, "respawn_epochs", {}).items():
                epochs[offset + int(local_id)] = int(count)
        positions = env.simulator.ped_pos.copy()
        actors = {}
        for slot, position in enumerate(positions):
            identity = f"{episode_id}:simulator-slot-{slot}"
            if epochs.get(slot, 0):
                identity += f":respawn-{epochs[slot]}"
            actors[identity] = position.tolist()
        robot, _, _ = collector._extract_socnav_blocks(observation)
        step = steps.get(episode_id, 0)
        steps[episode_id] = step + 1
        frames.append(
            {
                "episode_id": episode_id,
                "step": step,
                "actors_world": actors,
                "robot_position": np.asarray(robot["position"]).tolist(),
                "robot_heading": float(np.asarray(robot["heading"]).reshape(-1)[0]),
            }
        )
        return ids

    try:
        collector.observation_episode_ids = identities
        sys.argv = [collector.__file__, *argv]
        assert collector.main() == 0
    finally:
        collector.observation_episode_ids = original
        sys.argv = original_argv
    return frames


def verify(dataset: Path, frames: list[dict]) -> dict:
    """Compare every active target to independently copied simulator actor coordinates.

    Returns:
        Counts of verified episodes, source rows and target points.
    """
    oracle = {(f["episode_id"], f["step"]): f for f in frames}
    checked = 0
    with np.load(dataset, allow_pickle=False) as raw:
        validate_supervision_metadata(raw, path=dataset)
        for i, episode in enumerate(raw["sample_episode_id"]):
            step = int(raw["sample_step"][i])
            source = oracle[(episode, step)]
            heading = source["robot_heading"]
            rotation = np.asarray(
                [[np.cos(heading), np.sin(heading)], [-np.sin(heading), np.cos(heading)]]
            )
            for j in np.flatnonzero(raw["mask"][i]):
                actor = raw["sample_pedestrian_id"][i, j]
                assert actor in source["actors_world"]
                for k, future_step in enumerate(raw["sample_target_step"][i]):
                    future = oracle[(episode, int(future_step))]
                    present = actor in future["actors_world"]
                    if raw["target_mask"][i, j, k]:
                        assert present
                        relative = (
                            np.asarray(future["actors_world"][actor]) - source["robot_position"]
                        )
                        np.testing.assert_allclose(
                            raw["target"][i, j, k], rotation @ relative, atol=2e-5
                        )
                        checked += 1
        assert set(raw["sample_seed"]) == {1001, 1002, 1003}
        return {
            "episodes": len(set(raw["sample_episode_id"])),
            "samples": len(raw["state"]),
            "active_source_rows": int(raw["mask"].sum()),
            "checked_target_points": checked,
            "identity_check": "pass",
            "simulator_target_oracle": "pass",
        }


def main() -> None:
    """Produce immutable compact diagnostics and a path-bound checksum manifest."""
    results = {}
    for name, collector in (("base", base), ("hardcase", hardcase)):
        output = ROOT / f"{name}.npz"
        common = [
            "--max-steps",
            "12",
            "--horizon-steps",
            "4",
            "--max-agents",
            "16",
            "--output",
            str(output),
        ]
        if name == "base":
            argv = [
                "--episodes",
                "3",
                "--seed",
                "1001",
                "--episode-seeds",
                "1001",
                "1002",
                "1003",
                "--min-goal-distance",
                "0",
                *common,
            ]
        else:
            argv = [
                "--scenario-matrix",
                str(ROOT / "scenario.yaml"),
                "--seed-manifest",
                str(ROOT / "seeds.yaml"),
                *common,
            ]
        frames = collect(collector, argv)
        (ROOT / f"{name}-source-frames.json").write_text(json.dumps(frames, indent=2) + "\n")
        results[name] = {"argv": argv, **verify(output, frames)}
    (ROOT / "verification.json").write_text(json.dumps(results, indent=2) + "\n")
    sources = [
        "robot_sf/training/predictive_supervision.py",
        *[
            f"scripts/training/{module}.py"
            for module in (
                "collect_predictive_planner_data",
                "collect_predictive_hardcase_data",
                "build_predictive_mixed_dataset",
            )
        ],
    ]
    files = [
        *sorted(p for p in ROOT.iterdir() if p.is_file() and p.name != "manifest.json"),
        *map(Path, sources),
    ]
    manifest = {
        "schema_version": "predictive-identity-dev-proof.v1",
        "claim_boundary": "Collection identity diagnostics only; no model training, navigation evaluation or admission claim.",
        "producer_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "seeds": [1001, 1002, 1003],
        "artifacts": [
            {"artifact_path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in files
        ],
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
