"""CPU-only, five-development-seed replication of the issue-9609 wall probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from scripts.analysis import narrow_doorway_crash_vs_wait_issue_9545 as replay

SEEDS = tuple(range(1001, 1006))
OFFSETS = (20, 10, 5, 1)


def forward_wall_distance(obs: dict) -> float | None:
    """Measure nearest static occupied cell center in the robot's forward corridor.

    Uses model-ready grid metadata, including channel order and frame. Cell-center
    distances are diagnostic measurements, not physical wall-surface clearance.
    """
    channel = int(obs["occupancy_grid_meta_channel_indices"][0])
    if channel < 0:
        raise ValueError("Static obstacle channel missing")
    grid = np.asarray(obs["occupancy_grid"])[channel]
    rows, cols = np.nonzero(grid > 0.5)
    resolution = float(obs["occupancy_grid_meta_resolution"][0])
    xy = np.column_stack((cols + 0.5, rows + 0.5)) * resolution
    xy += np.asarray(obs["occupancy_grid_meta_origin"])
    if not bool(obs["occupancy_grid_meta_use_ego_frame"][0]):
        pose = np.asarray(obs["occupancy_grid_meta_robot_pose"])
        xy -= pose[:2]
        c, s = np.cos(pose[2]), np.sin(pose[2])
        xy = xy @ np.array([[c, -s], [s, c]])
    radius = float(np.asarray(obs["robot_radius"]).reshape(-1)[0])
    forward = xy[(xy[:, 0] >= 0) & (np.abs(xy[:, 1]) <= radius + resolution)]
    return float(np.min(np.linalg.norm(forward, axis=1))) if len(forward) else None


def policy_signals(model, obs: dict) -> tuple[list[float], float]:
    """Read deterministic Gaussian mean before clipping and critic at the same state."""
    tensor, _ = model.policy.obs_to_tensor(obs)
    with torch.no_grad():
        action = model.policy.get_distribution(tensor).mode().cpu().numpy().reshape(-1)
        value = model.policy.predict_values(tensor).cpu().numpy().item()
    if not np.all(np.isfinite(action)) or not np.isfinite(value):
        raise ValueError("Nonfinite actor/value output")
    return action.tolist(), float(value)


def classify(rows: list[dict]) -> str:
    """Require all four offsets, forward high commands, and an approaching wall."""
    if len(rows) != len(OFFSETS) or any(r["forward_wall_cell_m"] is None for r in rows):
        return "not_identifiable"
    distances = [r["forward_wall_cell_m"] for r in rows]
    if distances[-1] < distances[0] and all(r["adapter_command"][0] >= 1.85 for r in rows):
        return "holds"
    return "does_not_hold"


def run_probe(out: Path, *, max_steps: int = 600) -> dict:
    """Run fixed development seeds without fallback and preserve matched-state signals."""
    from robot_sf.benchmark.map_runner_policies.map_runner_actions import (
        policy_command_to_env_action,
    )
    from robot_sf.models import resolve_model_path

    artifacts = {}
    for model_id in (replay.MODEL_ID, replay.PREDICTIVE_MODEL_ID):
        path = Path(resolve_model_path(model_id))
        expected = replay._release_artifact_metadata(model_id)
        replay._verify_resolved_release_artifact(path, expected)
        artifacts[model_id] = expected
    planner = replay._make_planner()
    results = []
    for seed in SEEDS:
        env, _, _ = replay._build_diagnostic_env(seed)
        samples = []
        contact = None
        try:
            obs, _ = env.reset(seed=seed)
            for step in range(max_steps):
                runner_obs = replay._normalize_runner_obs(obs)
                command = planner.step(runner_obs)
                replay._assert_predictive_foresight_loaded(planner)
                model_obs = planner._build_model_obs_dict(runner_obs)
                action, value = policy_signals(planner._model, model_obs)
                ablated = {k: np.array(v, copy=True) for k, v in model_obs.items()}
                indices = ablated["occupancy_grid_meta_channel_indices"]
                ablated["occupancy_grid"][int(indices[0])] = 0
                if indices[3] >= 0 and indices[1] >= 0:
                    ablated["occupancy_grid"][int(indices[3])] = ablated["occupancy_grid"][
                        int(indices[1])
                    ]
                ablation_action, ablation_value = policy_signals(planner._model, ablated)
                native = policy_command_to_env_action(
                    env=env, config=env.env_config, command=(command["v"], command["omega"])
                )
                sample = {
                    "step": step,
                    "forward_wall_cell_m": forward_wall_distance(model_obs),
                    "physical_wall_clearance_m": replay._min_obstacle_clearance(env),
                    "raw_actor_velocity_delta_mean": action,
                    "adapter_command": [command["v"], command["omega"]],
                    "native_action": replay._serialize_env_action(native),
                    "critic_value": value,
                    "static_ablated_velocity_delta_mean": ablation_action,
                    "static_ablated_critic_value": ablation_value,
                }
                obs, _, terminated, truncated, info = env.step(np.asarray(native))
                sample["executed_speed_m_s"] = float(
                    np.linalg.norm(replay._normalize_runner_obs(obs)["robot_velocity_xy"])
                )
                samples.append(sample)
                if info.get("meta", {}).get("is_obstacle_collision", False):
                    contact = step
                    break
                if terminated or truncated:
                    break
            rows = (
                []
                if contact is None
                else [
                    {"offset": offset, **samples[contact - offset]}
                    for offset in OFFSETS
                    if contact >= offset
                ]
            )
            results.append(
                {"seed": seed, "contact_step": contact, "rows": rows, "result": classify(rows)}
            )
        finally:
            env.close()
    report = {
        "schema": "ppo-wall-probe-dev.v2",
        "review_status": "AI-GENERATED NEEDS-REVIEW",
        "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "producer_path": "scripts/analysis/ppo_wall_probe_dev.py",
        "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scenario_path": replay.SCENARIO_YAML,
        "scenario_sha256": hashlib.sha256(Path(replay.SCENARIO_YAML).read_bytes()).hexdigest(),
        "artifacts": artifacts,
        "fallback_used": False,
        "observation_normalization": "none; released PPO uses raw dict observations",
        "scope": "diagnostic only; narrow doorway, five dev seeds; no causal or population claim",
        "action_semantics": "velocity_delta; Gaussian means are signed deltas, adapter_command is the absolute target (v, omega)",
        "criterion": "all four pre-contact absolute adapter target speeds >=1.85 m/s and final forward occupied cell nearer than first",
        "results": results,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main() -> None:
    """Write the reproducible diagnostic report; never train or submit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run_probe(args.out)
    print(json.dumps([{"seed": r["seed"], "result": r["result"]} for r in result["results"]]))


if __name__ == "__main__":
    main()
