#!/usr/bin/env python3
"""Run step-level diagnostics for predictive planner on a hard-seed subset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from robot_sf.benchmark.map_runner import _build_env_config, _build_policy
from robot_sf.benchmark.predictive_planner_config import build_predictive_planner_algo_config
from robot_sf.gym_env.environment_factory import make_robot_env
from scripts.validation.predictive_eval_common import load_seed_manifest


def _json_ready(value):  # noqa: C901
    """Convert nested values to JSON-serializable primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return _json_ready(vars(value))
    return str(value)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-matrix", type=Path, required=True)
    parser.add_argument("--seed-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--algo-params", type=Path, default=None)
    parser.add_argument("--horizon", type=int, default=140)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/tmp/predictive_planner/diagnostics/hard_seed_diagnostics"),
    )
    return parser.parse_args()


def _scenario_pairs(
    scenario_matrix: Path, seed_manifest: dict[str, list[int]]
) -> list[tuple[dict, int]]:
    from robot_sf.training.scenario_loader import load_scenarios

    scenarios = load_scenarios(scenario_matrix)
    out: list[tuple[dict, int]] = []
    for scenario in scenarios:
        scenario_id = str(
            scenario.get("name") or scenario.get("scenario_id") or scenario.get("id") or "unknown"
        )
        for seed in seed_manifest.get(scenario_id, []):
            out.append((scenario, int(seed)))
    return out


def _base_algo_cfg(checkpoint: Path) -> dict:
    return build_predictive_planner_algo_config(checkpoint_path=checkpoint, device="cpu")


def _obs_min_robot_ped_distance(obs: dict) -> float:
    """Compute min robot-ped center distance from observation fields."""
    robot = obs.get("robot", {})
    peds = obs.get("pedestrians", {})
    robot_pos = np.asarray(robot.get("position", [0.0, 0.0]), dtype=float).reshape(-1)[:2]
    ped_pos = np.asarray(peds.get("positions", []), dtype=float)
    if ped_pos.ndim == 1:
        ped_pos = ped_pos.reshape(-1, 2) if ped_pos.size % 2 == 0 else np.zeros((0, 2), dtype=float)
    ped_count = int(np.asarray(peds.get("count", [ped_pos.shape[0]]), dtype=float).reshape(-1)[0])
    ped_count = max(0, min(ped_count, ped_pos.shape[0]))
    ped_pos = ped_pos[:ped_count]
    if ped_pos.size == 0:
        return float("nan")
    d = np.linalg.norm(ped_pos - robot_pos.reshape(1, 2), axis=1)
    return float(np.min(d))


def main() -> int:
    """Execute hard-seed diagnostics and write trace/summary artifacts."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seed_manifest = load_seed_manifest(args.seed_manifest)
    pairs = _scenario_pairs(args.scenario_matrix, seed_manifest)
    if not pairs:
        raise RuntimeError("No scenario/seed pairs resolved from manifest.")

    algo_cfg = _base_algo_cfg(args.checkpoint)
    if args.algo_params is not None:
        import yaml

        extra = yaml.safe_load(args.algo_params.read_text(encoding="utf-8")) or {}
        if not isinstance(extra, dict):
            raise TypeError(f"Algo params must be a mapping: {args.algo_params}")
        algo_cfg.update(extra)

    episode_rows: list[dict] = []
    for scenario, seed in pairs:
        scenario_id = str(
            scenario.get("name") or scenario.get("scenario_id") or scenario.get("id") or "unknown"
        )
        config = _build_env_config(scenario, scenario_path=args.scenario_matrix)
        policy_fn, _meta = _build_policy(
            "prediction_planner",
            dict(algo_cfg),
            robot_kinematics="differential_drive",
        )
        env = make_robot_env(config=config, seed=int(seed), debug=False)
        try:
            obs, _ = env.reset(seed=int(seed))
            steps = []
            done = False
            done_info: dict = {}
            for t in range(int(args.horizon)):
                action_v, action_w = policy_fn(obs)
                min_d = _obs_min_robot_ped_distance(obs)
                obs, reward, terminated, truncated, info = env.step(
                    np.array([action_v, action_w], dtype=float)
                )
                meta = info.get("meta", {}) if isinstance(info, dict) else {}
                step_row = {
                    "step": t,
                    "action_v": float(action_v),
                    "action_w": float(action_w),
                    "reward": float(reward),
                    "min_obs_robot_ped_dist": float(min_d) if np.isfinite(min_d) else None,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "is_success": bool(info.get("success") or info.get("is_success")),
                    "is_pedestrian_collision": bool(meta.get("is_pedestrian_collision", False)),
                    "is_obstacle_collision": bool(meta.get("is_obstacle_collision", False)),
                    "is_robot_collision": bool(meta.get("is_robot_collision", False)),
                }
                steps.append(step_row)
                if terminated or truncated or step_row["is_success"]:
                    done = True
                    done_info = {
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "success": step_row["is_success"],
                        "meta": meta,
                        "info": info,
                        "stop_step": t,
                    }
                    break
        finally:
            env.close()

        out_trace = args.output_dir / f"trace_{scenario_id}_{seed}.json"
        trace = {
            "scenario_id": scenario_id,
            "seed": int(seed),
            "horizon": int(args.horizon),
            "done": bool(done),
            "done_info": done_info,
            "steps": steps,
        }
        out_trace.write_text(json.dumps(_json_ready(trace), indent=2), encoding="utf-8")

        ped_coll_steps = [s["step"] for s in steps if s["is_pedestrian_collision"]]
        obs_min_vals = [
            s["min_obs_robot_ped_dist"] for s in steps if s["min_obs_robot_ped_dist"] is not None
        ]
        episode_rows.append(
            {
                "scenario_id": scenario_id,
                "seed": int(seed),
                "steps": len(steps),
                "success": bool(done_info.get("success", False)),
                "terminated": bool(done_info.get("terminated", False)),
                "truncated": bool(done_info.get("truncated", False)),
                "first_ped_collision_step": ped_coll_steps[0] if ped_coll_steps else None,
                "ped_collision_steps": ped_coll_steps,
                "min_obs_robot_ped_dist": float(min(obs_min_vals)) if obs_min_vals else None,
                "mean_action_v": float(np.mean([s["action_v"] for s in steps])) if steps else 0.0,
                "max_action_v": float(np.max([s["action_v"] for s in steps])) if steps else 0.0,
            }
        )

    summary_path = args.output_dir / "hard_seed_diagnostics_summary.json"
    summary_path.write_text(json.dumps(episode_rows, indent=2), encoding="utf-8")

    status_counter = Counter("success" if row["success"] else "failure" for row in episode_rows)
    ped_coll_counter = sum(1 for row in episode_rows if row["first_ped_collision_step"] is not None)

    report_lines = [
        "# Hard-Seed Step Diagnostics",
        "",
        f"- Scenario matrix: `{args.scenario_matrix}`",
        f"- Seed manifest: `{args.seed_manifest}`",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Episodes: `{len(episode_rows)}`",
        f"- Status counts: `{dict(status_counter)}`",
        f"- Episodes with pedestrian-collision flag: `{ped_coll_counter}`",
        "",
        "## Per Seed",
        "",
        "| Scenario | Seed | Success | Steps | First ped-collision step | Min obs robot-ped dist | Mean v | Max v |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(episode_rows, key=lambda r: (r["scenario_id"], r["seed"])):
        report_lines.append(
            "| "
            f"{row['scenario_id']} | {row['seed']} | {row['success']} | {row['steps']} | "
            f"{row['first_ped_collision_step']} | {row['min_obs_robot_ped_dist']} | "
            f"{row['mean_action_v']:.3f} | {row['max_action_v']:.3f} |"
        )

    report_path = args.output_dir / "hard_seed_diagnostics_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "report": str(report_path),
                "episodes": len(episode_rows),
                "status_counts": dict(status_counter),
                "episodes_with_ped_collision_flag": ped_coll_counter,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
