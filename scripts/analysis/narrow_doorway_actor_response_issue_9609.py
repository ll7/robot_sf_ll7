#!/usr/bin/env python3
"""Matched PPO observation/action diagnostic for issue #9609.

This is a diagnostic-only continuation of issue #9545.  It replays the three
canonical narrow-doorway seeds from the exact PR #9546 evidence boundary and
samples four states before contact.  At each state it records:

* static geometry in the model-ready ego occupancy grid,
* the actor mean, Stable-Baselines ``predict`` action, and critic value,
* the PPO adapter command, native environment action, and executed velocity,
* a matched intervention that removes only the static-geometry channels.

The intervention is intentionally out of distribution and is used only to
distinguish observation/adapter/policy-response explanations.  It is not a
benchmark row, retraining result, or population-level mechanism claim.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from robot_sf.benchmark.map_runner_policies.map_runner_actions import (
    policy_command_to_env_action,
)
from robot_sf.evidence.writers import (
    write_distance_series_csv,
    write_json,
    write_review_sidecar,
)
from scripts.analysis import narrow_doorway_crash_vs_wait_issue_9545 as parent

ISSUE = 9609
PARENT_ISSUE = 9545
PARENT_PR = 9546
PARENT_HEAD = "59949b0cac2fd35988e2d8947c55ab1dac2c70d3"
SEEDS = (225, 226, 227)
OFFSETS = (20, 10, 5, 1)
CLAIM_BOUNDARY = (
    "diagnostic-only evidence for the bound checkpoint/scenario/states; "
    "not benchmark, training-causal, population-level, paper, or dissertation evidence"
)
REPRO_COMMAND = (
    "uv run python scripts/analysis/narrow_doorway_actor_response_issue_9609.py "
    "--output-dir docs/context/evidence/issue_9609_narrow_doorway_actor_response"
)


def _write_review_sidecar(path: Path, *, artifact_path: Path | None = None) -> Path:
    sidecar_path = write_review_sidecar(path)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if artifact_path is not None:
        repo_root = Path(__file__).resolve().parents[2]
        try:
            sidecar["artifact_path"] = artifact_path.resolve().relative_to(repo_root).as_posix()
        except ValueError:
            sidecar["artifact_path"] = artifact_path.name
    sidecar.update(
        {
            "claim_boundary": CLAIM_BOUNDARY,
            "dissertation_admission_status": "not_admitted",
            "domain_approval_status": "not_granted",
            "review_status": "needs_independent_review",
        }
    )
    write_json(sidecar_path, sidecar)
    return sidecar_path


def _publish_artifact_sidecar_pair(
    staged_artifact: Path,
    staged_sidecar: Path,
    artifact_path: Path,
    staging_dir: Path,
) -> None:
    """Replace an artifact and its hash sidecar as a rollback-capable pair.

    The two public filenames cannot be renamed in one filesystem operation. Both
    replacements are therefore prepared first, previous bytes are backed up,
    and an ordinary publication error restores the old pair. A forced process
    termination between renames can leave a missing or mismatched sidecar, which
    downstream evidence checks reject by the recorded SHA-256.
    """
    sidecar_path = artifact_path.with_name(artifact_path.name + ".review.json")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    backup_dir = staging_dir / "previous"
    backup_dir.mkdir()
    backups: dict[Path, Path | None] = {}
    for target in (artifact_path, sidecar_path):
        if target.exists():
            backup = backup_dir / target.name
            shutil.copy2(target, backup)
            backups[target] = backup
        else:
            backups[target] = None

    try:
        os.replace(staged_artifact, artifact_path)
        os.replace(staged_sidecar, sidecar_path)
    except BaseException as publication_error:
        rollback_errors: list[OSError] = []
        for target in (sidecar_path, artifact_path):
            backup = backups[target]
            try:
                if backup is None:
                    target.unlink(missing_ok=True)
                else:
                    os.replace(backup, target)
            except OSError as rollback_error:
                rollback_errors.append(rollback_error)
        if rollback_errors:
            raise RuntimeError(
                "artifact/sidecar publication failed and rollback was incomplete; "
                f"inspect {artifact_path} and {sidecar_path}"
            ) from publication_error
        raise


def _publication_staging_root(path: Path) -> Path:
    """Choose a same-filesystem staging parent outside the evidence tree."""
    repo_root = Path(__file__).resolve().parents[2]
    evidence_root = (repo_root / "docs" / "context" / "evidence").resolve()
    try:
        path.resolve().relative_to(evidence_root)
    except ValueError:
        return path.parent
    return repo_root / "docs" / "context"


def _write_json_with_review_sidecar(path: Path, payload: dict[str, Any]) -> None:
    """Stage a marked JSON artifact and sidecar before publishing either file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    staging_root = _publication_staging_root(path)
    with tempfile.TemporaryDirectory(
        prefix=".issue9609-actor-response-", dir=staging_root
    ) as temporary_directory:
        staging_dir = Path(temporary_directory)
        staged_artifact = staging_dir / path.name
        write_json(staged_artifact, payload)
        staged_sidecar = _write_review_sidecar(staged_artifact, artifact_path=path)
        _publish_artifact_sidecar_pair(
            staged_artifact,
            staged_sidecar,
            path,
            staging_dir,
        )


def _policy_outputs(planner: Any, model_obs: dict[str, np.ndarray]) -> dict[str, Any]:
    """Return pre-clip actor mean, model prediction, and critic value."""
    import torch

    policy = planner._model.policy
    obs_tensor, _ = policy.obs_to_tensor(model_obs)
    with torch.no_grad():
        actor_mean = policy._predict(obs_tensor, deterministic=True)
        value = policy.predict_values(obs_tensor)
    model_predict = planner._predict_action(model_obs)
    if model_predict is None:
        raise RuntimeError("PPO model prediction returned no action")
    return {
        "actor_mean": np.asarray(actor_mean.detach().cpu().numpy(), dtype=float).reshape(-1),
        "model_predict": np.asarray(model_predict, dtype=float).reshape(-1),
        "critic_value": float(value.detach().cpu().numpy().reshape(-1)[0]),
    }


def _channel_indices(model_obs: dict[str, np.ndarray]) -> tuple[int, int, int]:
    indices = np.asarray(model_obs["occupancy_grid_meta_channel_indices"], dtype=int).reshape(-1)
    if indices.size < 4:
        raise RuntimeError(f"occupancy-grid channel index metadata is incomplete: {indices}")
    obstacle_idx, pedestrian_idx, _robot_idx, combined_idx = (int(x) for x in indices[:4])
    if min(obstacle_idx, pedestrian_idx, combined_idx) < 0:
        raise RuntimeError(f"required occupancy-grid channel is absent: {indices.tolist()}")
    return obstacle_idx, pedestrian_idx, combined_idx


def _nearest_forward_obstacle(model_obs: dict[str, np.ndarray]) -> tuple[float, int]:
    """Return nearest occupied static cell in the 3 m-wide forward corridor."""
    grid = np.asarray(model_obs["occupancy_grid"], dtype=float)
    obstacle_idx, _pedestrian_idx, _combined_idx = _channel_indices(model_obs)
    occupied_yx = np.argwhere(grid[obstacle_idx] > 0.5)
    resolution = float(
        np.asarray(model_obs["occupancy_grid_meta_resolution"], dtype=float).reshape(-1)[0]
    )
    origin = np.asarray(model_obs["occupancy_grid_meta_origin"], dtype=float).reshape(-1)[:2]
    if occupied_yx.size == 0 or resolution <= 0 or not np.all(np.isfinite(origin)):
        return float("nan"), 0
    ego_xy = np.column_stack(
        (
            origin[0] + (occupied_yx[:, 1] + 0.5) * resolution,
            origin[1] + (occupied_yx[:, 0] + 0.5) * resolution,
        )
    )
    forward = ego_xy[(ego_xy[:, 0] >= 0.0) & (np.abs(ego_xy[:, 1]) <= 3.0)]
    if forward.size == 0:
        return float("nan"), 0
    distances = np.linalg.norm(forward, axis=1)
    return float(np.min(distances)), int(forward.shape[0])


def _without_static_geometry(model_obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Remove obstacle cells while preserving all non-geometry observations."""
    ablated = {key: np.asarray(value).copy() for key, value in model_obs.items()}
    grid = ablated["occupancy_grid"]
    obstacle_idx, pedestrian_idx, combined_idx = _channel_indices(ablated)
    grid[obstacle_idx].fill(0.0)
    grid[combined_idx] = grid[pedestrian_idx]
    return ablated


def _trace_seed(seed: int) -> tuple[list[dict[str, Any]], int]:
    env, _scenario, config = parent._build_diagnostic_env(seed)
    planner = parent._make_planner()
    observations: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    try:
        obs, _info = env.reset(seed=seed)
        for step in range(400):
            normalized = parent._normalize_runner_obs(obs)
            model_obs = planner._build_model_obs_dict(normalized)
            parent._assert_predictive_foresight_loaded(planner)
            canonical = _policy_outputs(planner, model_obs)
            command = planner._action_vec_to_dict_from_array(canonical["model_predict"])
            native_action = np.asarray(
                policy_command_to_env_action(
                    env=env,
                    config=config,
                    command=(command["v"], command["omega"]),
                ),
                dtype=float,
            )
            speed_before = np.asarray(env.simulator.robots[0].current_speed, dtype=float).copy()
            clearance_before = parent._min_obstacle_clearance(env)
            observations.append(
                {
                    "model_obs": deepcopy(model_obs),
                    "canonical": canonical,
                    "command": dict(command),
                    "native_action": native_action.copy(),
                    "speed_before": speed_before,
                    "clearance_before": clearance_before,
                }
            )
            obs, _reward, terminated, truncated, info = env.step(native_action)
            speed_after = np.asarray(env.simulator.robots[0].current_speed, dtype=float).copy()
            meta = dict(info.get("meta", {}))
            trace.append(
                {
                    "step": step,
                    "speed_after": speed_after,
                    "collision": bool(meta.get("is_obstacle_collision", False)),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )
            if terminated or truncated:
                break
    finally:
        env.close()
        planner.close()
    contact_step = next(
        (row["step"] for row in trace if row["collision"]),
        -1,
    )
    if contact_step < 0:
        raise RuntimeError(f"seed {seed} did not reproduce obstacle contact")
    if contact_step != len(observations) - 1:
        raise RuntimeError(
            f"seed {seed} contact alignment failed: contact={contact_step}, "
            f"observations={len(observations)}"
        )

    rows: list[dict[str, Any]] = []
    planner = parent._make_planner()
    try:
        # Predictive provenance is asserted on the actual rollout planner after
        # every model-observation build above.  This second planner is used only
        # for direct actor/critic evaluation of stored observations, so no
        # foresight inference occurs and a load-status assertion would be both
        # premature and unrelated to the ablation signal.
        for offset in OFFSETS:
            state_step = contact_step - offset
            if state_step < 0:
                raise RuntimeError(f"seed {seed} lacks contact-minus-{offset} state")
            state = observations[state_step]
            model_obs = state["model_obs"]
            canonical = state["canonical"]
            ablated = _policy_outputs(planner, _without_static_geometry(model_obs))
            nearest_obstacle_m, forward_obstacle_cells = _nearest_forward_obstacle(model_obs)
            command = state["command"]
            model_predict = canonical["model_predict"]
            actor_mean = canonical["actor_mean"]
            speed_after = trace[state_step]["speed_after"]
            rows.append(
                {
                    "seed": seed,
                    "contact_step": contact_step,
                    "offset_before_contact": offset,
                    "state_step": state_step,
                    "static_geometry_present": bool(
                        math.isfinite(nearest_obstacle_m) and forward_obstacle_cells > 0
                    ),
                    "nearest_forward_obstacle_m": nearest_obstacle_m,
                    "forward_obstacle_cells": forward_obstacle_cells,
                    "ground_truth_clearance_before_m": float(state["clearance_before"]),
                    "canonical_actor_mean_v": float(actor_mean[0]),
                    "canonical_actor_mean_omega": float(actor_mean[1]),
                    "canonical_model_predict_v": float(model_predict[0]),
                    "canonical_model_predict_omega": float(model_predict[1]),
                    "canonical_critic_value": float(canonical["critic_value"]),
                    "adapter_command_v": float(command["v"]),
                    "adapter_command_omega": float(command["omega"]),
                    "native_action_linear_accel": float(state["native_action"][0]),
                    "native_action_angular_accel": float(state["native_action"][1]),
                    "executed_linear_speed_before": float(state["speed_before"][0]),
                    "executed_linear_speed_after": float(speed_after[0]),
                    "executed_angular_speed_before": float(state["speed_before"][1]),
                    "executed_angular_speed_after": float(speed_after[1]),
                    "adapter_forward_intent_preserved": bool(
                        model_predict[0] > 0.0 and command["v"] > 0.0 and speed_after[0] > 0.0
                    ),
                    "geometry_zero_actor_mean_v": float(ablated["actor_mean"][0]),
                    "geometry_zero_actor_mean_omega": float(ablated["actor_mean"][1]),
                    "geometry_zero_model_predict_v": float(ablated["model_predict"][0]),
                    "geometry_zero_model_predict_omega": float(ablated["model_predict"][1]),
                    "geometry_zero_critic_value": float(ablated["critic_value"]),
                    "actor_mean_v_delta_geometry_zero_minus_canonical": float(
                        ablated["actor_mean"][0] - actor_mean[0]
                    ),
                    "critic_delta_geometry_zero_minus_canonical": float(
                        ablated["critic_value"] - canonical["critic_value"]
                    ),
                    "fallback_or_degraded": False,
                }
            )
    finally:
        planner.close()
    return rows, contact_step


def _all_finite_directional(rows: list[dict[str, Any]], key: str, *, direction: str) -> bool:
    """Return whether every row carries a finite intervention delta in one direction."""
    values: list[float] = []
    for row in rows:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            return False
        if not math.isfinite(value):
            return False
        values.append(value)
    if direction == "negative":
        return bool(values) and all(value < 0.0 for value in values)
    if direction == "positive":
        return bool(values) and all(value > 0.0 for value in values)
    raise ValueError(f"unsupported direction: {direction}")


def classify(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify the bounded matched-state signals without broader inference."""
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), []).append(row)
    seed_summaries: list[dict[str, Any]] = []
    for seed, seed_rows in sorted(by_seed.items()):
        ordered = sorted(seed_rows, key=lambda row: int(row["offset_before_contact"]), reverse=True)
        early, late = ordered[0], ordered[-1]
        seed_summaries.append(
            {
                "seed": seed,
                "contact_step": int(late["contact_step"]),
                "nearest_forward_obstacle_early_m": float(early["nearest_forward_obstacle_m"]),
                "nearest_forward_obstacle_late_m": float(late["nearest_forward_obstacle_m"]),
                "critic_value_early": float(early["canonical_critic_value"]),
                "critic_value_late": float(late["canonical_critic_value"]),
                "minimum_model_predict_v": min(
                    float(row["canonical_model_predict_v"]) for row in seed_rows
                ),
                "minimum_adapter_command_v": min(
                    float(row["adapter_command_v"]) for row in seed_rows
                ),
                "minimum_executed_speed_after": min(
                    float(row["executed_linear_speed_after"]) for row in seed_rows
                ),
                "critic_worsened_toward_contact": bool(
                    float(late["canonical_critic_value"]) < float(early["canonical_critic_value"])
                ),
            }
        )

    geometry_present = all(bool(row["static_geometry_present"]) for row in rows)
    adapter_preserved = all(bool(row["adapter_forward_intent_preserved"]) for row in rows)
    actor_forward = all(float(row["canonical_model_predict_v"]) > 0.5 for row in rows)
    critic_worsened = all(x["critic_worsened_toward_contact"] for x in seed_summaries)
    no_degraded = all(not bool(row["fallback_or_degraded"]) for row in rows)
    actor_geometry_response = _all_finite_directional(
        rows, "actor_mean_v_delta_geometry_zero_minus_canonical", direction="negative"
    )
    critic_geometry_response = _all_finite_directional(
        rows, "critic_delta_geometry_zero_minus_canonical", direction="positive"
    )
    if (
        geometry_present
        and adapter_preserved
        and actor_forward
        and critic_worsened
        and actor_geometry_response
        and critic_geometry_response
        and no_degraded
    ):
        verdict = "actor_value_response_mismatch_supported"
    else:
        verdict = "not_identifiable"
    return {
        "schema": "issue_9609_actor_response_summary.v1",
        "issue": ISSUE,
        "parent_issue": PARENT_ISSUE,
        "source_pr": PARENT_PR,
        "source_pr_head": PARENT_HEAD,
        "claim_boundary": CLAIM_BOUNDARY,
        "evidence_tier": "diagnostic-only",
        "result_classification": verdict,
        "row_count": len(rows),
        "seeds": list(SEEDS),
        "offsets_before_contact": list(OFFSETS),
        "mechanism_activation": {
            "activated": True if verdict.endswith("supported") else "unknown",
            "activation_count": len(rows) if verdict.endswith("supported") else "unknown",
            "changed_command_source": False,
            "changed_outcome": "unknown",
            "likely_failure_reason": (
                "At the tested states, static doorway geometry is present and the critic assigns "
                "increasingly poor value, while the actor and adapter preserve forward motion."
                if verdict.endswith("supported")
                else "The bounded signals do not separate the competing explanations."
            ),
        },
        "checks": {
            "static_geometry_present_all_rows": geometry_present,
            "adapter_forward_intent_preserved_all_rows": adapter_preserved,
            "actor_forward_all_rows": actor_forward,
            "critic_worsened_toward_contact_all_seeds": critic_worsened,
            "geometry_removal_lowers_actor_mean_v_all_rows": actor_geometry_response,
            "geometry_removal_raises_critic_value_all_rows": critic_geometry_response,
            "fallback_or_degraded_rows": sum(bool(row["fallback_or_degraded"]) for row in rows),
        },
        "seed_summaries": seed_summaries,
        "limitations": [
            "Three deterministic seeds and twelve matched states do not establish a population-level mechanism.",
            "Static-geometry removal is an out-of-distribution diagnostic intervention, not a safe policy alternative.",
            "Evaluation-time actor/critic outputs do not identify the training-time optimization cause.",
            "The probe identifies the decision boundary, not a policy repair.",
        ],
    }


def run(output_dir: Path) -> dict[str, Any]:
    """Run the three-seed diagnostic and write compact reviewed artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    contacts: dict[str, int] = {}
    for seed in SEEDS:
        seed_rows, contact_step = _trace_seed(seed)
        rows.extend(seed_rows)
        contacts[str(seed)] = contact_step
    if len(rows) != len(SEEDS) * len(OFFSETS):
        raise RuntimeError(f"expected 12 matched rows, observed {len(rows)}")

    # The inherited builder writes binding.json as a side effect. Keep that
    # intermediate file outside the durable evidence directory so a failure
    # cannot replace the published binding before its final sidecar is ready.
    repo_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(
        prefix=".issue9609-parent-binding-", dir=repo_root / "docs" / "context"
    ) as binding_staging_directory:
        binding = parent.build_binding(Path(binding_staging_directory), 0.99)
    binding.update(
        {
            "schema": "issue_9609_actor_response_binding.v1",
            "issue": ISSUE,
            "parent_issue": PARENT_ISSUE,
            "source_pr": PARENT_PR,
            "source_pr_head": PARENT_HEAD,
            "seeds": list(SEEDS),
            "offsets_before_contact": list(OFFSETS),
            "contact_steps": contacts,
            "command": REPRO_COMMAND,
            "claim_boundary": CLAIM_BOUNDARY,
            "artifact_provenance": "tracked-compact-evidence",
            "fallback_or_degraded_execution": False,
        }
    )
    # Parent binding includes a wall-clock generation field and the current worktree
    # head; neither belongs in this deterministic child packet.
    binding.pop("generated_at_utc", None)
    binding.pop("git_head", None)
    binding["distance_conventions_by_column"] = {
        "ground_truth_clearance_before_m": {
            "distance_convention": "surface_clearance",
            "definition": (
                "Shortest robot-center to static-wall-segment distance minus the robot radius; "
                "positive values mean separation between the robot footprint and wall."
            ),
            "producer": "parent._min_obstacle_clearance",
        },
        "nearest_forward_obstacle_m": {
            "distance_convention": "center_center",
            "definition": (
                "Euclidean distance in the ego occupancy grid from its robot-center origin to "
                "the center of the nearest occupied static cell in the 3 m forward corridor."
            ),
            "producer": "_nearest_forward_obstacle",
        },
    }

    binding_path = output_dir / "binding.json"
    table_path = output_dir / "matched_state_rows.csv"
    summary_path = output_dir / "mechanism_summary.json"
    _write_json_with_review_sidecar(binding_path, binding)
    write_distance_series_csv(
        table_path,
        rows,
        convention="surface_clearance",
        series_name="ground_truth_clearance_before_m",
    )
    summary = classify(rows)
    write_json(summary_path, summary)
    for path in (table_path, summary_path):
        _write_review_sidecar(path)
    return summary


def main() -> int:
    """Run the command-line diagnostic."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["result_classification"].endswith("supported") else 2


if __name__ == "__main__":
    raise SystemExit(main())
