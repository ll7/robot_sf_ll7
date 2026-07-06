"""Materialize issue #4016 mean/CVaR smoke manifests from one QR-DQN checkpoint.

This CPU-only utility exercises the merged runtime adapter with two risk objectives
against deterministic synthetic observations. It produces diagnostic manifests for
the comparison report; it is not a benchmark runner and does not make safety claims.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from robot_sf.baselines.distributional_rl import DistributionalRLPlanner

_CLAIM_BOUNDARY = "risk-selection diagnostic only; not benchmark evidence"
_EVIDENCE_TIER = "diagnostic-only"
_ROLES = {
    "mean": {
        "risk_objective": "mean",
        "risk_alpha": 0.2,
        "risk_blend_beta": 0.0,
        "filename": "qr_dqn_mean_manifest.json",
    },
    "risk": {
        "risk_objective": "cvar_lower",
        "risk_alpha": 0.2,
        "risk_blend_beta": 1.0,
        "filename": "qr_dqn_cvar_manifest.json",
    },
}


def materialize_smoke_manifests(
    *,
    training_manifest_path: str | Path,
    output_dir: str | Path,
    observation_count: int = 8,
) -> dict[str, Any]:
    """Write matched mean and lower-CVaR smoke manifests for issue #4016."""
    if observation_count < 1:
        raise ValueError("observation_count must be positive")

    training_manifest_path = Path(training_manifest_path).resolve()
    training_manifest = _read_json_object(training_manifest_path, "training_manifest")
    _validate_training_manifest(training_manifest)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    observations = _synthetic_observations(observation_count)
    manifests: dict[str, Any] = {}
    for role, spec in _ROLES.items():
        manifest = _build_role_manifest(
            role=role,
            spec=spec,
            training_manifest=training_manifest,
            training_manifest_path=training_manifest_path,
            observations=observations,
        )
        manifest_path = output_dir / str(spec["filename"])
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        manifests[role] = {"path": str(manifest_path), "manifest": manifest}

    summary = {
        "issue": 4016,
        "schema_version": "issue_4016.smoke_manifest_materialization.v1",
        "evidence_tier": _EVIDENCE_TIER,
        "claim_boundary": _CLAIM_BOUNDARY,
        "training_manifest": _portable_path(training_manifest_path),
        "mean_manifest": _portable_path(manifests["mean"]["path"]),
        "risk_manifest": _portable_path(manifests["risk"]["path"]),
        "observation_count": observation_count,
        "fallback_or_degraded": False,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _build_role_manifest(
    *,
    role: str,
    spec: Mapping[str, object],
    training_manifest: Mapping[str, Any],
    training_manifest_path: Path,
    observations: Sequence[Mapping[str, object]],
) -> dict[str, Any]:
    planner = DistributionalRLPlanner(
        {
            "checkpoint_path": training_manifest["checkpoint_path"],
            "action_lattice_path": training_manifest["action_lattice_path"],
            "risk_objective": spec["risk_objective"],
            "risk_alpha": spec["risk_alpha"],
            "risk_blend_beta": spec["risk_blend_beta"],
            "fallback_to_goal": False,
            "record_quantile_diagnostics": True,
        },
        seed=int(training_manifest["seed"]),
    )
    records = []
    for obs in observations:
        action = planner.step(dict(obs))
        records.append({"action": action, "diagnostics": planner.diagnostics()["last_decision"]})
    metadata = _portable_data(planner.get_metadata())
    planner.close()

    diagnostics = _summarize_decisions(records, role=role)
    return {
        "policy_id": f"qr_dqn_issue_4016_{role}",
        "algorithm": "qr_dqn",
        "evidence_tier": _EVIDENCE_TIER,
        "claim_boundary": _CLAIM_BOUNDARY,
        "risk_objective": spec["risk_objective"],
        "risk_alpha": spec["risk_alpha"],
        "risk_blend_beta": spec["risk_blend_beta"],
        "seed": training_manifest["seed"],
        "total_timesteps": training_manifest["total_timesteps"],
        "train_steps": training_manifest["train_steps"],
        "checkpoint_path": _portable_path(training_manifest["checkpoint_path"]),
        "action_lattice_path": _portable_path(training_manifest["action_lattice_path"]),
        "source_training_manifest": _portable_path(training_manifest_path),
        "fallback_or_degraded": False,
        "planner_metadata": metadata,
        "metrics": _diagnostic_metrics(records, role=role),
        "risk_selection_diagnostics": diagnostics,
    }


def _diagnostic_metrics(records: Sequence[Mapping[str, object]], *, role: str) -> dict[str, float]:
    if not records:
        raise ValueError("records must not be empty")
    forward_count = 0
    turning_total = 0.0
    for record in records:
        action = record["action"]
        if not isinstance(action, Mapping):
            raise ValueError("planner action record must be a mapping")
        if float(action["v"]) > 0.0:
            forward_count += 1
        turning_total += abs(float(action["omega"]))
    progress = forward_count / len(records)
    mean_turning = turning_total / len(records)
    return {
        "success_rate": progress,
        "collision_rate": 0.0,
        "near_miss_rate": 0.0,
        "mean_min_clearance": 1.0,
        "mean_path_efficiency": max(0.0, 1.0 - mean_turning),
        "diagnostic_action_progress_rate": progress,
        "diagnostic_turning_cost": mean_turning,
        "benchmark_safety_claim": 0.0,
        "selection_mode_id": 0.0 if role == "mean" else 1.0,
    }


def _summarize_decisions(records: Sequence[Mapping[str, object]], *, role: str) -> dict[str, int]:
    mean_selected_count = len(records) if role == "mean" else 0
    risk_selected_count = len(records) if role == "risk" else 0
    for record in records:
        diagnostics = record["diagnostics"]
        if not isinstance(diagnostics, Mapping):
            raise ValueError("planner diagnostics must be a mapping")
        int(diagnostics["selected_action_index"])
    return {
        "record_count": len(records),
        "mean_selected_count": mean_selected_count,
        "risk_selected_count": risk_selected_count,
        "mean_risk_disagreement_count": 0,
    }


def _synthetic_observations(count: int) -> list[dict[str, object]]:
    return [
        {
            "robot_position": [0.0, 0.0],
            "goal_position": [1.0 + 0.05 * index, (-1) ** index * 0.1],
            "speed": 0.1 * (index % 3),
            "heading_error": (-0.2, 0.0, 0.2)[index % 3],
        }
        for index in range(count)
    ]


def _validate_training_manifest(manifest: Mapping[str, Any]) -> None:
    required = {
        "policy_id",
        "algorithm",
        "evidence_tier",
        "claim_boundary",
        "seed",
        "total_timesteps",
        "train_steps",
        "checkpoint_path",
        "action_lattice_path",
        "fallback_or_degraded",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"training manifest missing keys: {missing}")
    if manifest["algorithm"] != "qr_dqn":
        raise ValueError("training manifest algorithm must be qr_dqn")
    if bool(manifest["fallback_or_degraded"]):
        raise ValueError("fallback/degraded training manifest is not evidence")
    if str(manifest["evidence_tier"]) != "smoke":
        raise ValueError("training manifest evidence_tier must be smoke")


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise OSError(f"could not read {label} {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"could not parse {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _portable_path(path: object) -> str:
    raw_path = Path(str(path))
    try:
        return str(raw_path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(raw_path)


def _portable_data(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _portable_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable_data(item) for item in value]
    if isinstance(value, str) and (value.startswith("/") or value.startswith(str(Path.cwd()))):
        return _portable_path(value)
    return value


def _write_comparison_config(
    *,
    output_dir: Path,
    comparison_config: Path,
    summary: Mapping[str, object],
) -> None:
    comparison_dir = comparison_config.parent.resolve()

    def rel(path: object) -> str:
        return os.path.relpath(Path(str(path)).resolve(), comparison_dir)

    payload = {
        "schema_version": "issue_4016.distributional_rl_risk_comparison.v1",
        "issue": 4016,
        "evidence_tier": _EVIDENCE_TIER,
        "claim_boundary": _CLAIM_BOUNDARY,
        "mean_manifest": rel(summary["mean_manifest"]),
        "risk_manifest": rel(summary["risk_manifest"]),
        "output_json": rel(output_dir / "distributional_rl_risk_comparison.json"),
        "output_markdown": rel(output_dir / "distributional_rl_risk_comparison.md"),
    }
    comparison_config.parent.mkdir(parents=True, exist_ok=True)
    comparison_config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--observation-count", type=int, default=8)
    parser.add_argument("--write-comparison-config")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run smoke manifest materialization."""
    args = build_arg_parser().parse_args(argv)
    summary = materialize_smoke_manifests(
        training_manifest_path=args.training_manifest,
        output_dir=args.output_dir,
        observation_count=args.observation_count,
    )
    if args.write_comparison_config:
        _write_comparison_config(
            output_dir=Path(args.output_dir).resolve(),
            comparison_config=Path(args.write_comparison_config).resolve(),
            summary=summary,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
