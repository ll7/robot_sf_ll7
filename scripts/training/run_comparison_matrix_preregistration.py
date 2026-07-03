"""Validate and dry-run the issue #4244 seven-arm training comparison matrix."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = "training_comparison_matrix_preregistration.v1"
EXPECTED_ARM_IDS = (
    "ppo_baseline",
    "recurrent_ppo_lstm",
    "ppo_mamba",
    "ppo_lagrangian",
    "diffusion_policy",
    "qr_dqn",
    "offline_online_sac",
)
REQUIRED_METRICS = {"success_rate", "collision_rate", "snqi"}
REQUIRED_STOP_RULES = {
    "max_total_timesteps",
    "max_walltime_per_arm",
    "early_stop_allowed",
    "failed_smoke_contract_policy",
    "missing_required_gate_policy",
}
SAC_GATE = "sac_after_issue_4245_standalone_offline_pretraining"


class MatrixValidationError(ValueError):
    """Raised when the comparison matrix cannot be preregistered safely."""


@dataclass(frozen=True)
class MatrixArm:
    """One predeclared training arm in the issue #4244 matrix."""

    arm_id: str
    display_name: str
    algorithm_family: str
    training_config: str
    smoke_contract_manifest: str
    budget_ref: str
    required_gates: tuple[str, ...]

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> MatrixArm:
        """Build an arm from YAML and require the fields the runner consumes."""
        missing = [
            field
            for field in (
                "id",
                "display_name",
                "algorithm_family",
                "training_config",
                "smoke_contract_manifest",
                "budget_ref",
                "required_gates",
            )
            if payload.get(field) in (None, "")
        ]
        if missing:
            raise MatrixValidationError(
                f"arm {payload.get('id', '<unknown>')}: missing {missing[0]}"
            )
        gates = payload["required_gates"]
        if not isinstance(gates, list) or not all(isinstance(gate, str) for gate in gates):
            raise MatrixValidationError(
                f"arm {payload['id']}: required_gates must be a string list"
            )
        return cls(
            arm_id=str(payload["id"]),
            display_name=str(payload["display_name"]),
            algorithm_family=str(payload["algorithm_family"]),
            training_config=str(payload["training_config"]),
            smoke_contract_manifest=str(payload["smoke_contract_manifest"]),
            budget_ref=str(payload["budget_ref"]),
            required_gates=tuple(gates),
        )


@dataclass(frozen=True)
class ComparisonMatrix:
    """Validated issue #4244 comparison matrix contract."""

    path: Path
    payload: dict[str, Any]
    arms: tuple[MatrixArm, ...]

    @property
    def shared_budget(self) -> dict[str, Any]:
        """Return the shared budget every arm must reference."""
        return self.payload["comparison"]["shared_budget"]


def load_matrix(
    config_path: Path | str, *, repo_root: Path | str | None = None
) -> ComparisonMatrix:
    """Load and validate a preregistered comparison matrix."""
    path = Path(config_path)
    root = Path(repo_root) if repo_root is not None else _discover_repo_root(path)
    if not path.is_absolute():
        path = root / path
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise MatrixValidationError(f"cannot read matrix config: {exc}") from exc
    except yaml.YAMLError as exc:
        raise MatrixValidationError(f"invalid matrix YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise MatrixValidationError("matrix config must be a mapping")
    _validate_top_level(raw)
    arms = tuple(MatrixArm.from_mapping(arm) for arm in raw["arms"])
    matrix = ComparisonMatrix(path=path, payload=raw, arms=arms)
    _validate_arm_roster(matrix)
    _validate_contract_fields(matrix, repo_root=root)
    return matrix


def build_dry_run_manifest(
    matrix: ComparisonMatrix,
    *,
    repo_root: Path | str | None = None,
    limit_arms: int | None = None,
) -> dict[str, Any]:
    """Build a CPU-only dry-run manifest without executing training or submission."""
    root = Path(repo_root) if repo_root is not None else _discover_repo_root(matrix.path)
    arms = matrix.arms[:limit_arms] if limit_arms is not None else matrix.arms
    return {
        "schema_version": "training_comparison_matrix_dry_run.v1",
        "issue": matrix.payload["issue"],
        "source_config": _relative_to_repo(matrix.path, repo_root=root),
        "claim_boundary": matrix.payload["claim_boundary"],
        "training_executed": False,
        "slurm_or_gpu_submitted": False,
        "selected_arm_count": len(arms),
        "total_arm_count": len(matrix.arms),
        "metrics": matrix.payload["comparison"]["metrics"],
        "stop_rules": matrix.payload["comparison"]["stop_rules"],
        "arms": [_dry_run_arm(arm, matrix=matrix, repo_root=root) for arm in arms],
    }


def write_dry_run_manifest(manifest: dict[str, Any], output_path: Path | str) -> Path:
    """Write the dry-run manifest as deterministic JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _validate_top_level(raw: dict[str, Any]) -> None:
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise MatrixValidationError(
            f"expected schema_version {SCHEMA_VERSION!r}, found {raw.get('schema_version')!r}"
        )
    if raw.get("issue") != 4244:
        raise MatrixValidationError("matrix issue must be 4244")
    comparison = raw.get("comparison")
    if not isinstance(comparison, dict):
        raise MatrixValidationError("comparison must be a mapping")
    arms = raw.get("arms")
    if not isinstance(arms, list):
        raise MatrixValidationError("arms must be a list")
    gates = raw.get("inclusion_gates")
    if not isinstance(gates, list):
        raise MatrixValidationError("inclusion_gates must be a list")


def _validate_arm_roster(matrix: ComparisonMatrix) -> None:
    configured = tuple(arm.arm_id for arm in matrix.arms)
    if configured != EXPECTED_ARM_IDS:
        raise MatrixValidationError(
            f"arms must be the predeclared seven-arm roster in order: {EXPECTED_ARM_IDS}"
        )
    expected_count = matrix.payload["comparison"].get("arms_expected")
    if expected_count != len(EXPECTED_ARM_IDS):
        raise MatrixValidationError(f"comparison.arms_expected must be {len(EXPECTED_ARM_IDS)}")
    if len(set(configured)) != len(configured):
        raise MatrixValidationError("arm ids must be unique")


def _validate_contract_fields(matrix: ComparisonMatrix, *, repo_root: Path) -> None:
    comparison = matrix.payload["comparison"]
    _validate_metrics(comparison)
    _validate_stop_rules(comparison)
    gates = _validate_inclusion_gates(matrix)
    _validate_arms(matrix, gates=gates)
    _validate_no_host_state(str(comparison))
    _validate_existing_source_paths(matrix, repo_root=repo_root)


def _validate_metrics(comparison: dict[str, Any]) -> None:
    """Require the predeclared metric set used for ranking and diagnostics."""
    metrics = comparison.get("metrics")
    if not isinstance(metrics, dict):
        raise MatrixValidationError("comparison.metrics must be a mapping")
    metric_names = set()
    for values in metrics.values():
        if not isinstance(values, list):
            raise MatrixValidationError("comparison.metrics values must be lists")
        metric_names.update(str(value) for value in values)
    missing_metrics = REQUIRED_METRICS - metric_names
    if missing_metrics:
        raise MatrixValidationError(f"missing required metrics: {sorted(missing_metrics)}")


def _validate_stop_rules(comparison: dict[str, Any]) -> None:
    """Require explicit stop rules before queue planning."""
    stop_rules = comparison.get("stop_rules")
    if not isinstance(stop_rules, dict):
        raise MatrixValidationError("comparison.stop_rules must be a mapping")
    missing_stop_rules = REQUIRED_STOP_RULES - set(stop_rules)
    if missing_stop_rules:
        raise MatrixValidationError(f"missing required stop rules: {sorted(missing_stop_rules)}")


def _validate_inclusion_gates(matrix: ComparisonMatrix) -> dict[str, Any]:
    """Require the matrix-level gates that can exclude arms fail-closed."""
    gates = {
        str(gate.get("id")): gate
        for gate in matrix.payload["inclusion_gates"]
        if isinstance(gate, dict) and gate.get("id")
    }
    if SAC_GATE not in gates:
        raise MatrixValidationError(f"missing SAC inclusion gate {SAC_GATE!r}")
    if gates[SAC_GATE].get("required_issue") != 4245:
        raise MatrixValidationError(f"{SAC_GATE} must require issue 4245")
    return gates


def _validate_arms(matrix: ComparisonMatrix, *, gates: dict[str, Any]) -> None:
    """Validate per-arm references to shared budget and inclusion gates."""
    for arm in matrix.arms:
        _validate_relative_path(arm.training_config, label=f"{arm.arm_id}.training_config")
        _validate_relative_path(
            arm.smoke_contract_manifest,
            label=f"{arm.arm_id}.smoke_contract_manifest",
        )
        if arm.budget_ref != "shared_budget":
            raise MatrixValidationError(f"{arm.arm_id}: budget_ref must be shared_budget")
        if "identical_budget_check_passes" not in arm.required_gates:
            raise MatrixValidationError(f"{arm.arm_id}: missing identical budget gate")
        if "smoke_contract_manifest_passes" not in arm.required_gates:
            raise MatrixValidationError(f"{arm.arm_id}: missing smoke manifest gate")
        if arm.arm_id == "offline_online_sac" and SAC_GATE not in arm.required_gates:
            raise MatrixValidationError("offline_online_sac must require the issue #4245 SAC gate")
        unknown_gates = set(arm.required_gates) - set(gates)
        if unknown_gates:
            raise MatrixValidationError(f"{arm.arm_id}: unknown gates {sorted(unknown_gates)}")
        _validate_no_host_state(arm.training_config)
        _validate_no_host_state(arm.smoke_contract_manifest)


def _validate_existing_source_paths(matrix: ComparisonMatrix, *, repo_root: Path) -> None:
    """Require stable checked-in source paths, while allowing named placeholder lanes."""
    scenario_config = matrix.shared_budget.get("scenario_config")
    if not isinstance(scenario_config, str):
        raise MatrixValidationError(f"shared_budget.scenario_config not found: {scenario_config}")
    _resolve_repo_file(repo_root, scenario_config, label="shared_budget.scenario_config")
    for arm in matrix.arms:
        try:
            _resolve_repo_file(
                repo_root, arm.training_config, label=f"{arm.arm_id}.training_config"
            )
            continue
        except MatrixValidationError as exc:
            if "/placeholders/" not in arm.training_config or "not found" not in str(exc):
                raise
        if "/placeholders/" not in arm.training_config:
            raise MatrixValidationError(
                f"{arm.arm_id}: training_config not found: {arm.training_config}"
            )


def _dry_run_arm(arm: MatrixArm, *, matrix: ComparisonMatrix, repo_root: Path) -> dict[str, Any]:
    reasons = []
    manifest_path = repo_root / arm.smoke_contract_manifest
    if not manifest_path.is_file():
        reasons.append(f"smoke contract manifest not found: {arm.smoke_contract_manifest}")
    elif not _smoke_manifest_passes(manifest_path):
        reasons.append(f"smoke contract manifest did not pass: {arm.smoke_contract_manifest}")
    if arm.arm_id == "offline_online_sac":
        reasons.append("requires standalone offline-pretraining gate from issue #4245")
    status = "eligible_for_queue_plan" if not reasons else "excluded_by_inclusion_gate"
    return {
        "id": arm.arm_id,
        "display_name": arm.display_name,
        "algorithm_family": arm.algorithm_family,
        "training_config": arm.training_config,
        "budget": matrix.shared_budget,
        "required_gates": list(arm.required_gates),
        "smoke_contract_manifest": arm.smoke_contract_manifest,
        "status": status,
        "excluded_reasons": reasons,
        "queue_entry": {
            "issue": 4244,
            "arm_id": arm.arm_id,
            "status": "planned_not_submitted",
            "submit_in_this_pr": False,
            "training_config": arm.training_config,
            "output_root": f"output/training/comparison_matrix_issue_4244/{arm.arm_id}",
        },
    }


def _smoke_manifest_passes(manifest_path: Path) -> bool:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    return payload.get("status") in {"passed", "pass", "ok"} or payload.get("passed") is True


def _validate_relative_path(value: str, *, label: str) -> None:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise MatrixValidationError(f"{label} must be a repository-relative path")


def _resolve_repo_file(repo_root: Path, value: str, *, label: str) -> Path:
    _validate_relative_path(value, label=label)
    root = repo_root.resolve()
    raw_path = root / value
    resolved = raw_path.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise MatrixValidationError(f"{label} escapes repository root: {value}") from exc
    if _has_symlink_component(raw_path, root=root):
        raise MatrixValidationError(f"{label} must not traverse symlinks: {value}")
    if not resolved.is_file():
        raise MatrixValidationError(f"{label} not found: {value}")
    return resolved


def _has_symlink_component(path: Path, *, root: Path) -> bool:
    try:
        relative_parts = path.relative_to(root).parts
    except ValueError:
        return True
    current = root
    for part in relative_parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _validate_no_host_state(value: str) -> None:
    forbidden = ("imech", "auxme-imech", "licca")
    if any(token in value.lower() for token in forbidden):
        raise MatrixValidationError("matrix config must not encode transient host or queue state")


def _relative_to_repo(path: Path, *, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _discover_repo_root(path: Path) -> Path:
    start = path if path.is_dir() else path.parent
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists() and (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd()


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Comparison matrix YAML path.")
    parser.add_argument("--repo-root", default=".", help="Repository root for relative paths.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write a CPU-only dry-run manifest without executing training.",
    )
    parser.add_argument("--limit-arms", type=int, default=None, help="Limit dry-run arm count.")
    parser.add_argument(
        "--output",
        default="output/training/comparison_matrix_issue_4244/dry_run_manifest.json",
        help="Dry-run manifest output path.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = build_arg_parser().parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    matrix = load_matrix(args.config, repo_root=repo_root)
    if args.dry_run:
        manifest = build_dry_run_manifest(matrix, repo_root=repo_root, limit_arms=args.limit_arms)
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = repo_root / output_path
        write_dry_run_manifest(manifest, output_path)
        print(f"dry_run_manifest={_relative_to_repo(output_path, repo_root=repo_root)}")
    else:
        print(f"validated_arms={len(matrix.arms)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
