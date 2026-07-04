"""Validate and dry-run the issue #4244 seven-arm training comparison matrix."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from robot_sf.training.offline_pretraining_manifest import (
    OFFLINE_POLICY_CHECKPOINT_MANIFEST_SCHEMA_VERSION,
    OFFLINE_TO_ONLINE_FINETUNE_MANIFEST_SCHEMA_VERSION,
)

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
MECHANISM_BREAKDOWN_SCHEMA_ISSUE = 4242
MECHANISM_BREAKDOWN_REQUIRED_FIELDS = {
    "mechanism_schema_version",
    "mechanism_label",
    "mechanism_confidence",
    "mechanism_evidence_mode",
    "mechanism_evidence_uri",
}
SAC_GATE = "sac_after_issue_4245_standalone_offline_pretraining"
SAC_GATE_REQUIRED_ISSUE = 4245
# Evidence pointers the SAC inclusion gate must consume from the merged issue
# #4245 standalone offline-pretraining lane (see docs/context/evidence/).
SAC_GATE_EVIDENCE_KEYS = (
    "provenance_chain",
    "pretrain_manifest_summary",
    "finetune_manifest_summary",
)
# Manifest schema versions the issue #4245 lane produces. Sourced directly from the
# canonical offline-pretraining manifest module so a schema bump there cannot silently
# drift from this gate (which would otherwise spuriously exclude valid #4245 evidence).
PRETRAIN_MANIFEST_SCHEMA = OFFLINE_POLICY_CHECKPOINT_MANIFEST_SCHEMA_VERSION
FINETUNE_MANIFEST_SCHEMA = OFFLINE_TO_ONLINE_FINETUNE_MANIFEST_SCHEMA_VERSION


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
    _validate_analysis_plan(comparison, repo_root=repo_root)
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


def _validate_analysis_plan(comparison: dict[str, Any], *, repo_root: Path) -> None:
    """Require the analysis plan to point at landed mechanism instrumentation."""
    analysis_plan = comparison.get("analysis_plan")
    if not isinstance(analysis_plan, dict):
        raise MatrixValidationError("comparison.analysis_plan must be a mapping")
    if analysis_plan.get("rank_table") is not True:
        raise MatrixValidationError("comparison.analysis_plan.rank_table must be true")
    if analysis_plan.get("confidence_intervals") != "bootstrap_95":
        raise MatrixValidationError(
            "comparison.analysis_plan.confidence_intervals must be bootstrap_95"
        )
    if analysis_plan.get("mechanism_breakdown_schema_issue") != MECHANISM_BREAKDOWN_SCHEMA_ISSUE:
        raise MatrixValidationError(
            "comparison.analysis_plan.mechanism_breakdown_schema_issue must be 4242"
        )
    note_path = str(analysis_plan.get("mechanism_breakdown_schema_note", ""))
    _resolve_repo_file(repo_root, note_path, label="mechanism_breakdown_schema_note")
    fields = analysis_plan.get("mechanism_breakdown_required_fields")
    if not isinstance(fields, list):
        raise MatrixValidationError(
            "comparison.analysis_plan.mechanism_breakdown_required_fields must be a list"
        )
    missing_fields = MECHANISM_BREAKDOWN_REQUIRED_FIELDS - {str(field) for field in fields}
    if missing_fields:
        raise MatrixValidationError(
            f"comparison.analysis_plan missing mechanism breakdown fields: {sorted(missing_fields)}"
        )


def _validate_inclusion_gates(matrix: ComparisonMatrix) -> dict[str, Any]:
    """Require the matrix-level gates that can exclude arms fail-closed."""
    gates = {
        str(gate.get("id")): gate
        for gate in matrix.payload["inclusion_gates"]
        if isinstance(gate, dict) and gate.get("id")
    }
    if SAC_GATE not in gates:
        raise MatrixValidationError(f"missing SAC inclusion gate {SAC_GATE!r}")
    if gates[SAC_GATE].get("required_issue") != SAC_GATE_REQUIRED_ISSUE:
        raise MatrixValidationError(f"{SAC_GATE} must require issue {SAC_GATE_REQUIRED_ISSUE}")
    _validate_sac_gate_evidence_pointers(gates[SAC_GATE])
    return gates


def _validate_sac_gate_evidence_pointers(gate: dict[str, Any]) -> None:
    """Require the SAC gate to name repository-relative issue #4245 evidence.

    The gate's declared policy is ``exclude_arm_with_reason``, so structurally
    missing evidence pointers fail matrix validation (schema drift), but the
    presence and contents of the evidence files are evaluated at dry-run time so
    an absent lane excludes the arm with a reason rather than failing the matrix.
    """
    evidence = gate.get("evidence")
    if not isinstance(evidence, dict):
        raise MatrixValidationError(f"{SAC_GATE} must declare an evidence mapping")
    for key in SAC_GATE_EVIDENCE_KEYS:
        value = evidence.get(key)
        if not isinstance(value, str) or not value:
            raise MatrixValidationError(f"{SAC_GATE}.evidence.{key} must be a repo-relative path")
        _validate_relative_path(value, label=f"{SAC_GATE}.evidence.{key}")
        _validate_no_host_state(value)


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
    entry: dict[str, Any] = {
        "id": arm.arm_id,
        "display_name": arm.display_name,
        "algorithm_family": arm.algorithm_family,
        "training_config": arm.training_config,
        "budget": matrix.shared_budget,
        "required_gates": list(arm.required_gates),
        "smoke_contract_manifest": arm.smoke_contract_manifest,
    }
    if arm.arm_id == "offline_online_sac":
        evidence = _sac_gate_config(matrix).get("evidence") or {}
        gate = evaluate_sac_offline_pretraining_gate(evidence, repo_root=repo_root)
        entry["offline_pretraining_gate"] = gate
        if not gate["passed"]:
            reasons.append(f"issue #4245 offline-pretraining gate not satisfied: {gate['reason']}")
    entry["status"] = "eligible_for_queue_plan" if not reasons else "excluded_by_inclusion_gate"
    entry["excluded_reasons"] = reasons
    entry["queue_entry"] = {
        "issue": 4244,
        "arm_id": arm.arm_id,
        "status": "planned_not_submitted",
        "submit_in_this_pr": False,
        "training_config": arm.training_config,
        "output_root": f"output/training/comparison_matrix_issue_4244/{arm.arm_id}",
    }
    return entry


def _sac_gate_config(matrix: ComparisonMatrix) -> dict[str, Any]:
    """Return the SAC inclusion-gate mapping, or an empty mapping if absent."""
    for gate in matrix.payload["inclusion_gates"]:
        if isinstance(gate, dict) and gate.get("id") == SAC_GATE:
            return gate
    return {}


def evaluate_sac_offline_pretraining_gate(
    evidence: dict[str, Any], *, repo_root: Path
) -> dict[str, Any]:
    """Consume the issue #4245 offline-pretraining provenance/manifest evidence.

    The gate passes only when the standalone offline-pretraining lane's checked-in
    provenance chain and manifest summaries exist, declare issue #4245 for
    algorithm ``sac`` with the expected manifest schema versions, and link
    together by checkpoint SHA (offline checkpoint inherited by the fine-tune,
    both recorded in the provenance chain). Missing, malformed, or inconsistent
    evidence fails closed with a specific reason rather than silently including
    or dropping the arm.

    Returns:
        Dry-run gate status block with ``passed`` flag, required issue, and either
        the verified checkpoint SHAs or the failure ``reason``.
    """
    block: dict[str, Any] = {
        "required_issue": SAC_GATE_REQUIRED_ISSUE,
        "evidence": {key: evidence.get(key) for key in SAC_GATE_EVIDENCE_KEYS},
    }
    try:
        chain = _load_sac_gate_json(repo_root, evidence, "provenance_chain")
        pretrain = _load_sac_gate_json(repo_root, evidence, "pretrain_manifest_summary")
        finetune = _load_sac_gate_json(repo_root, evidence, "finetune_manifest_summary")
    except MatrixValidationError as exc:
        return {**block, "passed": False, "reason": str(exc)}

    reason = _sac_gate_inconsistency(chain, pretrain, finetune)
    if reason is not None:
        return {**block, "passed": False, "reason": reason}
    return {
        **block,
        "passed": True,
        "offline_checkpoint_sha256": pretrain["checkpoint_sha256"],
        "finetune_checkpoint_sha256": finetune["checkpoint_sha256"],
    }


def _sac_gate_inconsistency(
    chain: dict[str, Any], pretrain: dict[str, Any], finetune: dict[str, Any]
) -> str | None:
    """Return the first #4245 evidence contract violation, or None when consistent."""
    for name, payload in (
        ("provenance_chain", chain),
        ("pretrain_manifest_summary", pretrain),
        ("finetune_manifest_summary", finetune),
    ):
        if payload.get("issue") != SAC_GATE_REQUIRED_ISSUE:
            return f"{name} must declare issue {SAC_GATE_REQUIRED_ISSUE}"
    if pretrain.get("schema_version") != PRETRAIN_MANIFEST_SCHEMA:
        return f"pretrain manifest schema must be {PRETRAIN_MANIFEST_SCHEMA!r}"
    if finetune.get("schema_version") != FINETUNE_MANIFEST_SCHEMA:
        return f"finetune manifest schema must be {FINETUNE_MANIFEST_SCHEMA!r}"
    if pretrain.get("algorithm") != "sac" or finetune.get("algorithm") != "sac":
        return "offline-pretraining evidence must be for algorithm 'sac'"
    offline_sha = pretrain.get("checkpoint_sha256")
    if not offline_sha or chain.get("offline_checkpoint_sha256") != offline_sha:
        return "provenance chain offline checkpoint SHA does not match pretrain manifest"
    if finetune.get("parent_checkpoint_sha256") != offline_sha:
        return "finetune manifest does not inherit the offline checkpoint SHA"
    finetune_sha = finetune.get("checkpoint_sha256")
    if not finetune_sha or chain.get("finetune_checkpoint_sha256") != finetune_sha:
        return "provenance chain finetune checkpoint SHA does not match finetune manifest"
    return None


def _load_sac_gate_json(repo_root: Path, evidence: dict[str, Any], key: str) -> dict[str, Any]:
    """Load one SAC-gate evidence JSON file, failing closed on missing/malformed data."""
    value = evidence.get(key)
    if not isinstance(value, str) or not value:
        raise MatrixValidationError(f"missing evidence pointer {key}")
    path = _resolve_repo_file(repo_root, value, label=f"sac_gate.{key}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MatrixValidationError(f"{key} is not readable JSON: {value}") from exc
    if not isinstance(payload, dict):
        raise MatrixValidationError(f"{key} must be a JSON object: {value}")
    return payload


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
