#!/usr/bin/env python3
"""Validate the policy-search candidate registry shape.

The checker is intentionally metadata-only: it verifies that future agents can route
candidate execution and evidence interpretation without running any candidate.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import yaml

DEFAULT_REGISTRY = Path("docs/context/policy_search/candidate_registry.yaml")
IMPLEMENTED_STATUSES = {"implemented", "experimental_spike", "prototype"}
SLURM_STATUSES = {"slurm_handoff_required"}
LEARNED_FAMILIES = {
    "learned_auxiliary_cost",
    "learned_policy_network",
    "learned_style_value_scorer",
}


@dataclass(frozen=True)
class RegistryIssue:
    """One policy-search registry validation issue."""

    path: str
    message: str


def _load_yaml(path: Path) -> Any:
    """Load a YAML file."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _is_missing(value: Any) -> bool:
    """Return whether a value is absent for registry-contract purposes."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def _repo_root_for(path: Path) -> Path:
    """Return the nearest repository root, falling back to the registry directory."""
    for parent in [path.resolve().parent, *path.resolve().parents]:
        if (parent / ".git").exists():
            return parent
    return path.resolve().parent


def _resolve_repo_path(repo_root: Path, raw_path: Any) -> Path | None:
    """Resolve a repository-relative path value when it is a string."""
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    path = Path(raw_path.strip())
    return path if path.is_absolute() else repo_root / path


def _require_fields(
    issues: list[RegistryIssue],
    row: dict[str, Any],
    *,
    prefix: str,
    fields: set[str],
) -> None:
    """Require non-empty fields in a registry row."""
    for field in sorted(fields):
        if _is_missing(row.get(field)):
            issues.append(RegistryIssue(f"{prefix}.{field}", "is required"))


def _validate_path_field(
    issues: list[RegistryIssue],
    row: dict[str, Any],
    *,
    prefix: str,
    field: str,
    repo_root: Path,
) -> None:
    """Validate that a path field points at an existing repository file."""
    path = _resolve_repo_path(repo_root, row.get(field))
    if path is None:
        return
    if not path.exists():
        issues.append(RegistryIssue(f"{prefix}.{field}", f"path does not exist: {row[field]}"))


def _parse_registry_date(value: Any) -> date | None:
    """Parse a YAML date or ISO date string."""
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip())
        except ValueError:
            return None
    return None


def _validate_freshness(
    payload: dict[str, Any],
    *,
    issues: list[RegistryIssue],
    as_of: date,
    default_max_age_days: int,
) -> None:
    """Validate the top-level registry freshness convention."""
    updated_at = _parse_registry_date(payload.get("updated_at"))
    if updated_at is None:
        issues.append(RegistryIssue("updated_at", "must be an ISO date"))
        return
    if updated_at > as_of:
        issues.append(RegistryIssue("updated_at", f"must not be in the future relative to {as_of}"))
        return

    freshness = payload.get("freshness")
    freshness_note = ""
    max_age_days = default_max_age_days
    if isinstance(freshness, dict):
        freshness_note = str(freshness.get("stale_reason") or "").strip()
        raw_max_age = freshness.get("max_age_days")
        if isinstance(raw_max_age, int) and raw_max_age > 0:
            max_age_days = raw_max_age
        elif raw_max_age is not None:
            issues.append(RegistryIssue("freshness.max_age_days", "must be a positive integer"))
    elif freshness is not None:
        issues.append(RegistryIssue("freshness", "must be a mapping when present"))

    age_days = (as_of - updated_at).days
    if age_days > max_age_days and not freshness_note:
        issues.append(
            RegistryIssue(
                "updated_at",
                (
                    f"is {age_days} days old; update it or record "
                    "freshness.stale_reason for an intentionally stale registry"
                ),
            )
        )


def _current_utc_date() -> date:
    """Return today's date in UTC for freshness checks."""
    return datetime.now(UTC).date()


def _load_known_names(
    payload: dict[str, Any],
    *,
    repo_root: Path,
) -> None:
    """Load known promotion gates and stages into private payload keys."""
    gates_path = _resolve_repo_path(repo_root, payload.get("promotion_gates"))
    if gates_path is not None and gates_path.exists():
        gates_payload = _load_yaml(gates_path)
        gates = gates_payload.get("gates") if isinstance(gates_payload, dict) else None
        if isinstance(gates, dict):
            payload["_known_promotion_gates"] = set(gates)

    funnel_path = _resolve_repo_path(repo_root, payload.get("funnel_config"))
    if funnel_path is not None and funnel_path.exists():
        funnel_payload = _load_yaml(funnel_path)
        stages = funnel_payload.get("stages") if isinstance(funnel_payload, dict) else None
        if isinstance(stages, dict):
            payload["_known_stages"] = set(stages)


def _validate_top_level(
    payload: Any,
    *,
    issues: list[RegistryIssue],
    registry_path: Path,
    repo_root: Path,
    as_of: date,
    default_max_age_days: int,
) -> dict[str, Any]:
    """Validate top-level fields and return candidate mapping when present."""
    if not isinstance(payload, dict):
        issues.append(RegistryIssue("registry", "must be a mapping"))
        return {}
    if payload.get("version") != 1:
        issues.append(RegistryIssue("version", "must be 1"))
    _validate_freshness(
        payload,
        issues=issues,
        as_of=as_of,
        default_max_age_days=default_max_age_days,
    )
    for field in ("funnel_config", "promotion_gates"):
        if _is_missing(payload.get(field)):
            issues.append(RegistryIssue(field, "is required"))
        else:
            path = _resolve_repo_path(repo_root, payload[field])
            if path is not None and not path.exists():
                issues.append(RegistryIssue(field, f"path does not exist: {payload[field]}"))

    candidates = payload.get("candidates")
    if not isinstance(candidates, dict) or not candidates:
        issues.append(RegistryIssue("candidates", "must be a non-empty mapping"))
        return {}

    _load_known_names(payload, repo_root=repo_root)
    del registry_path
    return dict(candidates)


def _validate_required_stages(
    issues: list[RegistryIssue],
    row: dict[str, Any],
    *,
    prefix: str,
    known_stages: set[str],
) -> None:
    """Validate the required_stages list."""
    stages = row.get("required_stages")
    if not isinstance(stages, list) or not stages:
        issues.append(RegistryIssue(f"{prefix}.required_stages", "must be a non-empty list"))
        return
    for index, stage in enumerate(stages):
        if not isinstance(stage, str) or not stage.strip():
            issues.append(RegistryIssue(f"{prefix}.required_stages[{index}]", "must be non-empty"))
            continue
        base_stage = stage.removesuffix("_if_promising")
        if known_stages and stage not in known_stages and base_stage not in known_stages:
            issues.append(
                RegistryIssue(
                    f"{prefix}.required_stages[{index}]",
                    f"unknown stage: {stage}",
                )
            )


def _validate_learned_link(
    issues: list[RegistryIssue],
    row: dict[str, Any],
    *,
    prefix: str,
    repo_root: Path,
) -> None:
    """Validate learned-candidate registry or adapter-contract linkage."""
    family = str(row.get("family") or "")
    if family not in LEARNED_FAMILIES:
        return
    registry_id = row.get("learned_policy_registry_id")
    adapter_contract = row.get("adapter_contract")
    if _is_missing(registry_id) and _is_missing(adapter_contract):
        issues.append(
            RegistryIssue(
                prefix,
                "learned candidates require learned_policy_registry_id or adapter_contract",
            )
        )
        return
    if not _is_missing(adapter_contract):
        _validate_path_field(
            issues,
            row,
            prefix=prefix,
            field="adapter_contract",
            repo_root=repo_root,
        )


def _validate_slurm_row(
    issues: list[RegistryIssue],
    row: dict[str, Any],
    *,
    prefix: str,
    repo_root: Path,
) -> None:
    """Validate a deferred SLURM handoff row."""
    _require_fields(
        issues,
        row,
        prefix=prefix,
        fields={"slurm_handoff", "launch_packet_config_path", "promotion_gate"},
    )
    _validate_path_field(issues, row, prefix=prefix, field="slurm_handoff", repo_root=repo_root)
    _validate_path_field(
        issues,
        row,
        prefix=prefix,
        field="launch_packet_config_path",
        repo_root=repo_root,
    )
    if row.get("training_required") is not True:
        issues.append(RegistryIssue(f"{prefix}.training_required", "must be true for SLURM handoff"))

    packet_path = _resolve_repo_path(repo_root, row.get("launch_packet_config_path"))
    if packet_path is not None and packet_path.exists():
        packet = _load_yaml(packet_path)
        if not isinstance(packet, dict):
            issues.append(RegistryIssue(f"{prefix}.launch_packet_config_path", "must load mapping"))
            return
        for field in ("artifact_policy", "execution_boundary"):
            if not isinstance(packet.get(field), dict):
                issues.append(
                    RegistryIssue(
                        f"{prefix}.launch_packet_config_path.{field}",
                        "is required in launch packet",
                    )
                )


def _validate_candidate(
    issues: list[RegistryIssue],
    candidate_id: str,
    row: Any,
    *,
    repo_root: Path,
    known_gates: set[str],
    known_stages: set[str],
) -> None:
    """Validate one candidate row."""
    prefix = f"candidates.{candidate_id}"
    if not isinstance(row, dict):
        issues.append(RegistryIssue(prefix, "must be a mapping"))
        return

    _require_fields(
        issues,
        row,
        prefix=prefix,
        fields={"status", "family", "training_required", "hypothesis"},
    )
    status = str(row.get("status") or "")
    if status in IMPLEMENTED_STATUSES:
        _require_fields(
            issues,
            row,
            prefix=prefix,
            fields={"candidate_config_path", "promotion_gate"},
        )
        _validate_path_field(
            issues,
            row,
            prefix=prefix,
            field="candidate_config_path",
            repo_root=repo_root,
        )
        _validate_required_stages(issues, row, prefix=prefix, known_stages=known_stages)
    elif status in SLURM_STATUSES:
        _validate_slurm_row(issues, row, prefix=prefix, repo_root=repo_root)
    else:
        issues.append(RegistryIssue(f"{prefix}.status", f"unknown status: {status}"))

    gate = row.get("promotion_gate")
    if isinstance(gate, str) and known_gates and gate not in known_gates:
        issues.append(RegistryIssue(f"{prefix}.promotion_gate", f"unknown gate: {gate}"))

    family = str(row.get("family") or "")
    if "diagnostic" in family and row.get("claim_scope") not in {None, "diagnostic_only"}:
        issues.append(
            RegistryIssue(
                f"{prefix}.claim_scope",
                "diagnostic families may only use claim_scope: diagnostic_only",
            )
        )
    _validate_learned_link(issues, row, prefix=prefix, repo_root=repo_root)


def validate_registry(
    registry_path: Path = DEFAULT_REGISTRY,
    *,
    as_of: date | None = None,
    max_age_days: int = 90,
) -> list[RegistryIssue]:
    """Validate a policy-search candidate registry."""
    registry_path = Path(registry_path)
    repo_root = _repo_root_for(registry_path)
    issues: list[RegistryIssue] = []
    payload = _load_yaml(registry_path)
    payload_mapping = payload if isinstance(payload, dict) else {}
    candidates = _validate_top_level(
        payload,
        issues=issues,
        registry_path=registry_path,
        repo_root=repo_root,
        as_of=as_of or _current_utc_date(),
        default_max_age_days=max_age_days,
    )
    known_gates = payload_mapping.get("_known_promotion_gates", set())
    known_stages = payload_mapping.get("_known_stages", set())
    for candidate_id, row in candidates.items():
        _validate_candidate(
            issues,
            str(candidate_id),
            row,
            repo_root=repo_root,
            known_gates=known_gates,
            known_stages=known_stages,
        )
    return issues


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "registry",
        nargs="?",
        type=Path,
        default=DEFAULT_REGISTRY,
        help="Policy-search candidate registry YAML.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON issues.")
    parser.add_argument(
        "--as-of",
        type=date.fromisoformat,
        default=_current_utc_date(),
        help="Freshness reference date in YYYY-MM-DD form.",
    )
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=90,
        help="Default freshness window when registry freshness.max_age_days is absent.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the registry validator."""
    args = _parse_args()
    issues = validate_registry(
        args.registry,
        as_of=args.as_of,
        max_age_days=args.max_age_days,
    )
    if args.json:
        print(json.dumps([issue.__dict__ for issue in issues], indent=2, sort_keys=True))
    else:
        if not issues:
            print(f"OK: {args.registry} satisfies the policy-search registry contract.")
        for issue in issues:
            print(f"ERROR: {issue.path}: {issue.message}")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
