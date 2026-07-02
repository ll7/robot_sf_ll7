"""Fail-closed preflight for proxy-based checkpoint selection inputs (issue #3204).

The proxy-vs-ADE checkpoint-selection analyzer
(``scripts/research/analyze_predictive_checkpoint_proxy.py``, merged via #3307) can only
produce a conclusive result when its *inputs* are present:

1. a held-out hard-seed fixture to score proxy success on;
2. ">= 6 checkpoints from >= 1 real training run" whose registry ``local_path`` actually
   resolve on this machine; and
3. a training summary whose ``proxy.history`` records non-degenerate spread in hard-seed
   ``success_rate`` across enough proxy epochs.

Issue #3204 is currently *blocked* precisely because (2) and (3) are not satisfied: every
``predictive_*`` registry ``local_path`` points at a non-durable ``output/tmp/...`` path that
does not exist, and the one real probe run records ``success_rate = 0.0`` at every epoch (no
spread -> the analyzer returns ``inconclusive``). This tool turns that manual diagnostic into a
reusable, read-only preflight that fails closed (``status: blocked``) when the inputs are
absent or degenerate.

It selects no checkpoint, runs no training, submits no jobs, downloads nothing, and asserts no
benchmark result. It only maps declared inputs to their on-disk readiness and reports
``ready``/``blocked`` with exit code ``0``/``2``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from math import ceil, isfinite
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]

STATUS_PASSED = "passed"
STATUS_FAILED = "failed"
STATUS_BLOCKED = "blocked"
_KNOWN_BLOCKER_STATUSES = frozenset({STATUS_BLOCKED, "resolved", "diagnostic"})


def _summarize_blocked_artifacts(
    artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build compact counts for blocked-artifact decision telemetry."""
    if not artifacts:
        return {
            "total": 0,
            "by_artifact_type": {},
            "by_status": {},
            "by_storage_scope": {},
        }

    by_artifact_type: dict[str, int] = {}
    by_status: dict[str, int] = {}
    by_storage_scope: dict[str, int] = {}
    for artifact in artifacts:
        artifact_type = _summary_bucket(artifact.get("artifact_type"))
        status = _summary_bucket(artifact.get("status"))
        scope = _summary_bucket(artifact.get("storage_scope"))
        by_artifact_type[artifact_type] = by_artifact_type.get(artifact_type, 0) + 1
        by_status[status] = by_status.get(status, 0) + 1
        by_storage_scope[scope] = by_storage_scope.get(scope, 0) + 1

    return {
        "total": len(artifacts),
        "by_artifact_type": by_artifact_type,
        "by_status": by_status,
        "by_storage_scope": by_storage_scope,
    }


def _summary_bucket(value: Any) -> str:
    """Normalize absent optional summary fields without rewriting valid falsy values."""
    return str(value) if value is not None else "unknown"


DEFAULT_CONFIG = Path("configs/research/predictive_checkpoint_proxy_v1.yaml")
DEFAULT_REGISTRY = Path("model/registry.yaml")


def _load_analyzer() -> Any:
    """Load the merged proxy-vs-ADE analyzer module by path (no package install needed)."""
    tool = _REPO_ROOT / "scripts" / "research" / "analyze_predictive_checkpoint_proxy.py"
    spec = importlib.util.spec_from_file_location("analyze_predictive_checkpoint_proxy", tool)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"Could not load analyzer module from {tool}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _coerce_positive_int(value: Any, *, field: str) -> tuple[int | None, list[str]]:
    """Parse a contract field as a non-negative integer threshold.

    Issue #3204 allows small float-encoded thresholds from YAML (for example
    ``1.0e-9``) in existing manifests. We keep fail-closed behavior while
    accepting these manifest encodings by coercing to the next integer ceiling.
    """
    if value is None:
        return None, []
    if isinstance(value, bool):
        return None, [f"{field} must be a number, not boolean"]
    if isinstance(value, int):
        if value < 0:
            return None, [f"{field} must be >= 0"]
        return value, []
    if isinstance(value, float):
        if not isfinite(value):
            return None, [f"{field} must be a finite number, got {value}"]
        if value < 0:
            return None, [f"{field} must be >= 0"]
        return ceil(value), []
    return None, [f"{field} must be integer-like (int/float), got {type(value).__name__}"]


def _load_yaml(path: Path) -> Any:
    """Load a YAML file, returning None on read/parse failure."""
    try:
        with path.open(encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except (OSError, yaml.YAMLError):
        return None


def _check_config(config: Any, config_path: Path) -> tuple[str, list[str]]:
    """Verify readiness contract exists and declares required keys."""
    if config is None:
        return STATUS_BLOCKED, [f"readiness config not found or unreadable: {config_path}"]
    if not isinstance(config, dict):
        return STATUS_FAILED, ["readiness config must be a mapping"]

    errors: list[str] = []
    hard_seed = config.get("hard_seed_fixture")
    if not isinstance(hard_seed, str) or not hard_seed:
        errors.append("readiness config missing or invalid hard_seed_fixture")

    selector = config.get("checkpoint_selector")
    if not isinstance(selector, dict):
        errors.append("readiness config missing checkpoint_selector mapping")
    else:
        errors.extend(_validate_checkpoint_selector_schema(selector))

    summary_contract = config.get("proxy_summary_contract")
    if summary_contract is not None and not isinstance(summary_contract, dict):
        errors.append("proxy_summary_contract must be mapping when provided")
    errors.extend(_validate_blocked_artifact_schema(config.get("blocked_artifacts")))
    errors.extend(_validate_known_blocker_schema(config.get("known_blockers")))
    if errors:
        return STATUS_FAILED, errors
    return STATUS_PASSED, []


def _validate_checkpoint_selector_schema(selector: dict[str, Any]) -> list[str]:
    """Validate checkpoint-selector metadata without resolving artifacts."""
    errors: list[str] = []
    if not selector.get("registry_tag"):
        errors.append("checkpoint_selector missing registry_tag")
    min_resolvable, min_resolvable_messages = _coerce_positive_int(
        selector.get("min_resolvable_checkpoints"),
        field="min_resolvable_checkpoints",
    )
    if min_resolvable is None and not min_resolvable_messages:
        errors.append("checkpoint_selector missing integer min_resolvable_checkpoints")
    errors.extend(min_resolvable_messages)
    group_field = selector.get("training_run_group_field")
    if group_field is not None and (not isinstance(group_field, str) or not group_field):
        errors.append("checkpoint_selector training_run_group_field must be non-empty string")
    min_group = selector.get("min_resolvable_training_run_checkpoints")
    if min_group is not None:
        coerced, coercion_errors = _coerce_positive_int(
            min_group,
            field="min_resolvable_training_run_checkpoints",
        )
        errors.extend(coercion_errors)
        if coerced is None and not coercion_errors:
            errors.append(
                "checkpoint_selector min_resolvable_training_run_checkpoints must be integer-like"
            )
    return errors


def _validate_known_blocker_schema(known_blockers: Any) -> list[str]:
    """Validate optional known-blocker metadata in the readiness contract."""
    if known_blockers is None:
        return []
    if not isinstance(known_blockers, list):
        return ["known_blockers must be a list when provided"]

    errors: list[str] = []
    for index, blocker in enumerate(known_blockers):
        if not isinstance(blocker, dict):
            errors.append(f"known_blockers[{index}] must be a mapping")
            continue
        blocker_id = blocker.get("id")
        blocker_status = blocker.get("status")
        if not isinstance(blocker_id, str) or not blocker_id:
            errors.append(f"known_blockers[{index}] missing id")
        if blocker_status not in _KNOWN_BLOCKER_STATUSES:
            errors.append(
                f"known_blockers[{index}] status must be one of {sorted(_KNOWN_BLOCKER_STATUSES)}"
            )
    return errors


def _validate_blocked_artifact_schema(blocked_artifacts: Any) -> list[str]:
    """Validate optional blocked-artifact inventory in the readiness contract."""
    if blocked_artifacts is None:
        return []
    if not isinstance(blocked_artifacts, list):
        return ["blocked_artifacts must be a list when provided"]

    errors: list[str] = []
    required_fields = (
        "id",
        "artifact_type",
        "status",
        "storage_scope",
        "path_pattern",
        "revival_condition",
    )
    for index, artifact in enumerate(blocked_artifacts):
        if not isinstance(artifact, dict):
            errors.append(f"blocked_artifacts[{index}] must be a mapping")
            continue
        for field_name in required_fields:
            value = artifact.get(field_name)
            if not isinstance(value, str) or not value:
                errors.append(f"blocked_artifacts[{index}] missing {field_name}")
        artifact_status = artifact.get("status")
        if artifact_status not in _KNOWN_BLOCKER_STATUSES:
            errors.append(
                f"blocked_artifacts[{index}] status must be one of "
                f"{sorted(_KNOWN_BLOCKER_STATUSES)}"
            )
        metadata = artifact.get("required_metadata")
        if (
            not isinstance(metadata, list)
            or not metadata
            or any(not isinstance(item, str) or not item for item in metadata)
        ):
            errors.append(
                f"blocked_artifacts[{index}] required_metadata must be a non-empty list of strings"
            )
    return errors


def _check_blocked_artifacts(config: dict[str, Any]) -> tuple[str, list[str], list[dict[str, Any]]]:
    """Expose blocked input artifacts as a fail-closed report section."""
    artifacts = config.get("blocked_artifacts") or []
    if not isinstance(artifacts, list):
        return STATUS_FAILED, ["blocked_artifacts must be a list"], []

    normalized: list[dict[str, Any]] = []
    messages: list[str] = []
    failed = False
    blocked = False
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            failed = True
            messages.append(f"blocked_artifacts[{index}] must be a mapping")
            continue
        artifact_id = str(artifact.get("id", "unnamed"))
        status = artifact.get("status")
        if status not in _KNOWN_BLOCKER_STATUSES:
            failed = True
            messages.append(f"blocked_artifacts[{index}] has unsupported status: {status}")
            continue
        normalized.append(dict(artifact))
        if status == STATUS_BLOCKED:
            blocked = True
            messages.append(f"blocked artifact remains blocked: {artifact_id}")

    if failed:
        return STATUS_FAILED, messages, normalized
    if blocked:
        return STATUS_BLOCKED, messages, normalized
    return STATUS_PASSED, messages, normalized


def _check_known_blockers(config: dict[str, Any]) -> tuple[str, list[str], list[dict[str, Any]]]:
    """Expose configured known blocker status as a fail-closed report section."""
    blockers = config.get("known_blockers") or []
    if not isinstance(blockers, list):
        return STATUS_FAILED, ["known_blockers must be a list"], []

    normalized: list[dict[str, Any]] = []
    messages: list[str] = []
    failed = False
    blocked = False
    for index, blocker in enumerate(blockers):
        if not isinstance(blocker, dict):
            failed = True
            messages.append(f"known_blockers[{index}] must be a mapping")
            continue
        blocker_id = str(blocker.get("id", "unnamed"))
        status = blocker.get("status")
        if status not in _KNOWN_BLOCKER_STATUSES:
            failed = True
            messages.append(f"known_blockers[{index}] has unsupported status: {status}")
            continue
        normalized.append(dict(blocker))
        if status == STATUS_BLOCKED:
            blocked = True
            messages.append(f"known blocker remains blocked: {blocker_id}")

    if failed:
        return STATUS_FAILED, messages, normalized
    if blocked:
        return STATUS_BLOCKED, messages, normalized
    return STATUS_PASSED, messages, normalized


def _check_hard_seed_fixture(fixture_path: Path) -> tuple[str, list[str]]:
    """Verify the hard-seed fixture exists and parses to a non-empty mapping/list."""
    if not fixture_path.is_file():
        return STATUS_BLOCKED, [f"hard-seed fixture not found or is not a file: {fixture_path}"]
    data = _load_yaml(fixture_path)
    if not data:
        return STATUS_FAILED, [f"hard-seed fixture empty or unreadable: {fixture_path}"]
    return STATUS_PASSED, []


def _resolve_local_path(local_path: Any, repo_root: Path) -> Path | None:
    """Resolve a registry ``local_path`` against ``repo_root`` without downloading."""
    if not isinstance(local_path, str) or not local_path:
        return None
    candidate = Path(local_path)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate


def _artifact_scope(local_path: Any, resolved: Path | None, repo_root: Path) -> str:
    """Classify whether a registry path names durable or worktree-local storage."""
    if not isinstance(local_path, str) or not local_path:
        return "invalid"

    path = Path(local_path)
    if path.parts and path.parts[0] == "output":
        return "worktree_local_output"

    if resolved is not None:
        try:
            relative = resolved.resolve().relative_to(repo_root.resolve())
        except ValueError:
            return "external_or_absolute"
        if relative.parts and relative.parts[0] == "output":
            return "worktree_local_output"
        return "repo_relative"

    return "unknown"


def _public_artifact_metadata(entry: dict[str, Any]) -> dict[str, Any]:
    """Return durable public artifact metadata declared in the model registry."""
    release = entry.get("github_release")
    if not isinstance(release, dict):
        return {"status": "missing", "source": None}

    required = ("repo", "tag", "asset_name", "url", "sha256", "size_bytes")
    missing = [key for key in required if release.get(key) in (None, "")]
    return {
        "status": "declared" if not missing else "incomplete",
        "source": "github_release",
        "repo": release.get("repo"),
        "tag": release.get("tag"),
        "asset_name": release.get("asset_name"),
        "url": release.get("url"),
        "sha256": release.get("sha256"),
        "size_bytes": release.get("size_bytes"),
        "metadata_asset": release.get("metadata_asset"),
        "missing_fields": missing,
    }


def _training_run_group(entry: dict[str, Any], group_field: str | None) -> str | None:
    """Return declared training-run grouping metadata, if the contract asks for it."""
    if not group_field:
        return None
    value = entry.get(group_field)
    if isinstance(value, str) and value:
        return value
    return None


def _checkpoint_mapping_entry(
    model_id: str, entry: dict[str, Any], repo_root: Path, training_run_group_field: str | None
) -> dict[str, Any]:
    """Build one fail-closed checkpoint mapping row without selecting any checkpoint."""
    local_path = entry.get("local_path")
    resolved = _resolve_local_path(local_path, repo_root)
    scope = _artifact_scope(local_path, resolved, repo_root)
    training_run_group = _training_run_group(entry, training_run_group_field)

    if resolved is None:
        status = "invalid_local_path"
        reason = "local_path missing or not a non-empty string"
    elif resolved.is_file():
        status = "present"
        reason = "regular file resolves locally"
    elif resolved.is_dir():
        status = "not_checkpoint_file"
        reason = "local_path resolves to a directory"
    else:
        status = "missing_local_path"
        reason = "local_path does not resolve to a regular file"

    return {
        "model_id": model_id,
        "local_path": local_path,
        "resolved_path": str(resolved) if resolved is not None else None,
        "artifact_scope": scope,
        "public_artifact": _public_artifact_metadata(entry),
        "training_run_group": training_run_group,
        "training_run_group_field": training_run_group_field,
        "status": status,
        "reason": reason,
        "present": status == "present",
    }


def _training_run_group_summary(
    resolvable: list[dict[str, Any]], training_run_group_field: str | None
) -> tuple[dict[str, int], int]:
    """Count locally-resolved checkpoints by declared training-run lineage."""
    grouped_resolvable: dict[str, int] = {}
    missing_group_metadata = 0
    if not training_run_group_field:
        return grouped_resolvable, missing_group_metadata

    for candidate in resolvable:
        group = candidate.get("training_run_group")
        if isinstance(group, str) and group:
            grouped_resolvable[group] = grouped_resolvable.get(group, 0) + 1
        else:
            missing_group_metadata += 1
    return grouped_resolvable, missing_group_metadata


def _candidate_training_run_group_metadata_summary(
    candidates: list[dict[str, Any]], training_run_group_field: str | None
) -> dict[str, Any]:
    """Summarize lineage metadata across all checkpoint candidates."""
    summary: dict[str, Any] = {
        "field": training_run_group_field,
        "candidate_count": len(candidates),
        "with_group": 0,
        "missing_group": 0,
        "missing_group_by_status": {},
    }
    if not training_run_group_field:
        return summary

    missing_by_status: dict[str, int] = {}
    for candidate in candidates:
        group = candidate.get("training_run_group")
        if isinstance(group, str) and group:
            summary["with_group"] += 1
            continue
        summary["missing_group"] += 1
        status = str(candidate.get("status", "unknown"))
        missing_by_status[status] = missing_by_status.get(status, 0) + 1
    summary["missing_group_by_status"] = missing_by_status
    return summary


def _training_run_group_blocker(
    mapping: dict[str, Any], *, registry_tag: str, min_resolvable: int
) -> list[str]:
    """Return fail-closed blocker when no single training run has enough checkpoints."""
    group_field = mapping.get("training_run_group_field")
    if not group_field:
        return []
    min_group = mapping.get("min_resolvable_training_run_checkpoints") or min_resolvable
    grouped = mapping.get("resolvable_by_training_run_group") or {}
    max_group = max(grouped.values(), default=0)
    if max_group >= min_group:
        return []
    return [
        f"only {max_group} locally-resolvable '{registry_tag}' checkpoints share "
        f"training run group field '{group_field}'; need >= {min_group}. "
        f"Resolvable by group: {grouped}; "
        f"missing group metadata: {mapping.get('missing_training_run_group_metadata', 0)}"
    ]


def _check_checkpoint_artifacts(
    registry: dict[str, dict[str, Any]],
    *,
    registry_tag: str,
    min_resolvable: int,
    training_run_group_field: str | None,
    min_resolvable_training_run_checkpoints: int | None,
    repo_root: Path,
) -> tuple[str, list[str], dict[str, Any]]:
    """Map tagged checkpoint candidates and gate on local presence and lineage count."""
    tag = registry_tag.strip().lower()
    candidates: list[dict[str, Any]] = []
    for model_id, entry in registry.items():
        if not isinstance(entry, dict):
            continue
        tags = {str(t).strip().lower() for t in (entry.get("tags") or [])}
        if tag not in tags:
            continue
        candidates.append(
            _checkpoint_mapping_entry(model_id, entry, repo_root, training_run_group_field)
        )

    candidates.sort(key=lambda item: str(item["model_id"]))
    resolvable = [c for c in candidates if c["present"]]
    grouped_resolvable, missing_group_metadata = _training_run_group_summary(
        resolvable, training_run_group_field
    )
    candidate_group_metadata = _candidate_training_run_group_metadata_summary(
        candidates, training_run_group_field
    )
    blocked_by_status: dict[str, int] = {}
    blocked_by_artifact_scope: dict[str, int] = {}
    public_artifacts_by_status: dict[str, int] = {}
    for candidate in candidates:
        artifact_status = str(candidate["public_artifact"]["status"])
        public_artifacts_by_status[artifact_status] = (
            public_artifacts_by_status.get(artifact_status, 0) + 1
        )
        if candidate["present"]:
            continue
        blocked_by_status[candidate["status"]] = blocked_by_status.get(candidate["status"], 0) + 1
        artifact_scope = str(candidate["artifact_scope"])
        blocked_by_artifact_scope[artifact_scope] = (
            blocked_by_artifact_scope.get(artifact_scope, 0) + 1
        )

    mapping = {
        "registry_tag": registry_tag,
        "min_resolvable_checkpoints": min_resolvable,
        "candidate_count": len(candidates),
        "resolvable_count": len(resolvable),
        "training_run_group_field": training_run_group_field,
        "min_resolvable_training_run_checkpoints": min_resolvable_training_run_checkpoints,
        "candidate_training_run_group_metadata": candidate_group_metadata,
        "resolvable_by_training_run_group": grouped_resolvable,
        "missing_training_run_group_metadata": missing_group_metadata,
        "blocked_by_status": blocked_by_status,
        "blocked_by_artifact_scope": blocked_by_artifact_scope,
        "public_artifacts_by_status": public_artifacts_by_status,
        "candidates": candidates,
    }

    if not candidates:
        return STATUS_BLOCKED, [f"no registry entries carry tag '{registry_tag}'"], mapping

    if len(resolvable) < min_resolvable:
        absent = [f"{c['model_id']} ({c['status']})" for c in candidates if not c["present"]]
        return (
            STATUS_BLOCKED,
            [
                f"only {len(resolvable)} of {len(candidates)} '{registry_tag}' checkpoints "
                f"resolve locally; need >= {min_resolvable}. Blocked local_path entries: "
                f"{', '.join(absent)}"
            ],
            mapping,
        )

    incomplete_public_artifacts = [
        (
            str(candidate["model_id"]),
            candidate["public_artifact"].get("missing_fields") or [],
        )
        for candidate in candidates
        if candidate["public_artifact"]["status"] == "incomplete"
    ]
    if incomplete_public_artifacts:
        missing = [
            f"{model_id} missing {', '.join(fields)}"
            for model_id, fields in incomplete_public_artifacts
        ]
        return (
            STATUS_BLOCKED,
            [
                "declared public artifact metadata is incomplete; "
                f"blocked release metadata entries: {'; '.join(missing)}"
            ],
            mapping,
        )

    group_messages = _training_run_group_blocker(
        mapping, registry_tag=registry_tag, min_resolvable=min_resolvable
    )
    if group_messages:
        return STATUS_BLOCKED, group_messages, mapping
    return STATUS_PASSED, [], mapping


def _check_proxy_summary(  # noqa: C901
    summary_path: Path,
    *,
    analyzer: Any,
    require_enabled: bool,
    min_proxy_epochs: int,
    min_success_spread: float,
) -> tuple[str, list[str], dict[str, Any]]:
    """Judge whether a training summary would yield a conclusive proxy-vs-ADE comparison.

    Reuses the merged analyzer so the blocked/ready boundary stays identical to the tool that
    consumes the summary. An ``inconclusive`` verdict (e.g. the all-zero probe with no
    hard-success spread) fails closed.

    Returns:
        tuple[str, list[str], dict[str, Any]]: status, messages, and a compact summary payload.
    """
    if not summary_path.is_file():
        return STATUS_BLOCKED, [f"training summary not found or is not a file: {summary_path}"], {}
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return STATUS_FAILED, [f"training summary unreadable: {exc}"], {}
    if not isinstance(summary, dict):
        return STATUS_FAILED, ["training summary must be a JSON object/mapping"], {}

    metadata_errors = _validate_proxy_summary_metadata(summary, require_enabled=require_enabled)
    if metadata_errors:
        return _blocked_proxy_metadata_result(metadata_errors)

    report = analyzer.analyze_summary(summary)
    verdict = report.get("verdict")
    n_epochs = int(report.get("n_proxy_epochs") or 0)
    success_spread = report.get("success_spread")
    payload = {
        "verdict": verdict,
        "proxy_enabled": report.get("proxy_enabled"),
        "n_proxy_epochs": n_epochs,
        "success_spread": success_spread,
        "min_success_spread": min_success_spread,
    }

    errors: list[str] = []
    if require_enabled and not report.get("proxy_enabled"):
        errors.append("training summary has proxy.enabled != true")
    if verdict == "inconclusive":
        errors.append(
            f"analyzer verdict is inconclusive ({report.get('reason')}); "
            "proxy and ADE selection are indistinguishable"
        )
    try:
        spread = float(success_spread) if success_spread is not None else 0.0
    except (TypeError, ValueError):
        errors.append(f"training summary success_spread invalid: {success_spread!r}")
        spread = 0.0
    if spread < min_success_spread:
        errors.append(
            f"proxy hard-success spread {spread:.3g} below minimum {min_success_spread:.3g}"
        )
    if n_epochs < min_proxy_epochs:
        errors.append(
            f"only {n_epochs} usable proxy epochs; claim contract needs >= {min_proxy_epochs}"
        )

    if errors:
        return STATUS_BLOCKED, errors, payload
    return STATUS_PASSED, [], payload


def _blocked_proxy_metadata_result(
    metadata_errors: list[str],
) -> tuple[str, list[str], dict[str, Any]]:
    """Return a compact fail-closed payload for missing proxy metadata."""
    return (
        STATUS_BLOCKED,
        metadata_errors,
        {
            "proxy_enabled": False,
            "n_proxy_epochs": 0,
            "schema_status": "missing_proxy_metadata",
        },
    )


def _validate_proxy_summary_metadata(
    summary: dict[str, Any], *, require_enabled: bool
) -> list[str]:
    """Validate proxy-summary schema before analyzer normalizes missing fields."""
    proxy = summary.get("proxy")
    if not isinstance(proxy, dict):
        return ["training summary missing proxy mapping"]

    errors: list[str] = []
    if require_enabled and proxy.get("enabled") is not True:
        errors.append("training summary proxy.enabled must be true")

    history = proxy.get("history")
    if not isinstance(history, list):
        errors.append("training summary proxy.history must be a list")
    return errors


def _gate_checkpoint_selector(
    registry: Any,
    selector: dict[str, Any],
    repo_root: Path,
) -> tuple[str, list[str], dict[str, Any]]:
    """Coerce threshold fields and run _check_checkpoint_artifacts, fail-closed on invalid inputs."""
    min_resolvable, min_errors = _coerce_positive_int(
        selector.get("min_resolvable_checkpoints"),
        field="min_resolvable_checkpoints",
    )
    if min_errors:
        return STATUS_FAILED, [*min_errors], {}
    min_group_per_run, group_errors = _coerce_positive_int(
        selector.get("min_resolvable_training_run_checkpoints"),
        field="min_resolvable_training_run_checkpoints",
    )
    if group_errors:
        return STATUS_FAILED, [*group_errors], {}
    return _check_checkpoint_artifacts(
        registry,
        registry_tag=str(selector.get("registry_tag", "")),
        min_resolvable=min_resolvable or 0,
        training_run_group_field=selector.get("training_run_group_field"),
        min_resolvable_training_run_checkpoints=min_group_per_run,
        repo_root=repo_root,
    )


def check_readiness(
    *,
    config_path: Path,
    registry_path: Path,
    repo_root: Path,
    training_summary: Path | None = None,
) -> dict[str, Any]:
    """Run all proxy-checkpoint-selection readiness checks and return a structured report."""
    config = _load_yaml(config_path)
    # Pass the raw YAML result through so a non-mapping top-level document (e.g. a list) is
    # reported as a "must be a mapping" failure rather than coerced to a "not found" blocked.
    config_status, config_messages = _check_config(config, config_path)

    prerequisites: dict[str, dict[str, Any]] = {
        "readiness_config": {"status": config_status, "messages": config_messages}
    }

    # Only run input checks the config is well-formed enough to parameterize.
    cfg = config if isinstance(config, dict) and config_status != STATUS_BLOCKED else {}

    fixture_rel = cfg.get("hard_seed_fixture") if cfg else None
    if isinstance(fixture_rel, str) and fixture_rel:
        fixture_path = Path(fixture_rel)
        if not fixture_path.is_absolute():
            fixture_path = repo_root / fixture_path
        fixture_status, fixture_messages = _check_hard_seed_fixture(fixture_path)
    else:
        fixture_status, fixture_messages = STATUS_BLOCKED, ["hard_seed_fixture not declared"]
    prerequisites["hard_seed_fixture"] = {
        "status": fixture_status,
        "messages": fixture_messages,
    }

    selector = cfg.get("checkpoint_selector") if cfg else None
    if isinstance(selector, dict) and registry_path.exists():
        registry_module = _load_registry_module()
        registry = registry_module.load_registry(registry_path)
        ckpt_status, ckpt_messages, ckpt_mapping = _gate_checkpoint_selector(
            registry, selector, repo_root
        )
    elif not registry_path.exists():
        ckpt_status, ckpt_messages, ckpt_mapping = (
            STATUS_BLOCKED,
            [f"model registry not found: {registry_path}"],
            {},
        )
    else:
        ckpt_status, ckpt_messages, ckpt_mapping = (
            STATUS_BLOCKED,
            ["checkpoint_selector not usable from config"],
            {},
        )
    prerequisites["checkpoint_artifacts"] = {
        "status": ckpt_status,
        "messages": ckpt_messages,
        "mapping": ckpt_mapping,
    }

    # The training-summary check is optional: it only runs when a candidate summary is supplied,
    # because the conclusive proxy run is itself run-gated (resource:slurm).
    if training_summary is not None:
        summary_contract = (cfg.get("proxy_summary_contract") or {}) if cfg else {}
        analyzer = _load_analyzer()
        summary_status, summary_messages, summary_payload = _check_proxy_summary(
            training_summary,
            analyzer=analyzer,
            require_enabled=bool(summary_contract.get("require_enabled", True)),
            min_proxy_epochs=int(summary_contract.get("min_proxy_epochs", 6) or 6),
            min_success_spread=float(summary_contract.get("min_success_spread", 1.0e-9) or 1.0e-9),
        )
        prerequisites["proxy_training_summary"] = {
            "status": summary_status,
            "messages": summary_messages,
            "summary": summary_payload,
        }

    if cfg:
        artifact_status, artifact_messages, artifact_payload = _check_blocked_artifacts(cfg)
        prerequisites["blocked_artifacts"] = {
            "status": artifact_status,
            "messages": artifact_messages,
            "artifacts": artifact_payload,
            "summary": _summarize_blocked_artifacts(artifact_payload),
        }

        blocker_status, blocker_messages, blocker_payload = _check_known_blockers(cfg)
        prerequisites["known_blockers"] = {
            "status": blocker_status,
            "messages": blocker_messages,
            "blockers": blocker_payload,
        }

    errors: list[str] = []
    for payload in prerequisites.values():
        if payload["status"] != STATUS_PASSED:
            errors.extend(payload["messages"])

    status = "ready" if not errors else "blocked"
    return {
        "schema_version": "predictive-checkpoint-proxy-readiness-report.v1",
        "status": status,
        "errors": errors,
        "checked": {
            "readiness_config": str(config_path),
            "registry": str(registry_path),
            "training_summary": str(training_summary) if training_summary else None,
        },
        "prerequisites": prerequisites,
        "claim_boundary": (
            "Diagnostic readiness/preflight only. Reports whether proxy-based checkpoint "
            "selection inputs resolve locally; selects no checkpoint and asserts no benchmark "
            "result. A 'ready' status means inputs are present, not that a proxy beats ADE."
        ),
    }


def _load_registry_module() -> Any:
    """Import the canonical registry loader, adding the repo root to ``sys.path`` if needed."""
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from robot_sf.models import registry as registry_module

    return registry_module


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the proxy-checkpoint readiness contract YAML.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY,
        help="Path to the model registry YAML.",
    )
    parser.add_argument(
        "--training-summary",
        type=Path,
        default=None,
        help="Optional training summary JSON to gate proxy.history spread/epochs.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=_REPO_ROOT,
        help="Repository root for resolving relative paths.",
    )
    parser.add_argument("--json", action="store_true", help="Emit a JSON readiness report.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the readiness preflight and return a shell-friendly exit code (0 ready, 2 blocked)."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()

    def _resolve(path: Path | None) -> Path | None:
        if path is None:
            return None
        return path if path.is_absolute() else repo_root / path

    report = check_readiness(
        config_path=_resolve(args.config) or args.config,
        registry_path=_resolve(args.registry) or args.registry,
        repo_root=repo_root,
        training_summary=_resolve(args.training_summary),
    )

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        label = "READY" if report["status"] == "ready" else "BLOCKED"
        print(f"predictive-checkpoint-proxy readiness: {label}")
        for key, payload in report["prerequisites"].items():
            print(f" - {key}: {payload['status']}")
            for message in payload["messages"]:
                print(f"   * {message}")

    return 0 if report["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
