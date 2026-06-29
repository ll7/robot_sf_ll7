#!/usr/bin/env python3
"""Plan local model-artifact promotion or retirement decisions.

This tool is metadata-only. It computes small, reviewable plans for local
``output/...`` checkpoint references and durable ``model_id`` references; it
does not upload, copy, or commit checkpoint artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.local_model_artifacts import (
    BlocklistMetadata,
    display_path,
    iter_local_model_references,
    iter_yaml_files,
    load_blocklist,
    load_promoted_surfaces,
    path_lookup_candidates,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY_PATH = Path("model/registry.yaml")
DEFAULT_PROMOTED_SURFACES_PATH = Path("configs/benchmarks/promoted_config_surfaces.yaml")
DEFAULT_BLOCKLIST_PATH = Path("configs/baselines/local_model_artifact_blocklist.yaml")
SCHEMA_VERSION = "robot-sf-model-artifact-promotion-plan.v1"

INITIAL_TARGET_CONFIGS = [
    Path("configs/baselines/ppo_issue_791_horizon100_12178.yaml"),
    Path("configs/baselines/ppo_issue_856_all_scenarios_12223.yaml"),
    Path("configs/baselines/sac_gate_socnav_struct.yaml"),
    Path("configs/baselines/sac_gate_socnav_struct_ego.yaml"),
    Path("configs/baselines/sac_gate_socnav_struct_ego_multi.yaml"),
    Path("configs/baselines/sac_gate_socnav_struct_ego_safe.yaml"),
    Path("configs/baselines/drl_vo_default.yaml"),
]


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    """Load one YAML mapping from disk."""
    payload = _load_yaml_payload(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected top-level mapping")
    return payload


def _load_yaml_payload(path: Path) -> Any:
    """Load one YAML payload from disk."""
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_registry(path: Path) -> dict[str, dict[str, Any]]:
    """Load model registry entries keyed by model id."""
    if not path.exists():
        return {}
    payload = _load_yaml_mapping(path)
    models = payload.get("models")
    if not isinstance(models, list):
        raise ValueError(f"{path}: models must be a list")
    registry: dict[str, dict[str, Any]] = {}
    for entry in models:
        if not isinstance(entry, dict):
            continue
        model_id = str(entry.get("model_id") or "").strip()
        if model_id:
            registry[model_id] = entry
    return registry


def _is_durable_registry_entry(entry: dict[str, Any]) -> bool:
    """Return whether a registry entry has a durable artifact pointer."""
    github_release = entry.get("github_release")
    if _github_release_is_public_pointer(github_release):
        return True
    return bool(
        entry.get("wandb_artifact_path")
        or entry.get("wandb_run_path")
        or (entry.get("wandb_entity") and entry.get("wandb_project") and entry.get("wandb_run_id"))
    )


def _github_release_is_public_pointer(release: Any) -> bool:
    """Return whether GitHub release metadata is enough to retrieve and verify an asset."""
    if not isinstance(release, dict):
        return False
    has_location = bool(
        release.get("url")
        or (release.get("repo") and release.get("tag") and release.get("asset_name"))
    )
    return bool(has_location and release.get("sha256"))


def _sha256(path: Path) -> str:
    """Compute a SHA256 checksum for a local artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_model_id(config_path: str) -> str:
    """Build a conservative candidate registry id from a config path."""
    stem = Path(config_path).stem
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", stem).strip("_").lower()
    if not cleaned:
        cleaned = "local_model"
    return f"{cleaned}_local_artifact_candidate"


def _resolve_repo_path(path_value: str, *, repo_root: Path) -> Path:
    """Resolve an artifact path relative to the repository root."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return repo_root / path


def _artifact_metadata(path_value: str, *, repo_root: Path) -> dict[str, Any]:
    """Return metadata for a local artifact path without copying or uploading it."""
    resolved = _resolve_repo_path(path_value, repo_root=repo_root)
    metadata: dict[str, Any] = {
        "source_path": path_value,
        "exists": resolved.exists(),
        "size_bytes": None,
        "sha256": None,
    }
    if resolved.is_file():
        metadata["size_bytes"] = resolved.stat().st_size
        metadata["sha256"] = _sha256(resolved)
    return metadata


def _expand_config_paths(paths: list[Path], *, repo_root: Path) -> list[Path]:
    """Expand file and directory arguments into ordered YAML config files."""
    expanded: list[Path] = []
    for path in paths:
        resolved = path if path.is_absolute() else repo_root / path
        if resolved.is_dir():
            expanded.extend(iter_yaml_files([resolved]))
        elif resolved.suffix.lower() in {".yaml", ".yml"}:
            expanded.append(resolved)
        else:
            raise FileNotFoundError(f"Expected YAML file or directory: {path}")
    return expanded


def _promoted_reason_for(
    config_path: Path,
    *,
    repo_root: Path,
    promoted_surfaces: dict[str, str],
) -> str:
    """Return the promoted-surface reason for a config, if any."""
    candidates = path_lookup_candidates(config_path, repo_root=repo_root, cwd=Path.cwd())
    return next(
        (
            promoted_surfaces[candidate]
            for candidate in candidates
            if candidate in promoted_surfaces
        ),
        "",
    )


def _promotion_plan(
    *,
    config_path: str,
    field: str,
    value: str,
    artifact: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Build the metadata-only promotion plan for a local artifact reference."""
    return {
        "target_registry_id": _safe_model_id(config_path),
        "source_config": config_path,
        "source_field": field,
        "source_path": value,
        "sha256": artifact["sha256"],
        "size_bytes": artifact["size_bytes"],
        "license_access_note": (
            "Maintainer must confirm checkpoint license, access rights, source run, and durable "
            "publication target before promotion."
        ),
        "claim_boundary": "not benchmark evidence until durable artifact exists",
        "source_commit": payload.get("commit"),
        "training_config": (payload.get("provenance") or {}).get("training_config")
        if isinstance(payload.get("provenance"), dict)
        else None,
    }


def _row_for_local_reference(
    *,
    config_path: str,
    field: str,
    value: str,
    payload: dict[str, Any],
    repo_root: Path,
    promoted_reason: str,
    blocker: dict[str, str],
) -> dict[str, Any]:
    """Classify one local output artifact reference."""
    artifact = _artifact_metadata(value, repo_root=repo_root)
    promotion_plan = _promotion_plan(
        config_path=config_path,
        field=field,
        value=value,
        artifact=artifact,
        payload=payload,
    )
    blocker_reason = blocker.get("reason", "")
    blocker_decision = blocker.get("decision", "")
    blocker_next_action = blocker.get("next_action", "")
    if promoted_reason:
        return {
            "config_path": config_path,
            "classification": "manual_decision_required",
            "decision": "replace_local_artifact_before_benchmark_use",
            "surface": "benchmark_promoted",
            "field": field,
            "value": value,
            "artifact": artifact,
            "promotion_plan": promotion_plan,
            "blocker_reason": blocker_reason or None,
            "availability": "unavailable",
            "unavailable_reason": promoted_reason,
            "claim_boundary": "not benchmark evidence until durable artifact exists",
            "action": (
                f"{promoted_reason} This local-only path must not be treated as benchmark "
                "evidence; replace it with a durable model_id/artifact pointer or remove the "
                "config from promoted surfaces."
            ),
        }

    if blocker_decision:
        classification = "unavailable"
        decision = blocker_decision
        action = blocker_next_action or (
            "Recover the checkpoint and prove durable provenance before use, or retire/rewrite "
            "this config."
        )
    elif artifact["exists"]:
        classification = "promotable"
        decision = "prepare_promotion_metadata"
        action = (
            "Review provenance, license/access, source commit, and target registry id before any "
            "explicit promotion/upload step."
        )
    elif str(payload.get("profile") or "").strip().lower() == "experimental":
        classification = "retire_candidate"
        decision = "recover_or_retire"
        action = (
            "Missing experimental checkpoint: recover the checkpoint and prove provenance if it "
            "is worth promoting; otherwise retire or rewrite this config."
        )
    else:
        classification = "missing"
        decision = "recover_before_promotion"
        action = (
            "Missing checkpoint: recover the checkpoint from a durable source or retire this "
            "config; do not use it as benchmark evidence."
        )
    if blocker_reason:
        action = f"{blocker_reason} {action}"

    return {
        "config_path": config_path,
        "classification": classification,
        "decision": decision,
        "surface": "local_experimental",
        "field": field,
        "value": value,
        "artifact": artifact,
        "promotion_plan": promotion_plan,
        "blocker_reason": blocker_reason or None,
        "availability": "unavailable" if blocker_decision else "local_only",
        "unavailable_reason": blocker_reason or None,
        "claim_boundary": "not benchmark evidence until durable artifact exists",
        "action": action,
    }


def _row_for_model_id(
    *,
    config_path: str,
    model_id: str,
    registry: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Classify a config that references a model registry id."""
    entry = registry.get(model_id)
    if entry is not None and bool(entry.get("local_only")):
        return {
            "config_path": config_path,
            "classification": "retired_local_only",
            "decision": "retired_until_durable_artifact_recovered",
            "surface": "local_only_registry",
            "model_id": model_id,
            "registry_entry": {
                "model_id": model_id,
                "local_path": entry.get("local_path"),
                "replacement_model_id": entry.get("replacement_model_id"),
            },
            "claim_boundary": "not benchmark evidence; retired local-only artifact",
            "action": (
                "Config references an explicit local-only retired registry entry; recover a "
                "durable artifact with checksum before benchmark use, or keep retired."
            ),
        }
    if entry is not None and _is_durable_registry_entry(entry):
        return {
            "config_path": config_path,
            "classification": "already_registered",
            "decision": "no_local_promotion_needed",
            "surface": "durable_registry",
            "model_id": model_id,
            "registry_entry": {
                "model_id": model_id,
                "public_artifact_source": entry.get("public_artifact_source"),
                "github_release": entry.get("github_release"),
                "wandb_artifact_path": entry.get("wandb_artifact_path"),
                "wandb_run_path": entry.get("wandb_run_path"),
            },
            "claim_boundary": (
                "durable artifact pointer exists; benchmark claims still require benchmark proof"
            ),
            "action": "No local artifact promotion needed for this config.",
        }
    return {
        "config_path": config_path,
        "classification": "manual_decision_required",
        "decision": "resolve_model_id_alias",
        "surface": "registry_alias",
        "model_id": model_id,
        "claim_boundary": "not benchmark evidence until durable artifact exists",
        "action": (
            "model_id is not backed by a durable registry entry; add registry provenance, "
            "replace the alias, or retire the config."
        ),
    }


def _row_for_config(
    config_path: Path,
    *,
    repo_root: Path,
    registry: dict[str, dict[str, Any]],
    promoted_surfaces: dict[str, str],
    blocklist: BlocklistMetadata,
) -> dict[str, Any]:
    """Build one planner row for a YAML config."""
    rel_path = display_path(config_path, repo_root=repo_root)
    payload = _load_yaml_payload(config_path)
    if not isinstance(payload, dict):
        return {
            "config_path": rel_path,
            "classification": "manual_decision_required",
            "decision": "unsupported_yaml_shape",
            "surface": "non_mapping_yaml",
            "yaml_top_level_type": type(payload).__name__,
            "claim_boundary": "not benchmark evidence until artifact provenance is explicit",
            "action": (
                "Top-level YAML is not a mapping, so model artifact fields were not interpreted; "
                "inspect this file manually or exclude it from the model promotion scan."
            ),
        }
    local_references = iter_local_model_references(payload)
    promoted_reason = _promoted_reason_for(
        config_path,
        repo_root=repo_root,
        promoted_surfaces=promoted_surfaces,
    )
    if local_references:
        field, value = local_references[0]
        blocklist_key = (rel_path, field, value)
        blocker = {
            "reason": blocklist.reasons.get(blocklist_key, ""),
            "decision": blocklist.decisions.get(blocklist_key, ""),
            "next_action": blocklist.next_actions.get(blocklist_key, ""),
        }
        row = _row_for_local_reference(
            config_path=rel_path,
            field=field,
            value=value,
            payload=payload,
            repo_root=repo_root,
            promoted_reason=promoted_reason,
            blocker=blocker,
        )
        row["local_references"] = [
            {"field": reference_field, "value": reference_value}
            for reference_field, reference_value in local_references
        ]
        return row

    model_id = payload.get("model_id")
    if isinstance(model_id, str) and model_id.strip():
        return _row_for_model_id(config_path=rel_path, model_id=model_id.strip(), registry=registry)

    return {
        "config_path": rel_path,
        "classification": "manual_decision_required",
        "decision": "no_model_reference_found",
        "surface": "unknown",
        "claim_boundary": "not benchmark evidence until artifact provenance is explicit",
        "action": "No local model_path or model_id was found; inspect this config manually.",
    }


def build_promotion_report(
    paths: list[Path],
    *,
    repo_root: Path = REPO_ROOT,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    promoted_surfaces_path: Path = DEFAULT_PROMOTED_SURFACES_PATH,
    blocklist_path: Path = DEFAULT_BLOCKLIST_PATH,
) -> dict[str, Any]:
    """Build a metadata-only promotion/retirement report for config paths."""
    registry_resolved = registry_path if registry_path.is_absolute() else repo_root / registry_path
    promoted_resolved = (
        promoted_surfaces_path
        if promoted_surfaces_path.is_absolute()
        else repo_root / promoted_surfaces_path
    )
    blocklist_resolved = (
        blocklist_path if blocklist_path.is_absolute() else repo_root / blocklist_path
    )
    registry = _load_registry(registry_resolved)
    promoted_surfaces = load_promoted_surfaces(promoted_resolved, strict=True)
    blocklist = load_blocklist(blocklist_resolved, strict=True)
    ignored_manifest_paths = {
        registry_resolved.resolve(),
        promoted_resolved.resolve(),
        blocklist_resolved.resolve(),
    }
    config_paths = [
        path
        for path in _expand_config_paths(paths or INITIAL_TARGET_CONFIGS, repo_root=repo_root)
        if path.resolve() not in ignored_manifest_paths
    ]
    rows = [
        _row_for_config(
            path,
            repo_root=repo_root,
            registry=registry,
            promoted_surfaces=promoted_surfaces,
            blocklist=blocklist,
        )
        for path in config_paths
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "claim_boundary": (
            "Planner output is metadata/provenance guidance only; it is not benchmark evidence "
            "and does not publish artifacts."
        ),
        "rows": rows,
    }


def render_issue_table(report: dict[str, Any]) -> str:
    """Render a concise GitHub-issue-ready decision table."""
    lines = [
        "| Config | Classification | Decision | Action |",
        "| --- | --- | --- | --- |",
    ]
    for row in report["rows"]:
        action = str(row["action"]).replace("\n", " ")
        if len(action) > 180:
            action = action[:177].rstrip() + "..."
        lines.append(
            "| {config_path} | {classification} | {decision} | {action} |".format(
                config_path=row["config_path"],
                classification=row["classification"],
                decision=row["decision"],
                action=action,
            )
        )
    return "\n".join(lines)


def _write_report(path: Path, report: dict[str, Any]) -> None:
    """Write report JSON to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add shared path arguments to a subcommand parser."""
    parser.add_argument("paths", nargs="*", type=Path, help="YAML config files or directories.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--registry-path", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--promoted-surfaces", type=Path, default=DEFAULT_PROMOTED_SURFACES_PATH)
    parser.add_argument("--blocklist-path", type=Path, default=DEFAULT_BLOCKLIST_PATH)


def _build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan = subparsers.add_parser("scan", help="Scan configs and print a promotion table.")
    _add_common_args(scan)
    scan.add_argument("--json", action="store_true", help="Print report JSON instead of Markdown.")

    plan = subparsers.add_parser("plan", help="Plan one or more configs.")
    _add_common_args(plan)
    plan.add_argument("--config", action="append", type=Path, default=[], help="Config path.")
    plan.add_argument("--json", action="store_true", help="Print report JSON instead of Markdown.")

    write_report = subparsers.add_parser(
        "write-report",
        help="Write JSON report and print a GitHub issue decision table.",
    )
    _add_common_args(write_report)
    write_report.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the artifact-promotion planner CLI."""
    args = _build_parser().parse_args(argv)
    paths = list(args.paths)
    if args.command == "plan":
        paths = [*args.config, *paths]
    report = build_promotion_report(
        paths,
        repo_root=args.repo_root,
        registry_path=args.registry_path,
        promoted_surfaces_path=args.promoted_surfaces,
        blocklist_path=args.blocklist_path,
    )

    if args.command == "write-report":
        _write_report(args.output, report)
        print(render_issue_table(report))
        return 0
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_issue_table(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
