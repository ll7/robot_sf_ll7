"""Fail-closed checks for local-only model artifact paths in benchmark configs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import yaml

LOCAL_MODEL_KEYS = {"model_path", "resume_from"}
LOCAL_MODEL_PREFIXES = ("output/model_cache/", "output/models/", "output/slurm/")
DEFAULT_BLOCKLIST_PATH = Path("configs/baselines/local_model_artifact_blocklist.yaml")
DEFAULT_PROMOTED_SURFACES_PATH = Path("configs/benchmarks/promoted_config_surfaces.yaml")
PROMOTED_BLOCKED_STATUS = "promoted_blocked"


class BlocklistMetadata(NamedTuple):
    """Known local artifact blockers plus shared issue context."""

    reasons: dict[tuple[str, str, str], str]
    decisions: dict[tuple[str, str, str], str]
    next_actions: dict[tuple[str, str, str], str]
    follow_up_issue: str


@dataclass(frozen=True)
class LocalModelReference:
    """One classified local model path found in a YAML config."""

    path: str
    field: str
    value: str
    status: str
    reason: str
    surface: str = "local_experimental"
    decision: str = ""
    next_action: str = ""
    availability: str = ""


# Blocklist coverage-audit statuses (see ``audit_blocklist_coverage``).
BLOCKLIST_ACTIVE = "active"
BLOCKLIST_ORPHANED_CONFIG_MISSING = "orphaned_config_missing"
BLOCKLIST_ORPHANED_REFERENCE_GONE = "orphaned_reference_gone"


@dataclass(frozen=True)
class BlocklistAuditEntry:
    """Coverage status for one ``blocked_references`` entry.

    The local-artifact blocklist names exact ``(path, field, value)`` triples so a stale
    allowlist row cannot silently hide a new local path. When a baseline config is retired,
    removed, or migrated to a durable ``model_id``, the matching blocklist entry stops
    covering anything real and becomes an orphan that should be pruned. This audit makes
    those retired/migrated rows explicit so the preflight allowlist can shrink as configs are
    resolved.
    """

    path: str
    field: str
    value: str
    reason: str
    status: str
    detail: str


def is_local_output_model_path(value: Any) -> bool:
    """Return whether a value points at a local output model artifact.

    Args:
        value: YAML scalar to inspect.

    Returns:
        True when ``value`` is an ``output/`` model artifact path.
    """
    if not isinstance(value, str):
        return False
    return value.strip().startswith(LOCAL_MODEL_PREFIXES)


def iter_local_model_references(
    payload: Any,
    *,
    path_parts: tuple[str, ...] = (),
) -> list[tuple[str, str]]:
    """Return dotted-field/value pairs for local model artifact references.

    Args:
        payload: Parsed YAML payload.
        path_parts: Dotted-path accumulator used during recursion.

    Returns:
        Local model reference field/value pairs.
    """
    references: list[tuple[str, str]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_str = str(key)
            next_parts = (*path_parts, key_str)
            if key_str in LOCAL_MODEL_KEYS and is_local_output_model_path(value):
                references.append((".".join(next_parts), str(value).strip()))
            references.extend(iter_local_model_references(value, path_parts=next_parts))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            references.extend(
                iter_local_model_references(value, path_parts=(*path_parts, str(index)))
            )
    return references


def _load_yaml_mapping(path: Path, *, strict: bool) -> dict[str, Any]:
    """Load a YAML file as a mapping, optionally raising on malformed payloads.

    Returns:
        Parsed YAML mapping, or an empty mapping for lenient malformed input.
    """
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if isinstance(payload, dict):
        return payload
    if strict:
        raise ValueError(f"{path}: expected top-level mapping")
    return {}


def _parse_blocklist_entry(
    entry: Any,
    *,
    path: Path,
    index: int,
    strict: bool,
) -> tuple[str, str, str, str, str, str] | None:
    """Parse one blocklist entry into normalized fields.

    Returns:
        Tuple of ``(config_path, field, value, reason)`` when the entry is valid.
    """
    if not isinstance(entry, dict):
        if strict:
            raise ValueError(f"{path}: blocked_references[{index}] must be a mapping")
        return None
    config_path = str(entry.get("path") or "").strip()
    field = str(entry.get("field") or "").strip()
    value = str(entry.get("value") or "").strip()
    reason = str(entry.get("reason") or "").strip()
    decision = str(entry.get("decision") or "").strip()
    next_action = str(entry.get("next_action") or "").strip()
    if config_path and field and value and reason:
        return config_path, field, value, reason, decision, next_action
    if strict:
        raise ValueError(
            f"{path}: blocked_references[{index}] requires path, field, value, and reason"
        )
    return None


def load_blocklist(path: Path, *, strict: bool = False) -> BlocklistMetadata:
    """Load known blocked local artifacts keyed by config path, field, and value.

    Args:
        path: YAML blocklist path.
        strict: Whether malformed blocklists should raise. Runtime benchmark loading keeps this
            false to preserve the historical fail-closed local-artifact check even when the
            optional blocklist is absent; scanner/CI entry points use strict mode.

    Returns:
        Blocklist metadata with ``(path, field, value)`` blocker reasons.
    """
    if not path.exists():
        return BlocklistMetadata({}, {}, {}, "")
    payload = _load_yaml_mapping(path, strict=strict)
    if strict and payload.get("version") != 1:
        raise ValueError(f"{path}: version must be 1")
    follow_up_issue = str(payload.get("follow_up_issue") or "").strip()
    entries = payload.get("blocked_references") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        if strict:
            raise ValueError(f"{path}: blocked_references must be a list")
        return BlocklistMetadata({}, {}, {}, follow_up_issue)
    blocklist: dict[tuple[str, str, str], str] = {}
    decisions: dict[tuple[str, str, str], str] = {}
    next_actions: dict[tuple[str, str, str], str] = {}
    for index, entry in enumerate(entries):
        parsed = _parse_blocklist_entry(entry, path=path, index=index, strict=strict)
        if parsed is None:
            continue
        config_path, field, value, reason, decision, next_action = parsed
        key = (config_path, field, value)
        blocklist[key] = reason
        if decision:
            decisions[key] = decision
        if next_action:
            next_actions[key] = next_action
    return BlocklistMetadata(blocklist, decisions, next_actions, follow_up_issue)


def audit_blocklist_coverage(
    blocklist_path: Path,
    *,
    repo_root: Path,
) -> list[BlocklistAuditEntry]:
    """Classify each blocklist entry as active or orphaned against the current configs.

    Args:
        blocklist_path: YAML blocklist with exact ``(path, field, value)`` triples.
        repo_root: Repository root used to resolve repo-relative config paths.

    Returns:
        One :class:`BlocklistAuditEntry` per ``blocked_references`` triple. ``active`` entries
        still cover a present local reference; ``orphaned_config_missing`` entries name a config
        that no longer exists (retired/removed); ``orphaned_reference_gone`` entries name a config
        that exists but no longer carries the blocked local reference (e.g. migrated to a durable
        ``model_id`` or rewritten). Orphaned rows are safe to prune from the allowlist.

    Raises:
        FileNotFoundError: If ``blocklist_path`` does not exist. The audit fails closed rather than
            reporting an empty (vacuously green) result for a missing or mistyped blocklist path.
    """
    if not blocklist_path.is_file():
        raise FileNotFoundError(f"Blocklist file not found: {blocklist_path}")
    blocklist = load_blocklist(blocklist_path, strict=True)
    entries: list[BlocklistAuditEntry] = []
    for (config_path, field, value), reason in blocklist.reasons.items():
        resolved = Path(config_path)
        if not resolved.is_absolute():
            resolved = repo_root / resolved
        if not resolved.is_file():
            entries.append(
                BlocklistAuditEntry(
                    config_path,
                    field,
                    value,
                    reason,
                    BLOCKLIST_ORPHANED_CONFIG_MISSING,
                    "Config file no longer exists; remove this blocklist entry.",
                )
            )
            continue
        payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        present = set(iter_local_model_references(payload))
        if (field, value) in present:
            entries.append(
                BlocklistAuditEntry(
                    config_path,
                    field,
                    value,
                    reason,
                    BLOCKLIST_ACTIVE,
                    "Config still carries this local artifact reference.",
                )
            )
        else:
            entries.append(
                BlocklistAuditEntry(
                    config_path,
                    field,
                    value,
                    reason,
                    BLOCKLIST_ORPHANED_REFERENCE_GONE,
                    (
                        "Config no longer references this local artifact "
                        "(migrated or rewritten); remove this blocklist entry."
                    ),
                )
            )
    return entries


def _parse_promoted_surface_entry(
    entry: Any,
    *,
    path: Path,
    index: int,
    strict: bool,
) -> tuple[str, str] | None:
    """Parse one promoted-surface manifest entry.

    Returns:
        Tuple of ``(config_path, reason)`` when the entry is valid.
    """
    if isinstance(entry, str):
        config_path = entry.strip()
        reason = "Benchmark-promoted config surface."
    elif isinstance(entry, dict):
        config_path = str(entry.get("path") or "").strip()
        reason = str(entry.get("reason") or "").strip()
    else:
        if strict:
            raise ValueError(f"{path}: promoted_configs[{index}] must be a mapping or string")
        return None
    if config_path and reason:
        return config_path, reason
    if strict:
        raise ValueError(f"{path}: promoted_configs[{index}] requires path and reason")
    return None


def load_promoted_surfaces(path: Path, *, strict: bool = True) -> dict[str, str]:
    """Load benchmark-promoted config paths and their reasons.

    Args:
        path: YAML promoted-surface manifest path.
        strict: Whether malformed manifests should raise.

    Returns:
        Mapping from config path spelling to promotion reason.
    """
    if not path.exists():
        if strict:
            raise FileNotFoundError(path)
        return {}
    payload = _load_yaml_mapping(path, strict=strict)
    if strict and payload.get("version") != 1:
        raise ValueError(f"{path}: version must be 1")
    entries = payload.get("promoted_configs")
    if not isinstance(entries, list):
        if strict:
            raise ValueError(f"{path}: promoted_configs must be a list")
        return {}

    surfaces: dict[str, str] = {}
    for index, entry in enumerate(entries):
        parsed = _parse_promoted_surface_entry(entry, path=path, index=index, strict=strict)
        if parsed is None:
            continue
        config_path, reason = parsed
        surfaces[config_path] = reason
    return surfaces


def path_lookup_candidates(
    path: Path,
    *,
    repo_root: Path | None = None,
    cwd: Path | None = None,
    include_name: bool = False,
) -> list[str]:
    """Return stable path spellings for config-surface and blocklist matching."""
    candidates = [path.as_posix()]
    resolved = path.resolve()
    if repo_root is not None:
        try:
            candidates.append(resolved.relative_to(repo_root.resolve()).as_posix())
        except ValueError:
            pass
    if path.is_absolute():
        base = cwd.resolve() if cwd is not None else Path.cwd().resolve()
        try:
            candidates.append(resolved.relative_to(base).as_posix())
        except ValueError:
            pass
    if include_name:
        candidates.append(path.name)
    return list(dict.fromkeys(candidates))


def display_path(path: Path, *, repo_root: Path | None = None) -> str:
    """Return a stable display path, preferring a repository-relative spelling."""
    if repo_root is not None:
        try:
            return path.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            pass
    return path.as_posix()


def iter_yaml_files(paths: list[Path]) -> list[Path]:
    """Expand scan roots into sorted YAML files.

    Args:
        paths: YAML files or directories to scan.

    Returns:
        YAML files to inspect.
    """
    files: set[Path] = set()
    for path in paths:
        if path.is_dir():
            files.update(path.rglob("*.yaml"))
            files.update(path.rglob("*.yml"))
        elif path.suffix.lower() in {".yaml", ".yml"}:
            files.add(path)
    return sorted(files)


def classify_local_model_references(
    scan_paths: list[Path],
    *,
    repo_root: Path,
    blocklist_path: Path = DEFAULT_BLOCKLIST_PATH,
    promoted_surfaces_path: Path = DEFAULT_PROMOTED_SURFACES_PATH,
) -> list[LocalModelReference]:
    """Inspect YAML configs and classify local model references.

    Args:
        scan_paths: YAML files or directories to scan.
        repo_root: Repository root used for path display and repo-relative manifests.
        blocklist_path: Explicit blocker manifest for unresolved local artifacts.
        promoted_surfaces_path: Manifest of configs that must never use local artifact paths.

    Returns:
        Classified local artifact references.
    """
    blocklist = load_blocklist(blocklist_path, strict=True)
    promoted_surfaces = load_promoted_surfaces(promoted_surfaces_path, strict=True)
    follow_up_issue = blocklist.follow_up_issue or "https://github.com/ll7/robot_sf_ll7/issues/1764"
    expanded_scan_paths = list(scan_paths)
    expanded_scan_paths.extend(
        path if path.is_absolute() else repo_root / path
        for path in (Path(path) for path in promoted_surfaces)
    )

    rows: list[LocalModelReference] = []
    for yaml_path in iter_yaml_files(expanded_scan_paths):
        if yaml_path.resolve() == blocklist_path.resolve():
            continue
        rel_path = display_path(yaml_path, repo_root=repo_root)
        lookup_paths = path_lookup_candidates(
            yaml_path,
            repo_root=repo_root,
            cwd=Path.cwd(),
            include_name=True,
        )
        promoted_reason = next(
            (
                promoted_surfaces[candidate]
                for candidate in lookup_paths
                if candidate in promoted_surfaces
            ),
            "",
        )
        payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        for field, value in iter_local_model_references(payload):
            if promoted_reason:
                rows.append(
                    LocalModelReference(
                        rel_path,
                        field,
                        value,
                        PROMOTED_BLOCKED_STATUS,
                        (
                            f"{promoted_reason} Replace the local output/ reference with a "
                            "durable model_id or artifact pointer before using it as a promoted "
                            f"benchmark config. Follow-up: {follow_up_issue}"
                        ),
                        "benchmark_promoted",
                    )
                )
                continue
            key = (rel_path, field, value)
            reason = blocklist.reasons.get(key)
            if reason:
                decision = blocklist.decisions.get(key, "")
                next_action = blocklist.next_actions.get(key, "")
                rows.append(
                    LocalModelReference(
                        rel_path,
                        field,
                        value,
                        "blocked",
                        reason,
                        decision=decision,
                        next_action=next_action,
                        availability="unavailable" if decision or next_action else "",
                    )
                )
            else:
                rows.append(
                    LocalModelReference(
                        rel_path,
                        field,
                        value,
                        "unblocked",
                        "replace with model_id or add an explicit artifact-promotion blocker",
                        "local_experimental",
                        availability="unknown",
                    )
                )
    return rows


def validate_no_local_model_path_value(
    value: Any,
    *,
    field: str = "model_path",
    owner: str = "planner config",
) -> None:
    """Raise when a direct planner model path points at local ``output/`` artifacts."""
    if not is_local_output_model_path(value):
        return
    raise ValueError(
        f"{owner}:{field} points at local-only model artifact {str(value).strip()!r}. "
        "Local output/ artifacts are not durable across checkouts; use model_id or a "
        "tracked/durable artifact pointer instead."
    )


def validate_no_local_model_artifacts(
    payload: dict[str, Any],
    *,
    config_path: Path,
    blocklist_path: Path = DEFAULT_BLOCKLIST_PATH,
) -> None:
    """Raise when a benchmark config depends on a local ``output/`` model artifact.

    Args:
        payload: Parsed algorithm config mapping.
        config_path: Path to the config file being loaded.
        blocklist_path: Optional known-blocker metadata file.

    Raises:
        ValueError: If ``payload`` contains a local-only model artifact path.
    """
    references = iter_local_model_references(payload)
    if not references:
        return

    blocklist = load_blocklist(blocklist_path)
    lookup_paths = path_lookup_candidates(config_path, cwd=Path.cwd())
    display_path = lookup_paths[-1]
    messages: list[str] = []
    for field, value in references:
        key = next(
            (
                (candidate, field, value)
                for candidate in lookup_paths
                if (candidate, field, value) in blocklist.reasons
            ),
            None,
        )
        reason = blocklist.reasons[key] if key is not None else ""
        if reason:
            next_action = blocklist.next_actions.get(key, "") if key is not None else ""
            decision = blocklist.decisions.get(key, "") if key is not None else ""
            status_bits = []
            if decision:
                status_bits.append(f"decision={decision}")
            if next_action:
                status_bits.append(f"next_action={next_action}")
            status_suffix = f" ({'; '.join(status_bits)})" if status_bits else ""
            messages.append(
                f"{display_path}:{field} points at unavailable local artifact {value!r}: "
                f"{reason}{status_suffix}"
            )
        else:
            messages.append(
                f"{display_path}:{field} points at local artifact {value!r}; "
                "replace it with model_id or record an artifact-promotion blocker"
            )
    follow_up = f" Follow-up: {blocklist.follow_up_issue}." if blocklist.follow_up_issue else ""
    raise ValueError(
        "Benchmark algorithm config contains local-only model artifact path(s). "
        "Local output/ artifacts are not durable across checkouts. "
        + " | ".join(messages)
        + follow_up
    )
