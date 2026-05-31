"""Fail-closed checks for local-only model artifact paths in benchmark configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import yaml

LOCAL_MODEL_KEYS = {"model_path", "resume_from"}
LOCAL_MODEL_PREFIXES = ("output/model_cache/", "output/models/", "output/slurm/")
DEFAULT_BLOCKLIST_PATH = Path("configs/baselines/local_model_artifact_blocklist.yaml")


class BlocklistMetadata(NamedTuple):
    """Known local artifact blockers plus shared issue context."""

    reasons: dict[tuple[str, str, str], str]
    follow_up_issue: str


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


def _load_blocklist(path: Path) -> BlocklistMetadata:
    """Load known blocked local artifacts keyed by config path, field, and value.

    Returns:
        Blocklist metadata with ``(path, field, value)`` blocker reasons.
    """
    if not path.exists():
        return BlocklistMetadata({}, "")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    follow_up_issue = ""
    if isinstance(payload, dict):
        follow_up_issue = str(payload.get("follow_up_issue") or "").strip()
    entries = payload.get("blocked_references") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return BlocklistMetadata({}, follow_up_issue)
    blocklist: dict[tuple[str, str, str], str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        config_path = str(entry.get("path") or "").strip()
        field = str(entry.get("field") or "").strip()
        value = str(entry.get("value") or "").strip()
        reason = str(entry.get("reason") or "").strip()
        if config_path and field and value and reason:
            blocklist[(config_path, field, value)] = reason
    return BlocklistMetadata(blocklist, follow_up_issue)


def _path_lookup_candidates(path: Path) -> list[str]:
    """Return stable path spellings for blocklist matching."""
    candidates = [path.as_posix()]
    if path.is_absolute():
        try:
            candidates.append(path.relative_to(Path.cwd()).as_posix())
        except ValueError:
            pass
    return list(dict.fromkeys(candidates))


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

    blocklist = _load_blocklist(blocklist_path)
    lookup_paths = _path_lookup_candidates(config_path)
    display_path = lookup_paths[-1]
    messages: list[str] = []
    for field, value in references:
        reason = next(
            (
                blocklist.reasons[(candidate, field, value)]
                for candidate in lookup_paths
                if (candidate, field, value) in blocklist.reasons
            ),
            "",
        )
        if reason:
            messages.append(f"{display_path}:{field} points at local artifact {value!r}: {reason}")
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
