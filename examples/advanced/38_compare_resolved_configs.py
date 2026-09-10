"""Compare two resolved training configurations and explain semantic drift.

Raw YAML diffs hide inherited defaults, deep-merged mappings, list replacement,
and metadata noise. This example resolves both inputs through the canonical
``base_config`` resolver and reports each changed leaf with a stable change
class so reviewers can tell a semantic contract change from a provenance,
execution-environment, or presentation-only change.

Usage:
    uv run python examples/advanced/38_compare_resolved_configs.py \\
        --left examples/fixtures/config_drift/left.yaml \\
        --right examples/fixtures/config_drift/right.yaml --json

Change classes are deliberately conservative and source-derived:

- ``semantic``: keys the benchmark governance policy names as identity-bearing
  (schema/version markers, seed and seed-policy keys, action semantics, and the
  observation/metric/model contract markers). See
  ``docs/benchmark_governance.md``.
- ``provenance``: lineage and bookkeeping fields (commit, run/job identifiers,
  timestamps, authors, notes, attempt counters).
- ``execution_environment``: host, device, worker, scheduler, and output/cache
  location fields.
- ``presentation_only``: display names, descriptions, and tags.
- ``unknown``: any other changed leaf. Unknown differences fail closed: the
  report marks the pair ``not_comparable`` instead of guessing.

The script never mutates the inputs, never writes outside ``output/``, performs
no network access, and makes no scientific comparability claim beyond the
declared policy above.

Limitations:
    The unknown classifier is intentionally incomplete; it reports what the
    current owners do not declare rather than inventing a field policy.

References:
    - docs/benchmark_governance.md
    - examples/advanced/README_compare_resolved_configs.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from scripts.training.train_ppo import _load_expert_training_config_mapping

REPORT_SCHEMA = "resolved_config_drift.v1"

CHANGE_SEMANTIC = "semantic"
CHANGE_PROVENANCE = "provenance"
CHANGE_EXECUTION_ENV = "execution_environment"
CHANGE_PRESENTATION = "presentation_only"
CHANGE_UNKNOWN = "unknown"

VERDICT_IDENTICAL = "identical"
VERDICT_COMPARABLE = "comparable"
VERDICT_NOT_COMPARABLE = "not_comparable"

#: Keys the benchmark-governance policy treats as identity-bearing contract
#: markers. Changing one of these makes two resolved configs non-comparable.
SEMANTIC_KEYS = frozenset(
    {
        "action_semantics",
        "benchmark_protocol_version",
        "metric_schema_version",
        "model_profile_version",
        "observation_contract",
        "release_id",
        "scenario_schema_version",
        "seed",
        "seed_policy",
        "track_schema_version",
    }
)

#: Lineage/bookkeeping keys that never change simulation semantics.
PROVENANCE_KEYS = frozenset(
    {
        "attempt",
        "author",
        "commit",
        "created_at",
        "job_id",
        "notes",
        "run_id",
        "source",
        "submission_id",
        "updated_at",
        "wandb_entity",
        "wandb_project",
        "wandb_run_id",
    }
)

#: Host/device/scheduler and storage-location keys.
EXECUTION_ENVIRONMENT_KEYS = frozenset(
    {
        "cache_dir",
        "device",
        "gpu",
        "host",
        "log_dir",
        "num_workers",
        "output_dir",
        "partition",
        "slurm_account",
        "slurm_partition",
        "worker",
    }
)

#: Human-facing labels that do not affect resolved behavior.
PRESENTATION_KEYS = frozenset({"description", "display_name", "label", "name", "tags", "title"})

_CLASS_BY_KEY: dict[str, str] = {}
for _keys, _class in (
    (SEMANTIC_KEYS, CHANGE_SEMANTIC),
    (PROVENANCE_KEYS, CHANGE_PROVENANCE),
    (EXECUTION_ENVIRONMENT_KEYS, CHANGE_EXECUTION_ENV),
    (PRESENTATION_KEYS, CHANGE_PRESENTATION),
):
    for _key in _keys:
        _CLASS_BY_KEY[_key] = _class

_IDENTITY_CLASSES = frozenset({CHANGE_SEMANTIC, CHANGE_UNKNOWN})


class ConfigDriftError(ValueError):
    """Raised when an input cannot be compared without guessing."""

    def __init__(self, reason_code: str, message: str) -> None:
        """Store the stable reason code alongside the human-readable message."""

        super().__init__(message)
        self.reason_code = reason_code


def _classify_key(key: str) -> tuple[str, bool]:
    """Return the declared change class for a leaf key and its identity role."""

    change_class = _CLASS_BY_KEY.get(key.lower(), CHANGE_UNKNOWN)
    return change_class, change_class in _IDENTITY_CLASSES


def _detect_duplicate_keys(path: Path) -> None:
    """Fail closed when a YAML mapping declares the same key twice."""

    try:
        node = yaml.compose(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigDriftError("invalid_yaml", f"{path}: YAML parse error: {exc}") from exc
    if node is None:
        return

    def walk(current: Any) -> None:
        if isinstance(current, yaml.MappingNode):
            seen: set[str] = set()
            for key_node, value_node in current.value:
                key = str(key_node.value)
                if key in seen:
                    raise ConfigDriftError(
                        "duplicate_key",
                        f"{path}: duplicate mapping key {key!r}; refusing last-wins parsing",
                    )
                seen.add(key)
                walk(value_node)
        elif isinstance(current, yaml.SequenceNode):
            for item in current.value:
                walk(item)

    walk(node)


def _reject_unresolved_interpolation(value: Any, path: str) -> None:
    """Fail closed on environment/interpolation placeholders in resolved values."""

    if isinstance(value, str) and ("${" in value or "$(" in value):
        raise ConfigDriftError(
            "unresolved_interpolation",
            f"{path}: unresolved interpolation placeholder {value!r}; "
            "refusing to compare an unresolved value",
        )
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_unresolved_interpolation(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_unresolved_interpolation(item, f"{path}[{index}]")


def _sanitize_path_like(value: Any) -> Any:
    """Replace absolute path prefixes and reject parent-directory escapes."""

    if isinstance(value, str):
        if ".." in Path(value).parts:
            raise ConfigDriftError(
                "path_escape",
                f"value {value!r} escapes its configuration root; refusing to compare",
            )
        candidate = Path(value)
        if candidate.is_absolute():
            return f"<abs>/{candidate.name}"
        return value
    if isinstance(value, dict):
        return {key: _sanitize_path_like(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_path_like(item) for item in value]
    return value


def load_resolved_config(path: str | Path) -> dict[str, Any]:
    """Resolve one config through the canonical inheritance owner.

    Args:
        path: YAML config path; ``base_config`` inheritance is expanded by the
            canonical training-config resolver.

    Returns:
        The deep-merged, resolved mapping with ``base_config`` removed.

    Raises:
        ConfigDriftError: If the input is missing, malformed, duplicated,
            unresolved, or escapes its configuration root.
    """

    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigDriftError("missing_config", f"config not found: {config_path}")
    _detect_duplicate_keys(config_path)
    try:
        resolved = _load_expert_training_config_mapping(config_path)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise ConfigDriftError("invalid_schema", f"{config_path}: {exc}") from exc
    if not isinstance(resolved, dict):
        raise ConfigDriftError("invalid_schema", f"{config_path}: resolved config is not a mapping")
    _reject_unresolved_interpolation(resolved, config_path.name)
    return _sanitize_path_like(resolved)


def _leaf_paths(mapping: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a mapping into dotted-path leaves; lists stay whole values."""

    if isinstance(mapping, dict):
        leaves: dict[str, Any] = {}
        for key in sorted(mapping):
            child = f"{prefix}.{key}" if prefix else str(key)
            leaves.update(_leaf_paths(mapping[key], child))
        return leaves
    return {prefix: mapping}


def _declared_origins(path: Path, seen: frozenset[Path] = frozenset()) -> dict[str, str]:
    """Map each declared leaf path to the file that declares it.

    ``base_config`` inheritance is followed explicitly so the report can name
    the owning file for inherited values without re-merging them.

    Returns:
        A mapping of dotted leaf path to declaring config path.
    """

    resolved = path.resolve()
    if resolved in seen or not resolved.is_file():
        return {}
    try:
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    if not isinstance(raw, dict):
        return {}

    origins: dict[str, str] = {}
    for leaf in _leaf_paths({key: value for key, value in raw.items() if key != "base_config"}):
        origins[leaf] = _display_path(resolved)
    base_raw = raw.get("base_config")
    if base_raw is not None:
        base_path = Path(str(base_raw))
        if not base_path.is_absolute():
            base_path = resolved.parent / base_path
        for leaf, owner in _declared_origins(base_path, seen | frozenset({resolved})).items():
            origins.setdefault(leaf, owner)
    return origins


def _canonical_digest(mapping: dict[str, Any]) -> str:
    """Return the SHA-256 digest of a canonicalized mapping."""

    payload = json.dumps(mapping, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _comparability_digest(leaves: dict[str, Any]) -> str:
    """Digest only the identity-contributing leaves of one config."""

    identity = {path: leaves[path] for path in leaves if _classify_key(path.rsplit(".", 1)[-1])[1]}
    return _canonical_digest(identity)


def _display_path(path: str | Path) -> str:
    """Return a repo-relative display path without leaking absolute prefixes.

    Returns:
        The path relative to the current working directory when possible, or an
        ``<abs>/<basename>`` placeholder for paths outside it.
    """

    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return f"<abs>/{resolved.name}"


def _family_markers(mapping: dict[str, Any]) -> dict[str, Any]:
    """Return the declared config-family markers used for compatibility checks."""

    markers: dict[str, Any] = {}
    for key in ("algo", "category", "config_family", "family"):
        value = mapping.get(key)
        if value is not None:
            markers[key] = value
    return markers


def compare_resolved_configs(left: str | Path, right: str | Path) -> dict[str, Any]:
    """Compare two resolved configs and classify every changed leaf.

    Args:
        left: Left-hand YAML config path.
        right: Right-hand YAML config path.

    Returns:
        A deterministic ``resolved_config_drift.v1`` report dictionary.

    Raises:
        ConfigDriftError: If either input fails closed.
    """

    left_mapping = load_resolved_config(left)
    right_mapping = load_resolved_config(right)
    left_leaves = _leaf_paths(left_mapping)
    right_leaves = _leaf_paths(right_mapping)
    left_origins = _declared_origins(Path(left))
    right_origins = _declared_origins(Path(right))

    changes: list[dict[str, Any]] = []
    for path in sorted(set(left_leaves) | set(right_leaves)):
        left_value = left_leaves.get(path)
        right_value = right_leaves.get(path)
        if left_value == right_value and path in left_leaves and path in right_leaves:
            continue
        change_class, identity = _classify_key(path.rsplit(".", 1)[-1])
        changes.append(
            {
                "path": path,
                "left": left_value,
                "right": right_value,
                "left_origin": left_origins.get(path, "absent"),
                "right_origin": right_origins.get(path, "absent"),
                "change_class": change_class,
                "contributes_to_identity": identity,
            }
        )

    left_family = _family_markers(left_mapping)
    right_family = _family_markers(right_mapping)
    incompatible_family = bool(left_family and right_family and left_family != right_family)

    left_digest = _canonical_digest(left_mapping)
    right_digest = _canonical_digest(right_mapping)
    left_identity_digest = _comparability_digest(left_leaves)
    right_identity_digest = _comparability_digest(right_leaves)

    if incompatible_family or any(change["contributes_to_identity"] for change in changes):
        verdict = VERDICT_NOT_COMPARABLE
    elif not changes:
        verdict = VERDICT_IDENTICAL
    else:
        verdict = VERDICT_COMPARABLE

    reason_codes: list[str] = []
    if incompatible_family:
        reason_codes.append("incompatible_config_families")
    if any(change["change_class"] == CHANGE_SEMANTIC for change in changes):
        reason_codes.append("semantic_difference")
    if any(change["change_class"] == CHANGE_UNKNOWN for change in changes):
        reason_codes.append("unknown_difference")

    return {
        "schema": REPORT_SCHEMA,
        "verdict": verdict,
        "reason_codes": reason_codes,
        "left": {
            "path": _display_path(left),
            "digest": left_digest,
            "identity_digest": left_identity_digest,
        },
        "right": {
            "path": _display_path(right),
            "digest": right_digest,
            "identity_digest": right_identity_digest,
        },
        "identity_equivalent": left_identity_digest == right_identity_digest,
        "family_markers": {"left": left_family, "right": right_family},
        "changes": changes,
    }


def _format_text(report: dict[str, Any]) -> str:
    """Render a compact human-readable drift report."""

    lines = [
        f"verdict: {report['verdict']} ({', '.join(report['reason_codes']) or 'no reason codes'})",
        f"left  digest: {report['left']['digest']}",
        f"right digest: {report['right']['digest']}",
        f"identity equivalent: {report['identity_equivalent']}",
        "changes:",
    ]
    if not report["changes"]:
        lines.append("  (none)")
    for change in report["changes"]:
        lines.append(
            f"  - {change['path']}: {change['change_class']}"
            f" (identity={change['contributes_to_identity']})"
        )
        lines.append(f"      left : {change['left']!r}")
        lines.append(f"      right: {change['right']!r}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the resolved-config drift comparison.

    Returns:
        Process exit code: 0 for identical/comparable, 1 for not comparable,
        2 for a refused input or usage error.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True, help="Left-hand YAML config path.")
    parser.add_argument("--right", required=True, help="Right-hand YAML config path.")
    parser.add_argument("--json", action="store_true", help="Emit the stable JSON report.")
    args = parser.parse_args(argv)

    try:
        report = compare_resolved_configs(args.left, args.right)
    except ConfigDriftError as exc:
        print(json.dumps({"schema": REPORT_SCHEMA, "error": exc.reason_code, "message": str(exc)}))
        return 2

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_format_text(report))
    return 0 if report["verdict"] in {VERDICT_IDENTICAL, VERDICT_COMPARABLE} else 1


if __name__ == "__main__":
    sys.exit(main())
