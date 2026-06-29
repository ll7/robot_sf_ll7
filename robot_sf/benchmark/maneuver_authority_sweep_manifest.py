"""Fail-closed checker for issue #3213 maneuver-authority sweep manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = "maneuver-authority-sweep-manifest.v1"
MANIFEST_STATUS_READY = "ready"
MANIFEST_STATUS_MISSING = "missing"
MANIFEST_STATUS_MALFORMED = "malformed"
EXPECTED_OUTPUT_PREFIX = "output/benchmarks/issue_3213_maneuver_authority"

_REQUIRED_ARM_METADATA = (
    "action_lattice",
    "turn_authority",
    "kinematic_adapter",
    "expected_outputs",
)
_REQUIRED_ADAPTER_FIELDS = ("command_space", "robot_kinematics", "execution_mode")
_REQUIRED_OUTPUT_FIELDS = ("campaign_root", "campaign_summary", "campaign_report")
_MALFORMED_CODES = {
    "duplicate_arm_name",
    "invalid_action_lattice_metadata",
    "invalid_arm",
    "invalid_expected_output_path",
    "invalid_expected_output_root",
    "invalid_issue",
    "invalid_kinematic_adapter_metadata",
    "invalid_path",
    "invalid_schema_version",
    "invalid_turn_authority_metadata",
    "manifest_not_mapping",
    "missing_arm_metadata",
    "missing_arm_name",
    "missing_arms",
    "missing_grid_variant",
    "unknown_grid_variant",
}


def _load_yaml(path: Path) -> Any:
    """Load a YAML document.

    Returns:
        Parsed YAML content, or an empty mapping for an empty document.
    """
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _relative_path(value: Any) -> str | None:
    """Return ``value`` when it is a non-empty relative string path."""
    if not isinstance(value, str) or not value:
        return None
    if Path(value).is_absolute():
        return None
    return value


def _diagnostic(code: str, **details: Any) -> dict[str, Any]:
    """Build a compact structured diagnostic.

    Returns:
        Diagnostic dictionary with a stable ``code`` field plus caller details.
    """
    return {"code": code, **details}


def _missing_report(manifest_path: str | Path) -> dict[str, Any]:
    """Return a fail-closed missing-manifest report."""
    return {
        "schema_version": SCHEMA_VERSION,
        "status": MANIFEST_STATUS_MISSING,
        "issue": 3213,
        "manifest": str(manifest_path),
        "arms": [],
        "diagnostics": [_diagnostic("missing_manifest", path=str(manifest_path))],
    }


def _malformed_report(manifest_path: str | Path, code: str) -> dict[str, Any]:
    """Return a fail-closed malformed-manifest report."""
    return {
        "schema_version": SCHEMA_VERSION,
        "status": MANIFEST_STATUS_MALFORMED,
        "issue": 3213,
        "manifest": str(manifest_path),
        "arms": [],
        "diagnostics": [_diagnostic(code)],
    }


def _validate_repo_path(
    *,
    repo_root: Path,
    path_value: Any,
    field: str,
    code: str,
    diagnostics: list[dict[str, Any]],
    arm: str | None = None,
) -> str | None:
    """Validate a required manifest path and report diagnostics.

    Returns:
        The relative path string when syntactically valid; otherwise ``None``.
    """
    relative = _relative_path(path_value)
    if relative is None:
        diagnostics.append(_diagnostic("invalid_path", field=field, arm=arm))
        return None
    if not (repo_root / relative).exists():
        diagnostics.append(_diagnostic(code, field=field, path=relative, arm=arm))
    return relative


def _grid_variant_names(repo_root: Path, grid_path: str | None) -> set[str]:
    """Return declared grid variant names, or an empty set when unavailable."""
    if grid_path is None:
        return set()
    path = repo_root / grid_path
    if not path.exists():
        return set()
    payload = _load_yaml(path)
    variants = payload.get("variants", []) if isinstance(payload, dict) else []
    return {
        str(variant.get("name"))
        for variant in variants
        if isinstance(variant, dict) and variant.get("name")
    }


def _require_metadata_dicts(
    arm: dict[str, Any],
    arm_name: str,
    diagnostics: list[dict[str, Any]],
) -> None:
    """Report missing top-level arm metadata dictionaries."""
    for field in _REQUIRED_ARM_METADATA:
        if not isinstance(arm.get(field), dict):
            diagnostics.append(_diagnostic("missing_arm_metadata", arm=arm_name, field=field))


def _validate_action_lattice(
    arm: dict[str, Any],
    arm_name: str,
    diagnostics: list[dict[str, Any]],
) -> None:
    """Validate action-lattice metadata fields."""
    action_lattice = arm.get("action_lattice", {})
    if not isinstance(action_lattice, dict):
        return
    for field in ("candidate_speed_count", "heading_delta_count"):
        if not isinstance(action_lattice.get(field), int) or action_lattice[field] < 1:
            diagnostics.append(
                _diagnostic("invalid_action_lattice_metadata", arm=arm_name, field=field)
            )


def _validate_turn_authority(
    arm: dict[str, Any],
    arm_name: str,
    diagnostics: list[dict[str, Any]],
) -> None:
    """Validate turn-authority metadata fields."""
    turn_authority = arm.get("turn_authority", {})
    if not isinstance(turn_authority, dict):
        return
    if not isinstance(turn_authority.get("max_angular_speed"), int | float):
        diagnostics.append(
            _diagnostic(
                "invalid_turn_authority_metadata",
                arm=arm_name,
                field="max_angular_speed",
            )
        )
    if not isinstance(turn_authority.get("heading_delta_count"), int):
        diagnostics.append(
            _diagnostic(
                "invalid_turn_authority_metadata",
                arm=arm_name,
                field="heading_delta_count",
            )
        )


def _validate_adapter(
    arm: dict[str, Any],
    arm_name: str,
    diagnostics: list[dict[str, Any]],
) -> None:
    """Validate kinematic-adapter metadata fields."""
    adapter = arm.get("kinematic_adapter", {})
    if not isinstance(adapter, dict):
        return
    for field in _REQUIRED_ADAPTER_FIELDS:
        if not isinstance(adapter.get(field), str) or not adapter[field]:
            diagnostics.append(
                _diagnostic("invalid_kinematic_adapter_metadata", arm=arm_name, field=field)
            )


def _validate_expected_outputs(
    *,
    arm: dict[str, Any],
    arm_name: str,
    diagnostics: list[dict[str, Any]],
) -> None:
    """Validate declared output locations without requiring generated outputs."""
    outputs = arm.get("expected_outputs", {})
    if not isinstance(outputs, dict):
        return
    allowed_prefixes = (EXPECTED_OUTPUT_PREFIX, f"{EXPECTED_OUTPUT_PREFIX}/")
    for field in _REQUIRED_OUTPUT_FIELDS:
        relative = _relative_path(outputs.get(field))
        if relative is None:
            diagnostics.append(
                _diagnostic("invalid_expected_output_path", arm=arm_name, field=field)
            )
        elif not relative.startswith(allowed_prefixes):
            diagnostics.append(
                _diagnostic(
                    "invalid_expected_output_root",
                    arm=arm_name,
                    field=field,
                    path=relative,
                    expected_prefix=EXPECTED_OUTPUT_PREFIX,
                )
            )


def _validate_arm_metadata(
    arm: dict[str, Any],
    arm_name: str,
    diagnostics: list[dict[str, Any]],
) -> None:
    """Validate all per-arm metadata blocks."""
    _require_metadata_dicts(arm, arm_name, diagnostics)
    _validate_action_lattice(arm, arm_name, diagnostics)
    _validate_turn_authority(arm, arm_name, diagnostics)
    _validate_adapter(arm, arm_name, diagnostics)
    _validate_expected_outputs(arm=arm, arm_name=arm_name, diagnostics=diagnostics)


def _validate_header(
    payload: dict[str, Any],
    diagnostics: list[dict[str, Any]],
) -> None:
    """Validate manifest header identity fields."""
    if payload.get("schema_version") != SCHEMA_VERSION:
        diagnostics.append(
            _diagnostic(
                "invalid_schema_version",
                expected=SCHEMA_VERSION,
                actual=payload.get("schema_version"),
            )
        )
    if payload.get("issue") != 3213:
        diagnostics.append(_diagnostic("invalid_issue", expected=3213, actual=payload.get("issue")))


def _validate_shared_paths(
    payload: dict[str, Any],
    repo_root: Path,
    diagnostics: list[dict[str, Any]],
) -> set[str]:
    """Validate shared manifest paths.

    Returns:
        Known variant names from the referenced benchmark grid, or an empty set.
    """
    grid_path = _validate_repo_path(
        repo_root=repo_root,
        path_value=payload.get("benchmark_grid"),
        field="benchmark_grid",
        code="missing_benchmark_grid",
        diagnostics=diagnostics,
    )
    _validate_repo_path(
        repo_root=repo_root,
        path_value=payload.get("hard_seed_manifest"),
        field="hard_seed_manifest",
        code="missing_hard_seed_manifest",
        diagnostics=diagnostics,
    )
    return _grid_variant_names(repo_root, grid_path)


def _validate_arm_identity(
    *,
    arm: dict[str, Any],
    seen_names: set[str],
    diagnostics: list[dict[str, Any]],
) -> str | None:
    """Validate a sweep arm name.

    Returns:
        The arm name when valid; otherwise ``None``.
    """
    arm_name = arm.get("name")
    if not isinstance(arm_name, str) or not arm_name:
        diagnostics.append(_diagnostic("missing_arm_name"))
        return None
    if arm_name in seen_names:
        diagnostics.append(_diagnostic("duplicate_arm_name", arm=arm_name))
    seen_names.add(arm_name)
    return arm_name


def _validate_grid_variant(
    *,
    arm: dict[str, Any],
    arm_name: str,
    grid_variants: set[str],
    diagnostics: list[dict[str, Any]],
) -> Any:
    """Validate the declared grid variant.

    Returns:
        The raw grid variant value from the arm payload.
    """
    grid_variant = arm.get("grid_variant")
    if not isinstance(grid_variant, str) or not grid_variant:
        diagnostics.append(_diagnostic("missing_grid_variant", arm=arm_name))
    elif grid_variants and grid_variant not in grid_variants:
        diagnostics.append(
            _diagnostic("unknown_grid_variant", arm=arm_name, grid_variant=grid_variant)
        )
    return grid_variant


def _report_arm(arm: dict[str, Any], arm_name: str, grid_variant: Any) -> dict[str, Any]:
    """Return the normalized arm payload used in checker reports."""
    return {
        "name": arm_name,
        "grid_variant": grid_variant,
        "algo_config": arm.get("algo_config"),
        "action_lattice": arm.get("action_lattice"),
        "turn_authority": arm.get("turn_authority"),
        "kinematic_adapter": arm.get("kinematic_adapter"),
        "expected_outputs": arm.get("expected_outputs"),
    }


def _validate_arms(
    payload: dict[str, Any],
    repo_root: Path,
    grid_variants: set[str],
    diagnostics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Validate all sweep arms.

    Returns:
        Normalized arm entries used by the checker report.
    """
    arms = payload.get("arms")
    if not isinstance(arms, list) or not arms:
        diagnostics.append(_diagnostic("missing_arms"))
        return []

    seen_names: set[str] = set()
    report_arms: list[dict[str, Any]] = []
    for arm in arms:
        if not isinstance(arm, dict):
            diagnostics.append(_diagnostic("invalid_arm"))
            continue
        arm_name = _validate_arm_identity(
            arm=arm,
            seen_names=seen_names,
            diagnostics=diagnostics,
        )
        if arm_name is None:
            continue
        _validate_repo_path(
            repo_root=repo_root,
            path_value=arm.get("algo_config"),
            field="algo_config",
            code="missing_algo_config",
            diagnostics=diagnostics,
            arm=arm_name,
        )
        grid_variant = _validate_grid_variant(
            arm=arm,
            arm_name=arm_name,
            grid_variants=grid_variants,
            diagnostics=diagnostics,
        )
        _validate_arm_metadata(arm, arm_name, diagnostics)
        report_arms.append(_report_arm(arm, arm_name, grid_variant))
    return report_arms


def _status_for_diagnostics(diagnostics: list[dict[str, Any]]) -> str:
    """Return the fail-closed status implied by diagnostics."""
    if any(diagnostic["code"] in _MALFORMED_CODES for diagnostic in diagnostics):
        return MANIFEST_STATUS_MALFORMED
    if diagnostics:
        return MANIFEST_STATUS_MISSING
    return MANIFEST_STATUS_READY


def check_maneuver_authority_sweep_manifest(
    manifest_path: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return a fail-closed readiness report for a maneuver-authority sweep manifest.

    The checker validates only preflight metadata and repository-local inputs. It does not
    execute a sweep, inspect generated results, or promote any benchmark interpretation.
    """
    root = Path.cwd() if repo_root is None else Path(repo_root)
    root = root.resolve()
    manifest = Path(manifest_path)
    resolved_manifest = manifest if manifest.is_absolute() else root / manifest
    if not resolved_manifest.exists():
        return _missing_report(manifest_path)

    payload = _load_yaml(resolved_manifest)
    if not isinstance(payload, dict):
        return _malformed_report(manifest_path, "manifest_not_mapping")

    diagnostics: list[dict[str, Any]] = []
    _validate_header(payload, diagnostics)
    grid_variants = _validate_shared_paths(payload, root, diagnostics)
    report_arms = _validate_arms(payload, root, grid_variants, diagnostics)

    return {
        "schema_version": SCHEMA_VERSION,
        "status": _status_for_diagnostics(diagnostics),
        "issue": payload.get("issue"),
        "manifest": str(manifest_path),
        "benchmark_grid": payload.get("benchmark_grid"),
        "hard_seed_manifest": payload.get("hard_seed_manifest"),
        "default_output_root": payload.get("default_output_root"),
        "arms": report_arms,
        "diagnostics": diagnostics,
        "scope_note": (
            "Metadata-only preflight; no benchmark campaign execution or success interpretation."
        ),
    }
