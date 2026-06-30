#!/usr/bin/env python3
"""Fail-closed preflight for the issue #3213 maneuver-authority sweep manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = "maneuver_authority_sweep_manifest.v1"
DEFAULT_MANIFEST = Path("configs/benchmarks/predictive_hardcase_authority_grid_issue_3213.yaml")

_ACTION_LATTICE_KEYS = {
    "inherits_defaults",
    "candidate_heading_deltas_rad",
    "candidate_speed_samples_m_s",
    "near_field_heading_deltas_rad",
    "near_field_speed_cap_m_s",
}


def _as_mapping(value: Any, label: str, errors: list[str]) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    errors.append(f"{label} must be a mapping")
    return {}


def _as_list(value: Any, label: str, errors: list[str]) -> list[Any]:
    if isinstance(value, list):
        return value
    errors.append(f"{label} must be a list")
    return []


def _repo_path(repo_root: Path, value: Any, label: str, errors: list[str]) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{label} must be a non-empty string path")
        return None
    path = Path(value)
    if path.is_absolute():
        errors.append(f"{label} must be repo-relative: {value}")
        return None
    return repo_root / path


def _require_existing_path(repo_root: Path, value: Any, label: str, errors: list[str]) -> None:
    path = _repo_path(repo_root, value, label, errors)
    if path is not None and not path.exists():
        errors.append(f"{label} does not exist: {value}")


def _validate_expected_output(repo_root: Path, payload: dict[str, Any], errors: list[str]) -> None:
    expected_output = _as_mapping(payload.get("expected_output"), "expected_output", errors)
    for key in ("root", "summary", "report"):
        _repo_path(repo_root, expected_output.get(key), f"expected_output.{key}", errors)
    root = expected_output.get("root")
    if isinstance(root, str) and not root.startswith("output/"):
        errors.append("expected_output.root must stay under output/")


def _validate_variant_metadata(
    *,
    repo_root: Path,
    variant: dict[str, Any],
    variant_name: str,
    errors: list[str],
) -> None:
    _require_existing_path(
        repo_root, variant.get("algo_config"), f"{variant_name}.algo_config", errors
    )

    expected_output = _as_mapping(
        variant.get("expected_output"),
        f"{variant_name}.expected_output",
        errors,
    )
    for key in ("hard_jsonl_pattern", "global_jsonl_pattern"):
        _repo_path(
            repo_root, expected_output.get(key), f"{variant_name}.expected_output.{key}", errors
        )

    metadata = _as_mapping(
        variant.get("authority_metadata"),
        f"{variant_name}.authority_metadata",
        errors,
    )
    action_lattice = _as_mapping(
        metadata.get("action_lattice"),
        f"{variant_name}.authority_metadata.action_lattice",
        errors,
    )
    if action_lattice and not (_ACTION_LATTICE_KEYS & set(action_lattice)):
        errors.append(f"{variant_name}.authority_metadata.action_lattice has no known lattice keys")

    turn_authority = _as_mapping(
        metadata.get("turn_authority"),
        f"{variant_name}.authority_metadata.turn_authority",
        errors,
    )
    if turn_authority and "max_angular_speed_rad_s" not in turn_authority:
        errors.append(
            f"{variant_name}.authority_metadata.turn_authority must include max_angular_speed_rad_s"
        )

    adapter = _as_mapping(
        metadata.get("kinematic_adapter"),
        f"{variant_name}.authority_metadata.kinematic_adapter",
        errors,
    )
    if adapter:
        mode = adapter.get("mode")
        if not isinstance(mode, str) or not mode.strip():
            errors.append(
                f"{variant_name}.authority_metadata.kinematic_adapter.mode "
                "must be a non-empty string"
            )
        _require_existing_path(
            repo_root,
            adapter.get("config_source"),
            f"{variant_name}.authority_metadata.kinematic_adapter.config_source",
            errors,
        )
        changed_params = _as_list(
            adapter.get("changed_params"),
            f"{variant_name}.authority_metadata.kinematic_adapter.changed_params",
            errors,
        )
        params = _as_mapping(variant.get("params", {}), f"{variant_name}.params", errors)
        missing_params = sorted(set(params) - set(changed_params))
        if missing_params:
            errors.append(
                f"{variant_name}.authority_metadata.kinematic_adapter.changed_params "
                f"does not cover params: {missing_params}"
            )


def build_report(manifest_path: Path, repo_root: Path) -> dict[str, Any]:
    """Return a machine-readable preflight report for a maneuver-authority manifest."""
    errors: list[str] = []
    manifest_path = manifest_path if manifest_path.is_absolute() else repo_root / manifest_path
    if not manifest_path.exists():
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "manifest": str(manifest_path),
            "errors": [f"manifest does not exist: {manifest_path}"],
        }

    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    payload = _as_mapping(payload, "manifest", errors)
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if payload.get("issue") != 3213:
        errors.append("issue must be 3213")
    _require_existing_path(
        repo_root, payload.get("hard_seed_manifest"), "hard_seed_manifest", errors
    )
    _validate_expected_output(repo_root, payload, errors)

    variants = _as_list(payload.get("variants"), "variants", errors)
    if len(variants) < 3:
        errors.append("variants must include at least three authority settings")

    names: set[str] = set()
    for index, item in enumerate(variants, start=1):
        variant = _as_mapping(item, f"variant #{index}", errors)
        name = variant.get("name")
        if not isinstance(name, str) or not name.strip():
            errors.append(f"variant #{index}.name must be a non-empty string")
            name = f"#{index}"
        if name in names:
            errors.append(f"duplicate variant name: {name}")
        names.add(name)
        _validate_variant_metadata(
            repo_root=repo_root, variant=variant, variant_name=name, errors=errors
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "failed" if errors else "ok",
        "manifest": str(manifest_path),
        "variant_count": len(variants),
        "variants": sorted(names),
        "errors": errors,
        "note": "Preflight only; does not run sweeps or interpret benchmark success.",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the fail-closed manifest preflight."""
    args = parse_args(argv)
    report = build_report(args.manifest, args.repo_root)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
