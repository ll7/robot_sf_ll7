#!/usr/bin/env python3
"""Fail-closed readiness checker for the Package A benchmark (issue #3078).

Package A is the seed/planner-rank stability plus held-out scenario-family transfer
result for the dissertation benchmark (parent issue #3057). Before that campaign is
executed, several input prerequisites must already exist: the held-out-family scenario
inputs, the seed-plan tooling, and the frozen protocol / result-store entry points.

This checker is deliberately bounded. It:

- loads a Package A readiness manifest (see
  ``configs/benchmarks/issue_3078_package_a_readiness.yaml``),
- verifies that every declared input prerequisite path exists on disk,
- verifies the seed-plan metadata fields are present and non-empty,
- verifies the declared output location is safe (under ``output/`` and disposable),
- and **fails closed** (non-zero exit, ``status: not_ready``) when any prerequisite is
  missing or malformed.

It does NOT execute the benchmark, submit Slurm/GPU jobs, interpret ranks, or move any
claim boundary. A passing check is provenance/readiness evidence only, never benchmark
evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

SCHEMA_VERSION = "package_a_readiness_report.v1"

# Sections the manifest must declare for the readiness check to be meaningful.
REQUIRED_SECTIONS = (
    "package",
    "heldout_family_inputs",
    "seed_plan",
    "frozen_protocol",
    "outputs",
)


class ManifestError(ValueError):
    """Raised when the manifest is structurally unusable (cannot run the check)."""


@dataclass
class ReadinessReport:
    """Structured, fail-closed readiness verdict for Package A inputs."""

    manifest_path: str
    package_id: str
    schema_version: str = SCHEMA_VERSION
    checked_paths: list[dict[str, Any]] = field(default_factory=list)
    missing_paths: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        """Return ``ready`` only when nothing is missing and no structural issue exists."""
        return "ready" if not self.missing_paths and not self.issues else "not_ready"

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report to a JSON-friendly mapping."""
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "manifest_path": self.manifest_path,
            "package_id": self.package_id,
            "checked_paths": self.checked_paths,
            "missing_paths": self.missing_paths,
            "issues": self.issues,
        }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_manifest(path: Path) -> dict[str, Any]:
    """Load the manifest YAML, raising ManifestError on a non-mapping document."""
    if not path.exists():
        raise ManifestError(f"Manifest not found: {path}")
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ManifestError("Manifest root must be a mapping")
    missing_sections = [s for s in REQUIRED_SECTIONS if s not in manifest]
    if missing_sections:
        raise ManifestError(f"Manifest missing required sections: {sorted(missing_sections)}")
    return manifest


def _required_paths(section: dict[str, Any], section_name: str) -> list[str]:
    """Return the ``required_paths`` list for a section, validating its shape."""
    raw = section.get("required_paths", [])
    if not isinstance(raw, list) or not raw:
        raise ManifestError(f"{section_name}.required_paths must be a non-empty list")
    paths: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            raise ManifestError(f"{section_name}.required_paths entries must be non-empty strings")
        paths.append(item.strip())
    return paths


def _check_paths(
    repo_root: Path,
    section: dict[str, Any],
    section_name: str,
    report: ReadinessReport,
) -> None:
    """Record existence of every required path in a section into the report."""
    for rel_path in _required_paths(section, section_name):
        candidate = repo_root / rel_path
        exists = candidate.exists()
        report.checked_paths.append({"section": section_name, "path": rel_path, "exists": exists})
        if not exists:
            report.missing_paths.append(rel_path)


def _check_optional_tool(
    repo_root: Path,
    section: dict[str, Any],
    key: str,
    section_name: str,
    report: ReadinessReport,
) -> None:
    """Check a single optional tool path declared under ``key`` if present."""
    tool = section.get(key)
    if tool is None:
        return
    if not isinstance(tool, str) or not tool.strip():
        report.issues.append(f"{section_name}.{key} must be a non-empty string when present")
        return
    rel_path = tool.strip()
    exists = (repo_root / rel_path).exists()
    report.checked_paths.append(
        {"section": f"{section_name}.{key}", "path": rel_path, "exists": exists}
    )
    if not exists:
        report.missing_paths.append(rel_path)


def _check_seed_plan_metadata(seed_plan: dict[str, Any], report: ReadinessReport) -> None:
    """Verify the declared seed-plan metadata fields are present and non-empty."""
    required_metadata = seed_plan.get("required_metadata", [])
    if not isinstance(required_metadata, list) or not required_metadata:
        report.issues.append("seed_plan.required_metadata must be a non-empty list")
        return
    for key in required_metadata:
        value = seed_plan.get(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            report.issues.append(f"seed_plan metadata field '{key}' is missing or empty")


def _check_outputs(outputs: dict[str, Any], report: ReadinessReport) -> None:
    """Verify the declared output location is safe (under output/, disposable)."""
    local_root = str(outputs.get("local_root", "")).strip()
    if not local_root:
        report.issues.append("outputs.local_root is required")
        return
    if PurePosixPath(local_root).parts[:1] != ("output",):
        report.issues.append("outputs.local_root must stay under output/")
    if outputs.get("disposable") is not True:
        report.issues.append("outputs.disposable must be true for local output roots")


def check_readiness(manifest_path: Path, *, repo_root: Path | None = None) -> ReadinessReport:
    """Run the full Package A readiness check and return a structured report."""
    repo_root = repo_root or _repo_root()
    manifest = _load_manifest(manifest_path)
    package_id = str(manifest.get("package", {}).get("id", "unknown"))
    report = ReadinessReport(manifest_path=str(manifest_path), package_id=package_id)

    heldout = manifest["heldout_family_inputs"]
    _check_paths(repo_root, heldout, "heldout_family_inputs", report)
    _check_optional_tool(repo_root, heldout, "leakage_audit_tool", "heldout_family_inputs", report)

    seed_plan = manifest["seed_plan"]
    _check_paths(repo_root, seed_plan, "seed_plan", report)
    _check_seed_plan_metadata(seed_plan, report)

    _check_paths(repo_root, manifest["frozen_protocol"], "frozen_protocol", report)
    _check_outputs(manifest["outputs"], report)

    return report


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_repo_root() / "configs" / "benchmarks" / "issue_3078_package_a_readiness.yaml",
        help="Path to the Package A readiness manifest.",
    )
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Emit the readiness report as JSON on stdout.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns 0 when ready, 1 when not ready, 2 on manifest error."""
    args = _parse_args(argv)
    try:
        report = check_readiness(args.manifest)
    except ManifestError as exc:
        if args.as_json:
            print(
                json.dumps({"schema_version": SCHEMA_VERSION, "status": "error", "error": str(exc)})
            )
        else:
            print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    payload = report.to_dict()
    if args.as_json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Package A readiness: {payload['status']} ({payload['package_id']})")
        for missing in payload["missing_paths"]:
            print(f"  missing prerequisite: {missing}")
        for issue in payload["issues"]:
            print(f"  issue: {issue}")

    return 0 if report.status == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
