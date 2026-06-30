#!/usr/bin/env python3
"""Fail-closed readiness checker for Package A benchmark issue #3078.

Package A covers seed/planner-rank stability plus held-out scenario-family
transfer under parent issue #3057. This checker validates the readiness
decision packet only. It does not execute benchmark campaigns, submit compute,
interpret rankings, or move any paper-facing claim boundary.
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
DECISION_PACKET_SCHEMA_VERSION = "package_a_decision_packet.v1"

REQUIRED_SECTIONS = (
    "package",
    "heldout_family_inputs",
    "seed_plan",
    "frozen_protocol",
    "command_contracts",
    "outputs",
    "durable_evidence",
    "readiness_decision",
)


class ManifestError(ValueError):
    """Raised when the manifest is structurally unusable."""


@dataclass
class ReadinessReport:
    """Structured, fail-closed readiness verdict for Package A inputs."""

    manifest_path: str
    package_id: str
    schema_version: str = SCHEMA_VERSION
    checked_paths: list[dict[str, Any]] = field(default_factory=list)
    checked_commands: list[dict[str, Any]] = field(default_factory=list)
    missing_paths: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        """Return ``ready`` only when no missing path or structural issue exists."""
        return "ready" if not self.missing_paths and not self.issues else "not_ready"

    def to_dict(self) -> dict[str, Any]:
        """Serialize report as a JSON-friendly mapping."""
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "manifest_path": self.manifest_path,
            "package_id": self.package_id,
            "checked_paths": self.checked_paths,
            "checked_commands": self.checked_commands,
            "missing_paths": self.missing_paths,
            "issues": self.issues,
        }


@dataclass
class DecisionPacket:
    """Fail-closed Package A evidence-boundary decision packet."""

    manifest_path: str
    package_id: str
    readiness_status: str
    classification: str
    reasons: list[str] = field(default_factory=list)
    result_stores: list[dict[str, Any]] = field(default_factory=list)
    seed_analysis_reports: list[dict[str, Any]] = field(default_factory=list)
    heldout_partition_manifests: list[dict[str, Any]] = field(default_factory=list)
    schema_version: str = DECISION_PACKET_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Serialize packet JSON-friendly mapping."""
        return {
            "schema_version": self.schema_version,
            "package_id": self.package_id,
            "manifest_path": self.manifest_path,
            "readiness_status": self.readiness_status,
            "classification": self.classification,
            "reasons": self.reasons,
            "result_stores": self.result_stores,
            "seed_analysis_reports": self.seed_analysis_reports,
            "heldout_partition_manifests": self.heldout_partition_manifests,
            "forbidden_actions_confirmed": {
                "benchmark_campaign_run": False,
                "compute_submit": False,
                "ranking_claim_promotion": False,
                "paper_claim_edits": False,
            },
        }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _clean_str(value: Any) -> str:
    """Coerce a manifest scalar to a stripped string, treating ``None`` as empty.

    A bare ``str(None)`` yields the literal ``"None"``, which would otherwise
    pass downstream non-empty checks and let a ``null`` YAML field masquerade as
    a present value. Returning ``""`` for ``None`` keeps the checker fail-closed.
    """
    if value is None:
        return ""
    return str(value).strip()


def _load_manifest(path: Path) -> dict[str, Any]:
    """Load manifest YAML, raising ManifestError on non-mapping documents."""
    if not path.exists():
        raise ManifestError(f"Manifest not found: {path}")
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ManifestError("Manifest root must be mapping")
    missing_sections = [section for section in REQUIRED_SECTIONS if section not in manifest]
    if missing_sections:
        raise ManifestError(f"Manifest missing required sections: {sorted(missing_sections)}")
    non_mapping = [
        section for section in REQUIRED_SECTIONS if not isinstance(manifest[section], dict)
    ]
    if non_mapping:
        raise ManifestError(f"Manifest sections must be mappings: {sorted(non_mapping)}")
    return manifest


def _required_paths(section: dict[str, Any], section_name: str) -> list[str]:
    """Return ``required_paths`` for a section, validating shape."""
    raw = section.get("required_paths", [])
    if not isinstance(raw, list) or not raw:
        raise ManifestError(f"{section_name}.required_paths must be non-empty list")

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
    """Record required path existence for a manifest section."""
    for rel_path in _required_paths(section, section_name):
        exists = (repo_root / rel_path).exists()
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
    """Check a single optional tool path declared under ``key``."""
    tool = section.get(key)
    if tool is None:
        return
    if not isinstance(tool, str) or not tool.strip():
        report.issues.append(f"{section_name}.{key} must be non-empty string when present")
        return

    rel_path = tool.strip()
    exists = (repo_root / rel_path).exists()
    report.checked_paths.append(
        {"section": f"{section_name}.{key}", "path": rel_path, "exists": exists}
    )
    if not exists:
        report.missing_paths.append(rel_path)


def _check_seed_plan_metadata(seed_plan: dict[str, Any], report: ReadinessReport) -> None:
    """Verify declared seed-plan metadata fields are present and non-empty."""
    required_metadata = seed_plan.get("required_metadata", [])
    if not isinstance(required_metadata, list) or not required_metadata:
        report.issues.append("seed_plan.required_metadata must be non-empty list")
        return

    for key in required_metadata:
        value = seed_plan.get(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            report.issues.append(f"seed_plan metadata field '{key}' missing or empty")


def _check_command_contracts(
    repo_root: Path, command_contracts: dict[str, Any], report: ReadinessReport
) -> None:
    """Verify command contracts are explicit metadata, not work to run here."""
    contracts = command_contracts.get("contracts")
    if not isinstance(contracts, list) or not contracts:
        report.issues.append("command_contracts.contracts must be non-empty list")
        return

    for index, contract in enumerate(contracts):
        _check_command_contract(repo_root, contract, index, report)


def _check_command_contract(
    repo_root: Path, contract: Any, index: int, report: ReadinessReport
) -> None:
    """Verify one command contract and its declared path dependencies."""
    prefix = f"command_contracts.contracts[{index}]"
    if not isinstance(contract, dict):
        report.issues.append(f"{prefix} must be mapping")
        return

    contract_id = _clean_str(contract.get("id"))
    stage = _clean_str(contract.get("stage"))
    command = _clean_str(contract.get("command"))
    executes_benchmark = contract.get("executes_benchmark_campaign")
    allowed_here = contract.get("allowed_in_readiness_check")

    if not contract_id:
        report.issues.append(f"{prefix}.id is required")
    if not stage:
        report.issues.append(f"{prefix}.stage is required")
    if not command:
        report.issues.append(f"{prefix}.command is required")
    if not isinstance(executes_benchmark, bool):
        report.issues.append(f"{prefix}.executes_benchmark_campaign must be boolean")
    if not isinstance(allowed_here, bool):
        report.issues.append(f"{prefix}.allowed_in_readiness_check must be boolean")
    if executes_benchmark is True and allowed_here is True:
        report.issues.append(
            f"{prefix} cannot both execute benchmark campaign and be allowed in readiness check"
        )

    report.checked_commands.append(
        {
            "id": contract_id,
            "stage": stage,
            "allowed_in_readiness_check": allowed_here,
            "executes_benchmark_campaign": executes_benchmark,
        }
    )

    for rel_path in _required_paths(contract, prefix):
        exists = (repo_root / rel_path).exists()
        report.checked_paths.append({"section": prefix, "path": rel_path, "exists": exists})
        if not exists:
            report.missing_paths.append(rel_path)


def _check_outputs(outputs: dict[str, Any], report: ReadinessReport) -> None:
    """Verify declared output location is safe and disposable."""
    local_root = _clean_str(outputs.get("local_root"))
    if not local_root:
        report.issues.append("outputs.local_root is required")
        return
    if PurePosixPath(local_root).parts[:1] != ("output",):
        report.issues.append("outputs.local_root must stay under output/")
    if outputs.get("disposable") is not True:
        report.issues.append("outputs.disposable must be true for local output roots")


def _check_durable_evidence(durable_evidence: dict[str, Any], report: ReadinessReport) -> None:
    """Verify durable evidence plan exists outside disposable output."""
    plan = durable_evidence.get("plan")
    if not isinstance(plan, dict):
        report.issues.append("durable_evidence.plan must be mapping")
        return

    rel_path = _clean_str(plan.get("path"))
    if not rel_path:
        report.issues.append("durable_evidence.plan.path is required")
    elif PurePosixPath(rel_path).parts[:1] == ("output",):
        report.issues.append("durable_evidence.plan.path must not point under output/")

    if plan.get("required_before_claim") is not True:
        report.issues.append("durable_evidence.plan.required_before_claim must be true")


def _check_readiness_decision(decision: dict[str, Any], report: ReadinessReport) -> None:
    """Fail closed unless compute and claim promotion are explicitly disabled."""
    required_false = {
        "benchmark_campaign_run": "must remain false in readiness-only PR",
        "compute_submit_authorized": "must remain false without compute authorization",
        "ranking_claim_promotion": "must remain false until durable evidence is classified",
        "paper_claim_edits": "must remain false in readiness-only PR",
        "fallback_degraded_success_allowed": "must remain false under benchmark policy",
    }
    for key, message in required_false.items():
        if decision.get(key) is not False:
            report.issues.append(f"readiness_decision.{key} {message}")

    if decision.get("result_classification_required") is not True:
        report.issues.append("readiness_decision.result_classification_required must be true")


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
    _check_command_contracts(repo_root, manifest["command_contracts"], report)
    _check_outputs(manifest["outputs"], report)
    _check_durable_evidence(manifest["durable_evidence"], report)
    _check_readiness_decision(manifest["readiness_decision"], report)
    return report


def build_decision_packet(
    manifest_path: Path,
    *,
    repo_root: Path | None = None,
    result_stores: list[Path] | None = None,
    seed_analysis_reports: list[Path] | None = None,
    heldout_partition_manifests: list[Path] | None = None,
) -> DecisionPacket:
    """Build fail-closed Package A decision packet from supplied evidence surfaces."""
    repo_root = repo_root or _repo_root()
    readiness = check_readiness(manifest_path, repo_root=repo_root)
    packet = DecisionPacket(
        manifest_path=str(manifest_path),
        package_id=readiness.package_id,
        readiness_status=readiness.status,
        classification="diagnostic_review_ready",
    )

    if readiness.status != "ready":
        packet.reasons.extend(
            [f"readiness missing path: {path}" for path in readiness.missing_paths]
        )
        packet.reasons.extend(readiness.issues)

    packet.result_stores = [
        _validate_result_store_path(repo_root, path) for path in (result_stores or [])
    ]
    packet.seed_analysis_reports = [
        _validate_seed_analysis_report(repo_root, path) for path in (seed_analysis_reports or [])
    ]
    packet.heldout_partition_manifests = [
        _validate_heldout_partition_manifest(repo_root, path)
        for path in (heldout_partition_manifests or [])
    ]

    if not packet.result_stores:
        packet.reasons.append("no canonical campaign result store supplied")
    if not packet.seed_analysis_reports:
        packet.reasons.append("no seed-sufficiency analysis report supplied")
    if not packet.heldout_partition_manifests:
        packet.reasons.append("no held-out-family partition manifest supplied")

    for surface_name in (
        "result_stores",
        "seed_analysis_reports",
        "heldout_partition_manifests",
    ):
        for checked in getattr(packet, surface_name):
            packet.reasons.extend(checked.get("errors", []))

    packet.classification = (
        "diagnostic_review_ready" if not packet.reasons else "blocked_pending_package_a_evidence"
    )
    return packet


def _resolve_repo_path(repo_root: Path, path: Path) -> Path:
    """Resolve absolute or repository-relative path without requiring existence."""
    return path if path.is_absolute() else repo_root / path


def _display_path(repo_root: Path, path: Path) -> str:
    """Return path relative to repo root when possible."""
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _validate_result_store_path(repo_root: Path, path: Path) -> dict[str, Any]:
    """Validate one canonical campaign result store path."""
    from scripts.tools.campaign_result_store import validate_result_store

    resolved = _resolve_repo_path(repo_root, path)
    if not resolved.exists():
        return {
            "path": _display_path(repo_root, resolved),
            "ok": False,
            "errors": [f"result store does not exist: {_display_path(repo_root, resolved)}"],
        }
    validation = validate_result_store(resolved)
    return {
        "path": _display_path(repo_root, resolved),
        "ok": validation.ok,
        "errors": validation.errors,
    }


def _validate_seed_analysis_report(repo_root: Path, path: Path) -> dict[str, Any]:
    """Validate one seed-sufficiency analysis JSON report."""
    resolved = _resolve_repo_path(repo_root, path)
    if not resolved.is_file():
        return {
            "path": _display_path(repo_root, resolved),
            "ok": False,
            "errors": [
                f"seed-sufficiency analysis report does not exist: "
                f"{_display_path(repo_root, resolved)}"
            ],
        }
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "path": _display_path(repo_root, resolved),
            "ok": False,
            "errors": [f"seed-sufficiency analysis report is unreadable: {exc}"],
        }

    errors: list[str] = []
    if payload.get("schema_version") != "seed_sufficiency_analysis.v1":
        errors.append("seed-sufficiency analysis report schema_version must be v1")
    if not isinstance(payload.get("headline_rank_stability_contract"), dict):
        errors.append("seed-sufficiency analysis report missing headline rank-stability contract")
    if not isinstance(payload.get("rank_stability"), list):
        errors.append("seed-sufficiency analysis report missing rank_stability rows")
    return {
        "path": _display_path(repo_root, resolved),
        "ok": not errors,
        "errors": errors,
    }


def _validate_heldout_partition_manifest(repo_root: Path, path: Path) -> dict[str, Any]:
    """Validate one held-out-family partition manifest."""
    from scripts.tools.validate_heldout_transfer_partitions import validate_partition_manifest

    resolved = _resolve_repo_path(repo_root, path)
    if not resolved.is_file():
        return {
            "path": _display_path(repo_root, resolved),
            "ok": False,
            "errors": [
                f"held-out-family partition manifest does not exist: "
                f"{_display_path(repo_root, resolved)}"
            ],
        }
    errors = validate_partition_manifest(resolved)
    return {
        "path": _display_path(repo_root, resolved),
        "ok": not errors,
        "errors": errors,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_repo_root() / "configs" / "benchmarks" / "issue_3078_package_a_readiness.yaml",
        help="Path to Package A readiness manifest.",
    )
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Emit readiness report as JSON on stdout.",
    )
    parser.add_argument(
        "--decision-packet",
        action="store_true",
        help="Emit Package A evidence-boundary decision packet.",
    )
    parser.add_argument(
        "--result-store",
        action="append",
        type=Path,
        default=[],
        help="Canonical campaign result-store directory to validate.",
    )
    parser.add_argument(
        "--seed-analysis-report",
        action="append",
        type=Path,
        default=[],
        help="seed_sufficiency_analysis.v1 JSON file to validate.",
    )
    parser.add_argument(
        "--heldout-partition-manifest",
        action="append",
        type=Path,
        default=[],
        help="Held-out-family transfer partition manifest to validate.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _build_parser().parse_args(argv)
    try:
        if args.decision_packet:
            packet = build_decision_packet(
                args.manifest,
                result_stores=args.result_store,
                seed_analysis_reports=args.seed_analysis_report,
                heldout_partition_manifests=args.heldout_partition_manifest,
            )
            payload = packet.to_dict()
            if args.as_json:
                print(json.dumps(payload, indent=2))
            else:
                print(f"Package A decision packet: {payload['classification']}")
                for reason in payload["reasons"]:
                    print(f" reason: {reason}")
            return 0 if packet.classification == "diagnostic_review_ready" else 1

        report = check_readiness(args.manifest)
    except ManifestError as exc:
        if args.as_json:
            print(json.dumps({"status": "not_ready", "error": str(exc)}, indent=2))
        else:
            print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    payload = report.to_dict()
    if args.as_json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Package A readiness: {payload['status']} ({payload['package_id']})")
        for missing in payload["missing_paths"]:
            print(f" missing prerequisite: {missing}")
        for issue in payload["issues"]:
            print(f" issue: {issue}")

    return 0 if report.status == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
