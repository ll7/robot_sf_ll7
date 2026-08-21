"""Verify the future external admission boundary for a Chapter 7 v2 package.

The builder intentionally emits a blocked package and no approval receipt.
This verifier is the fail-closed promotion path for a later maintainer-owned
receipt after domain approval has been recorded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator, ValidationError

from scripts.analysis import verify_ch7_evidence_admission as admission

PACKAGE_SCHEMA = (
    Path(__file__).parents[2] / "robot_sf/benchmark/schemas/ch7-evidence-package.v2.json"
)
RECEIPT_SCHEMA = (
    Path(__file__).parents[2] / "robot_sf/benchmark/schemas/ch7-evidence-admission.v2.json"
)
DIAGNOSTIC_SCHEMA_VERSION = "ch7-evidence-admission-diagnostic.v1"
V2_FORBIDDEN_CLAIMS = (
    "matched_comparison",
    "causal_divergence",
    "counterfactual_branching",
    "trajectory_divergence",
    "universal_planner_ranking",
    "collision_metric_semantics",
)
V2_SAFE_METRICS = (
    "success_fraction",
    "near_misses_mean",
    "time_to_goal_norm_mean",
    "path_efficiency_mean",
)
V2_PORTFOLIO_REPO_PATH = "configs/analysis/ch7_worked_example_portfolio.v2.yaml"
V2_PORTFOLIO_PATH = Path(__file__).parents[2] / V2_PORTFOLIO_REPO_PATH
V2_PORTFOLIO_SHA256 = "ebf2e943b6cea7e647f71171c08e904edf19b818cd2e1853ee5409a80d74f010"


class Ch7EvidenceAdmissionV2Error(ValueError):
    """Raised when a v2 package or admission receipt fails closed."""


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Ch7EvidenceAdmissionV2Error(f"{label} is unreadable") from exc
    if not isinstance(payload, Mapping):
        raise Ch7EvidenceAdmissionV2Error(f"{label} must be an object")
    return dict(payload)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate(payload: Mapping[str, Any], schema_path: Path, label: str) -> None:
    try:
        schema = _read_object(schema_path, f"{label} schema")
        errors = sorted(Draft202012Validator(schema).iter_errors(payload), key=str)
    except (TypeError, ValidationError) as exc:
        raise Ch7EvidenceAdmissionV2Error(f"{label} schema is invalid") from exc
    if errors:
        details = "; ".join(error.message for error in errors[:3])
        raise Ch7EvidenceAdmissionV2Error(f"{label} validation failed: {details}")


def _verify_package_members_for_diagnostic(package: Path) -> tuple[str, list[str]]:
    """Verify a fresh build or a complete durable package without weakening sidecar checks."""

    sidecars = tuple(path for path in package.rglob("*.review.json") if path.is_file())
    try:
        return admission._verify_members(
            package,
            label="Chapter 7 v2 evidence package",
            require_review_sidecars=bool(sidecars),
        )
    except admission.Ch7EvidenceAdmissionError as exc:
        raise Ch7EvidenceAdmissionV2Error(f"package member verification failed: {exc}") from exc


def _load_portfolio_selection() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load the repository-controlled v2 portfolio and its exact metric boundary."""

    if _sha256_file(V2_PORTFOLIO_PATH) != V2_PORTFOLIO_SHA256:
        raise Ch7EvidenceAdmissionV2Error(
            "v2 portfolio file digest does not match the approved input"
        )
    try:
        payload = yaml.safe_load(V2_PORTFOLIO_PATH.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise Ch7EvidenceAdmissionV2Error("v2 portfolio config is unreadable") from exc
    if not isinstance(payload, Mapping):
        raise Ch7EvidenceAdmissionV2Error("v2 portfolio config must be an object")
    selection = payload.get("selection")
    metrics = payload.get("metrics")
    if not isinstance(selection, Mapping) or not isinstance(metrics, Mapping):
        raise Ch7EvidenceAdmissionV2Error("v2 portfolio lacks selection or metric contract")
    if metrics.get("included") != list(V2_SAFE_METRICS):
        raise Ch7EvidenceAdmissionV2Error("v2 portfolio safe metric contract changed")
    excluded = metrics.get("excluded")
    if not isinstance(excluded, list) or len(excluded) != 10 or len(set(excluded)) != 10:
        raise Ch7EvidenceAdmissionV2Error("v2 portfolio excluded metric contract is invalid")
    return dict(selection), [
        {
            "metric": metric,
            "issue": 7042,
            "status": "excluded",
            "reason": "collision-related metric naming remains blocked; v2 does not quote this field",
        }
        for metric in excluded
    ]


def _expected_projection(selection: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    expected: dict[str, dict[str, Any]] = {}
    for panel in ("cross_topology", "cross_mechanism", "narrow_doorway_terminal"):
        panel_selection = selection.get(panel)
        if not isinstance(panel_selection, Mapping):
            raise Ch7EvidenceAdmissionV2Error(f"v2 portfolio lacks {panel} selection")
        scenarios = panel_selection.get("scenarios")
        planners = panel_selection.get("planners")
        if (
            not isinstance(scenarios, list)
            or not isinstance(planners, list)
            or not scenarios
            or not planners
            or not all(isinstance(value, str) for value in (*scenarios, *planners))
        ):
            raise Ch7EvidenceAdmissionV2Error(f"v2 portfolio {panel} selection is invalid")
        expected[panel] = {
            "scenarios": list(scenarios),
            "planners": list(planners),
            "cell_count": len(scenarios) * len(planners),
        }
    return expected


def _verify_manifest_projection(
    manifest: Mapping[str, Any],
    expected_projection: Mapping[str, Any],
    expected_excluded: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Verify manifest-level portfolio, metric, and projection bindings."""

    portfolio = manifest.get("inputs", {}).get("portfolio_config")
    if not isinstance(portfolio, Mapping):
        raise Ch7EvidenceAdmissionV2Error("v2 manifest portfolio binding is missing")
    if (
        portfolio.get("path") != V2_PORTFOLIO_REPO_PATH
        or portfolio.get("sha256") != V2_PORTFOLIO_SHA256
    ):
        raise Ch7EvidenceAdmissionV2Error("v2 manifest portfolio binding is not approved")
    if manifest.get("projection") != expected_projection:
        raise Ch7EvidenceAdmissionV2Error("v2 manifest projection differs from the portfolio")
    metrics = manifest.get("metrics")
    if not isinstance(metrics, Mapping) or metrics.get("included") != list(V2_SAFE_METRICS):
        raise Ch7EvidenceAdmissionV2Error("v2 manifest safe metric boundary changed")
    excluded = metrics.get("excluded")
    if not isinstance(excluded, list):
        raise Ch7EvidenceAdmissionV2Error("v2 manifest excluded metric boundary is missing")
    if [
        (item.get("metric"), item.get("issue"), item.get("status"))
        for item in excluded
        if isinstance(item, Mapping)
    ] != [(item["metric"], item["issue"], item["status"]) for item in expected_excluded]:
        raise Ch7EvidenceAdmissionV2Error("v2 manifest excluded metric boundary changed")
    return excluded


def _verify_atlas_json(
    package: Path,
    expected_projection: Mapping[str, Any],
    excluded: Sequence[Mapping[str, Any]],
) -> set[tuple[str, str, str]]:
    """Verify the atlas JSON and return its expected cell count."""

    atlas = _read_object(package / "publication/reduced_atlas.json", "v2 reduced atlas")
    _validate(
        atlas,
        Path(__file__).parents[2]
        / "robot_sf/benchmark/schemas/ch7-reduced-publication-atlas.v3.json",
        "v2 reduced atlas",
    )
    if atlas.get("projections") != expected_projection:
        raise Ch7EvidenceAdmissionV2Error("v2 atlas projection differs from the portfolio")
    if atlas.get("excluded_metrics") != excluded:
        raise Ch7EvidenceAdmissionV2Error("v2 atlas excluded metric boundary differs from manifest")
    cells = atlas.get("cells")
    if not isinstance(cells, list):
        raise Ch7EvidenceAdmissionV2Error("v2 atlas cells are missing")
    expected_cells = {
        (panel, scenario, planner)
        for panel, projection in expected_projection.items()
        for scenario in projection["scenarios"]
        for planner in projection["planners"]
    }
    actual_cells = {
        (cell.get("panel"), cell.get("scenario_id"), cell.get("planner_key"))
        for cell in cells
        if isinstance(cell, Mapping)
    }
    if (
        actual_cells != expected_cells
        or len(cells) != len(expected_cells)
        or len(actual_cells) != len(cells)
    ):
        raise Ch7EvidenceAdmissionV2Error("v2 atlas cell identity differs from the portfolio")
    excluded_names = {item["metric"] for item in excluded}
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise Ch7EvidenceAdmissionV2Error("v2 atlas contains a malformed cell")
        if excluded_names.intersection(cell):
            raise Ch7EvidenceAdmissionV2Error(
                "v2 atlas contains an excluded collision or SNQI metric"
            )
        if any(metric not in cell for metric in V2_SAFE_METRICS):
            raise Ch7EvidenceAdmissionV2Error("v2 atlas cell is missing a safe metric")
    return expected_cells


def _verify_atlas_csv(
    package: Path,
    expected_cells: set[tuple[str, str, str]],
    excluded: Sequence[Mapping[str, Any]],
) -> None:
    """Verify that the CSV projection carries the same safe-metric boundary."""

    excluded_names = {item["metric"] for item in excluded}
    try:
        with (package / "publication/reduced_atlas.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            reader = csv.DictReader(stream)
            fieldnames = reader.fieldnames or []
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        raise Ch7EvidenceAdmissionV2Error("v2 reduced atlas CSV is unreadable") from exc
    if len(rows) != len(expected_cells):
        raise Ch7EvidenceAdmissionV2Error(
            "v2 reduced atlas CSV cell count differs from the portfolio"
        )
    actual_cells = {
        (row.get("panel"), row.get("scenario_id"), row.get("planner_key")) for row in rows
    }
    if actual_cells != expected_cells or len(actual_cells) != len(rows):
        raise Ch7EvidenceAdmissionV2Error(
            "v2 reduced atlas CSV cell identity differs from the portfolio"
        )
    if excluded_names.intersection(fieldnames):
        raise Ch7EvidenceAdmissionV2Error("v2 reduced atlas CSV contains an excluded metric")
    if any(metric not in fieldnames for metric in V2_SAFE_METRICS):
        raise Ch7EvidenceAdmissionV2Error("v2 reduced atlas CSV is missing a safe metric")


def _verify_atlas_projection(
    package: Path,
    expected_projection: Mapping[str, Any],
    excluded: Sequence[Mapping[str, Any]],
) -> None:
    """Verify the JSON and CSV projections against the canonical contract."""

    expected_cells = _verify_atlas_json(package, expected_projection, excluded)
    _verify_atlas_csv(package, expected_cells, excluded)


def _verify_projection_binding(
    package: Path,
    manifest: Mapping[str, Any],
    expected_projection: Mapping[str, Any],
    excluded: Sequence[Mapping[str, Any]],
) -> None:
    """Verify the independent source/projection binding sidecar."""

    binding = _read_object(package / "source/projection_binding.json", "v2 projection binding")
    if binding.get("selection") != expected_projection:
        raise Ch7EvidenceAdmissionV2Error(
            "v2 projection binding selection differs from the portfolio"
        )
    if binding.get("safe_metrics") != list(V2_SAFE_METRICS):
        raise Ch7EvidenceAdmissionV2Error("v2 projection binding safe metrics changed")
    if binding.get("excluded_metrics") != excluded:
        raise Ch7EvidenceAdmissionV2Error("v2 projection binding exclusions differ from manifest")
    source = manifest.get("source")
    source_binding = binding.get("source_package")
    if not isinstance(source, Mapping) or not isinstance(source_binding, Mapping):
        raise Ch7EvidenceAdmissionV2Error("v2 source binding is missing")
    source_pairs = {
        "sha256sums_sha256": "v1_package_sha256sums",
        "manifest_sha256": "v1_manifest_sha256",
        "member": "v1_audit_member",
        "member_sha256": "v1_audit_member_sha256",
        "terminal_member": "v1_reduced_atlas_member",
        "terminal_member_sha256": "v1_reduced_atlas_member_sha256",
    }
    if any(
        source_binding.get(binding_key) != source.get(manifest_key)
        for binding_key, manifest_key in source_pairs.items()
    ):
        raise Ch7EvidenceAdmissionV2Error("v2 projection binding source differs from manifest")


def _verify_projection_contract(package: Path, manifest: Mapping[str, Any]) -> None:
    """Independently verify the v2 projection, portfolio identity, and exclusion boundary."""

    selection, expected_excluded = _load_portfolio_selection()
    expected_projection = _expected_projection(selection)
    excluded = _verify_manifest_projection(manifest, expected_projection, expected_excluded)
    _verify_atlas_projection(package, expected_projection, excluded)
    _verify_projection_binding(package, manifest, expected_projection, excluded)


def _receipt_template(
    manifest: Mapping[str, Any], *, sums_sha: str, manifest_sha: str
) -> dict[str, Any]:
    """Build a non-validating receipt-shaped template from verified package metadata."""

    source = manifest["source"]
    inputs = manifest["inputs"]
    portfolio = inputs["portfolio_config"]
    roles = manifest["roles"]
    return {
        "template_status": "not_a_receipt",
        "schema_version": "ch7-evidence-admission.v2",
        "issue": 7087,
        "status": "template_only",
        "package": {
            "sha256sums_sha256": sums_sha,
            "manifest_sha256": manifest_sha,
        },
        "source": {
            "v1_package_sha256sums": source["v1_package_sha256sums"],
            "v1_manifest_sha256": source["v1_manifest_sha256"],
            "v1_audit_member_sha256": source["v1_audit_member_sha256"],
            "v1_reduced_atlas_member_sha256": source["v1_reduced_atlas_member_sha256"],
            "portfolio_config_sha256": portfolio["sha256"],
            "source_registry_sha256": None,
        },
        "approval": {
            "approval_id": None,
            "approval_url": None,
            "decision": None,
        },
        "scope": {
            "claim_boundary": manifest["claim_boundary"],
            "forbidden_claims": list(V2_FORBIDDEN_CLAIMS),
        },
        "roles": {
            "available": {role: {"grain": details["grain"]} for role, details in roles.items()}
        },
        "retrieval": {
            "source_package_key": None,
            "audit_member_key": None,
            "source_registry_key": None,
        },
    }


def diagnose_v2_package(package: Path) -> dict[str, Any]:
    """Check a blocked v2 package without creating or accepting an admission receipt."""

    sums_sha, _listed = _verify_package_members_for_diagnostic(package)
    package = package.resolve()
    manifest = _read_object(package / "manifest.json", "v2 package manifest")
    _validate(manifest, PACKAGE_SCHEMA, "v2 package manifest")
    _verify_projection_contract(package, manifest)
    if (
        manifest.get("status") != "blocked_pending_domain_approval"
        or manifest.get("admission_status") != "not_admitted"
        or manifest.get("source_integrity_gate") != "blocked_pending_domain_approval"
    ):
        raise Ch7EvidenceAdmissionV2Error(
            "check-only mode requires a blocked, not-admitted v2 package"
        )
    admission_block = manifest.get("admission")
    if (
        not isinstance(admission_block, Mapping)
        or admission_block.get("status") != "not_admitted"
        or admission_block.get("receipt_required") is not True
        or admission_block.get("receipt_schema") != "ch7-evidence-admission.v2"
    ):
        raise Ch7EvidenceAdmissionV2Error(
            "blocked v2 package does not declare the external admission boundary"
        )
    source = manifest["source"]
    portfolio = manifest["inputs"]["portfolio_config"]
    manifest_sha = _sha256_file(package / "manifest.json")
    blockers: list[dict[str, str]] = [
        {
            "code": "domain_approval_pending",
            "reason": "v2 domain approval is outside the package builder and verifier",
        },
        {
            "code": "external_admission_receipt_required",
            "reason": "a maintainer-owned ch7-evidence-admission.v2 receipt is required",
        },
    ]
    excluded = manifest["metrics"]["excluded"]
    exclusion_boundary = {
        "ruling_issue": 7042,
        "status": "excluded_by_frozen_ruling",
        "metrics": [item["metric"] for item in excluded],
    }
    exclusion_reasons = [
        str(item.get("reason", "")).lower()
        for item in excluded
        if isinstance(item, Mapping) and item.get("issue") == 7042
    ]
    if any(
        not any(marker in reason for marker in ("closed #7042 ruling", "frozen #7042 ruling"))
        for reason in exclusion_reasons
    ):
        blockers.append(
            {
                "code": "metric_semantics_excluded_issue_7042",
                "reason": "collision-sensitive metrics and SNQI are excluded by the closed #7042 ruling",
            }
        )
    return {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "issue": 7087,
        "status": "blocked_pending_domain_approval",
        "admission_status": "not_admitted",
        "package": {
            "sha256sums_sha256": sums_sha,
            "manifest_sha256": manifest_sha,
        },
        "source": {
            "v1_package_sha256sums": source["v1_package_sha256sums"],
            "v1_manifest_sha256": source["v1_manifest_sha256"],
            "v1_audit_member_sha256": source["v1_audit_member_sha256"],
            "v1_reduced_atlas_member_sha256": source["v1_reduced_atlas_member_sha256"],
            "portfolio_config_sha256": portfolio["sha256"],
        },
        "diagnostics": {
            "package_checksums_verified": True,
            "package_manifest_schema_verified": True,
            "admission_authorized": False,
            "empirical_outcomes_admitted": False,
            "receipt_created": False,
            "blockers": blockers,
            "exclusion_boundary": exclusion_boundary,
        },
        "receipt_template": _receipt_template(
            manifest, sums_sha=sums_sha, manifest_sha=manifest_sha
        ),
    }


def verify_v2_admission(package: Path, receipt: Path) -> dict[str, Any]:
    """Verify an admitted v2 package against its exact external receipt."""

    try:
        sums_sha, _listed = admission._verify_members(
            package, label="Chapter 7 v2 evidence package"
        )
    except admission.Ch7EvidenceAdmissionError as exc:
        raise Ch7EvidenceAdmissionV2Error(f"package member verification failed: {exc}") from exc
    package = package.resolve()
    manifest = _read_object(package / "manifest.json", "v2 package manifest")
    _validate(manifest, PACKAGE_SCHEMA, "v2 package manifest")
    _verify_projection_contract(package, manifest)
    if (
        manifest.get("status") != "admitted"
        or manifest.get("admission_status") != "admitted"
        or manifest.get("source_integrity_gate") != "passed"
        or manifest.get("admission", {}).get("status") != "admitted"
    ):
        raise Ch7EvidenceAdmissionV2Error("v2 package is not in an admitted state")
    receipt_payload = _read_object(receipt, "v2 admission receipt")
    _validate(receipt_payload, RECEIPT_SCHEMA, "v2 admission receipt")
    package_binding = receipt_payload["package"]
    if package_binding["sha256sums_sha256"] != sums_sha:
        raise Ch7EvidenceAdmissionV2Error("receipt does not bind package SHA256SUMS")
    manifest_sha = _sha256_file(package / "manifest.json")
    if package_binding["manifest_sha256"] != manifest_sha:
        raise Ch7EvidenceAdmissionV2Error("receipt does not bind package manifest")
    source_binding = receipt_payload["source"]
    expected_source = manifest["source"]
    for field in (
        "v1_package_sha256sums",
        "v1_manifest_sha256",
        "v1_audit_member_sha256",
        "v1_reduced_atlas_member_sha256",
    ):
        if source_binding[field] != expected_source[field]:
            raise Ch7EvidenceAdmissionV2Error(f"receipt source binding differs: {field}")
    if (
        source_binding["portfolio_config_sha256"]
        != manifest["inputs"]["portfolio_config"]["sha256"]
    ):
        raise Ch7EvidenceAdmissionV2Error("receipt portfolio binding differs from manifest")
    if receipt_payload["scope"]["claim_boundary"] != manifest["claim_boundary"]:
        raise Ch7EvidenceAdmissionV2Error("receipt claim boundary differs from manifest")
    expected_roles = {
        role: {"grain": details["grain"]} for role, details in manifest["roles"].items()
    }
    if receipt_payload["roles"]["available"] != expected_roles:
        raise Ch7EvidenceAdmissionV2Error("receipt role scope differs from manifest")
    return {
        "status": "admitted",
        "package_sha256sums_sha256": sums_sha,
        "manifest_sha256": manifest_sha,
        "receipt_sha256": _sha256_file(receipt.resolve()),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="verify a blocked package and print diagnostics without creating a receipt",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run v2 admission verification and return a typed CLI status code."""

    parser = _parser()
    args = parser.parse_args(argv)
    if args.check_only:
        if args.receipt is not None:
            parser.error("--receipt cannot be combined with --check-only")
        try:
            result = diagnose_v2_package(args.package)
        except (Ch7EvidenceAdmissionV2Error, OSError, ValidationError) as exc:
            print(f"ch7 v2 evidence diagnostic unavailable: {exc}")
            return 2
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    if args.receipt is None:
        parser.error("--receipt is required unless --check-only is used")
    try:
        result = verify_v2_admission(args.package, args.receipt)
    except (Ch7EvidenceAdmissionV2Error, OSError, ValidationError) as exc:
        print(f"ch7 v2 evidence admission unavailable: {exc}")
        return 2
    print(f"ch7 v2 evidence admission status: {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
