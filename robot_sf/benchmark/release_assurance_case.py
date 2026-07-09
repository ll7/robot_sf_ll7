"""Release-bundle assurance case export.

This module turns the release manifest and optional release gate specs/reports
into a machine-readable claim -> argument -> evidence document.  It is
deliberately a packaging/provenance contract: it checks referenced files and
hashes, but it does not run a benchmark campaign or promote paper-facing claims.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from robot_sf.benchmark.identity.hash_utils import sha256_file as _sha256_file
from robot_sf.benchmark.release_gates import GateSpec, load_release_gate_spec
from robot_sf.benchmark.release_protocol import BenchmarkReleaseManifest, load_release_manifest
from robot_sf.common.artifact_paths import get_repository_root

RELEASE_ASSURANCE_CASE_SCHEMA_VERSION = "benchmark_release_assurance_case.v1"


@dataclass(frozen=True)
class ReleaseAssuranceEvidenceProblem:
    """One fail-closed stale-reference problem in a release assurance case."""

    evidence_id: str
    path: str
    reason: str


def _repo_relative(path: Path, repo_root: Path) -> str:
    """Return a stable repository-relative path when possible."""

    resolved = path.resolve()
    root = repo_root.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return str(resolved)


def _evidence_leaf(
    *,
    evidence_id: str,
    description: str,
    path: Path,
    repo_root: Path,
    manifest_field: str,
    expected_sha256: str | None = None,
    kind: str = "file",
) -> dict[str, Any]:
    """Build one present evidence leaf with current and expected checksums.

    Returns:
        Evidence leaf payload.
    """

    sha256 = _sha256_file(path)
    leaf: dict[str, Any] = {
        "id": evidence_id,
        "kind": kind,
        "description": description,
        "path": _repo_relative(path, repo_root),
        "sha256": sha256,
        "manifest_field": manifest_field,
    }
    if expected_sha256:
        leaf["expected_sha256"] = expected_sha256
    return leaf


def _gate_claim_text(gate: GateSpec) -> str:
    """Return concise claim text for a release gate specification."""

    comparator = "<=" if gate.direction == "max" else ">="
    required = "required" if gate.required else "advisory"
    return (
        f"{gate.category} gate `{gate.gate_id}` requires `{gate.metric}` "
        f"{comparator} {gate.threshold:g} ({required})."
    )


def build_release_assurance_case(
    manifest: BenchmarkReleaseManifest,
    *,
    gate_spec_path: Path | None = None,
    release_gate_report_path: Path | None = None,
    generated_at_utc: str | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Build a release-bundle claim -> argument -> evidence payload.

    Args:
        manifest: Parsed release manifest.
        gate_spec_path: Optional release-gate spec to expose as claim layer.
        release_gate_report_path: Optional report proving gate evaluations.
        generated_at_utc: Stable timestamp override for reproducible examples.
        repo_root: Repository root used to format evidence paths.

    Returns:
        JSON-serializable release assurance case.
    """

    root = (repo_root or get_repository_root()).resolve()
    evidence: list[dict[str, Any]] = [
        _evidence_leaf(
            evidence_id="E_release_manifest",
            description="Release manifest that pins release identity, roster, scope, and artifacts.",
            path=manifest.path,
            repo_root=root,
            manifest_field="release_manifest",
        ),
        _evidence_leaf(
            evidence_id="E_campaign_config",
            description="Canonical campaign config referenced by the release manifest.",
            path=manifest.canonical_campaign_config_path,
            repo_root=root,
            manifest_field="canonical_campaign_config",
            expected_sha256=manifest.campaign_config_sha256,
        ),
        _evidence_leaf(
            evidence_id="E_scenario_matrix",
            description="Scenario matrix referenced by the release manifest.",
            path=manifest.scenario_matrix_path,
            repo_root=root,
            manifest_field="scenario.matrix_path",
            expected_sha256=manifest.scenario_matrix_sha256,
        ),
        _evidence_leaf(
            evidence_id="E_release_checklist",
            description="Release checklist/preflight owner referenced by the release manifest.",
            path=manifest.release_checklist_path,
            repo_root=root,
            manifest_field="release_checklist_path",
        ),
        _evidence_leaf(
            evidence_id="E_citation",
            description="Citation metadata referenced by the release manifest.",
            path=manifest.citation_path,
            repo_root=root,
            manifest_field="citation_path",
        ),
    ]
    if manifest.snqi_weights_path is not None:
        evidence.append(
            _evidence_leaf(
                evidence_id="E_snqi_weights",
                description="Social Navigation Quality Index weights referenced by the manifest.",
                path=manifest.snqi_weights_path,
                repo_root=root,
                manifest_field="metrics.snqi_weights_path",
                expected_sha256=manifest.snqi_weights_sha256,
            )
        )
    if manifest.snqi_baseline_path is not None:
        evidence.append(
            _evidence_leaf(
                evidence_id="E_snqi_baseline",
                description="Social Navigation Quality Index baseline referenced by the manifest.",
                path=manifest.snqi_baseline_path,
                repo_root=root,
                manifest_field="metrics.snqi_baseline_path",
                expected_sha256=manifest.snqi_baseline_sha256,
            )
        )

    gate_claim_ids: list[str] = []
    gate_argument_ids: list[str] = []
    gate_specs: list[GateSpec] = []
    if gate_spec_path is not None:
        gate_specs = load_release_gate_spec(gate_spec_path)
        evidence.append(
            _evidence_leaf(
                evidence_id="E_release_gate_spec",
                description="Release gate specification used as the claim layer.",
                path=gate_spec_path,
                repo_root=root,
                manifest_field="release_gate_spec",
            )
        )
    if release_gate_report_path is not None:
        evidence.append(
            _evidence_leaf(
                evidence_id="E_release_gate_report",
                description="Release gate evaluation report over benchmark rows.",
                path=release_gate_report_path,
                repo_root=root,
                manifest_field="release_gate_report",
            )
        )

    arguments: list[dict[str, Any]] = [
        {
            "id": "A_release_manifest_integrity",
            "claim_id": "C_release_bundle_manifest",
            "strategy": "manifest_integrity",
            "text": (
                "The release bundle is argued from a versioned manifest whose core files "
                "are present and, where the manifest pins a digest, checksum-matched."
            ),
            "evidence_ids": [
                "E_release_manifest",
                "E_campaign_config",
                "E_scenario_matrix",
                "E_release_checklist",
                "E_citation",
            ]
            + (["E_snqi_weights"] if manifest.snqi_weights_path is not None else [])
            + (["E_snqi_baseline"] if manifest.snqi_baseline_path is not None else []),
        },
        {
            "id": "A_roster_seed_scope",
            "claim_id": "C_roster_seed_scope_context",
            "strategy": "manifest_context",
            "text": (
                "Planner roster, seed policy, scenario scope, and kinematic scope are "
                "taken from the release manifest rather than reconstructed from code."
            ),
            "evidence_ids": ["E_release_manifest", "E_campaign_config", "E_scenario_matrix"],
        },
    ]

    if gate_specs:
        for gate in gate_specs:
            claim_id = f"C_gate_{gate.gate_id}"
            argument_id = f"A_gate_{gate.gate_id}"
            gate_claim_ids.append(claim_id)
            gate_argument_ids.append(argument_id)
            evidence_ids = ["E_release_gate_spec"]
            if release_gate_report_path is not None:
                evidence_ids.append("E_release_gate_report")
            arguments.append(
                {
                    "id": argument_id,
                    "claim_id": claim_id,
                    "strategy": "threshold_gate_specification",
                    "text": (
                        "Gate claim is derived directly from the gate spec threshold, "
                        "metric, category, requirement flag, and scope."
                    ),
                    "evidence_ids": evidence_ids,
                    "gate": {
                        "gate_id": gate.gate_id,
                        "metric": gate.metric,
                        "threshold": gate.threshold,
                        "direction": gate.direction,
                        "category": gate.category,
                        "required": gate.required,
                        "scope": dict(gate.scope or {}),
                    },
                }
            )

    claims = [
        {
            "id": "C_release_bundle_manifest",
            "text": (
                f"Release `{manifest.release_id}` is defined by a versioned release "
                "manifest with checksum-backed core inputs."
            ),
            "argument_ids": ["A_release_manifest_integrity"],
        },
        {
            "id": "C_roster_seed_scope_context",
            "text": (
                "Planner roster, seed policy, scenario matrix, and kinematic scope are "
                "explicit release context claims."
            ),
            "argument_ids": ["A_roster_seed_scope"],
        },
    ]
    claims.extend(
        {
            "id": f"C_gate_{gate.gate_id}",
            "text": _gate_claim_text(gate),
            "argument_ids": [f"A_gate_{gate.gate_id}"],
        }
        for gate in gate_specs
    )

    return {
        "schema_version": RELEASE_ASSURANCE_CASE_SCHEMA_VERSION,
        "generated_at_utc": generated_at_utc
        or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "release": {
            "release_id": manifest.release_id,
            "release_tag": manifest.release_tag,
            "benchmark_protocol_version": manifest.benchmark_protocol_version,
            "maturity": manifest.maturity,
        },
        "claim_boundary": (
            "Machine-readable assurance export only. This document records release "
            "manifest, gate-spec, and evidence-reference integrity; it does not run "
            "a benchmark campaign or promote paper/dissertation claims."
        ),
        "release_context": {
            "planner_keys": list(manifest.planner_keys),
            "planner_groups": manifest.planner_groups,
            "seed_policy": manifest.seed_policy,
            "kinematics_matrix": list(manifest.expected_kinematics_matrix),
            "holonomic_command_mode": manifest.expected_holonomic_command_mode,
            "required_artifact_paths": list(manifest.required_artifact_paths),
        },
        "claims": claims,
        "arguments": arguments,
        "evidence": evidence,
        "root_claim_ids": [
            "C_release_bundle_manifest",
            "C_roster_seed_scope_context",
            *gate_claim_ids,
        ],
        "root_argument_ids": [
            "A_release_manifest_integrity",
            "A_roster_seed_scope",
            *gate_argument_ids,
        ],
    }


def validate_release_assurance_case_schema(payload: dict[str, Any]) -> None:
    """Validate a release assurance case payload against its JSON Schema."""

    schema_path = (
        get_repository_root() / "robot_sf/benchmark/schemas/release_assurance_case.schema.v1.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(payload)
    _validate_release_assurance_case_links(payload)


def _ids_by_section(payload: dict[str, Any], section: str) -> set[str]:
    """Return unique IDs for one list section, raising on duplicates."""

    seen: set[str] = set()
    duplicates: set[str] = set()
    for item in payload.get(section, []):
        item_id = str(item.get("id", ""))
        if item_id in seen:
            duplicates.add(item_id)
        seen.add(item_id)
    if duplicates:
        raise ValueError(f"{section} contains duplicate ids: {sorted(duplicates)}")
    return seen


def _validate_release_assurance_case_links(payload: dict[str, Any]) -> None:  # noqa: C901
    """Validate claim -> argument -> evidence ID references."""

    claim_ids = _ids_by_section(payload, "claims")
    argument_ids = _ids_by_section(payload, "arguments")
    evidence_ids = _ids_by_section(payload, "evidence")
    for claim in payload.get("claims", []):
        for argument_id in claim.get("argument_ids", []):
            if argument_id not in argument_ids:
                raise ValueError(f"claim {claim['id']} references missing argument {argument_id}")
    for argument in payload.get("arguments", []):
        if argument["claim_id"] not in claim_ids:
            raise ValueError(
                f"argument {argument['id']} references missing claim {argument['claim_id']}"
            )
        for evidence_id in argument.get("evidence_ids", []):
            if evidence_id not in evidence_ids:
                raise ValueError(
                    f"argument {argument['id']} references missing evidence {evidence_id}"
                )
    for claim_id in payload.get("root_claim_ids", []):
        if claim_id not in claim_ids:
            raise ValueError(f"root_claim_ids references missing claim {claim_id}")
    for argument_id in payload.get("root_argument_ids", []):
        if argument_id not in argument_ids:
            raise ValueError(f"root_argument_ids references missing argument {argument_id}")


def validate_release_assurance_case_references(
    payload: dict[str, Any],
    *,
    repo_root: Path | None = None,
) -> list[ReleaseAssuranceEvidenceProblem]:
    """Return stale or missing evidence-reference problems for ``payload``."""

    root = (repo_root or get_repository_root()).resolve()
    problems: list[ReleaseAssuranceEvidenceProblem] = []
    for leaf in payload.get("evidence", []):
        if not isinstance(leaf, dict):
            continue
        evidence_id = str(leaf.get("id", ""))
        rel_path = str(leaf.get("path", ""))
        expected_sha = str(leaf.get("sha256", ""))
        if not rel_path:
            problems.append(
                ReleaseAssuranceEvidenceProblem(evidence_id, rel_path, "missing evidence path")
            )
            continue
        path = Path(rel_path)
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            problems.append(
                ReleaseAssuranceEvidenceProblem(evidence_id, rel_path, "evidence file not found")
            )
            continue
        actual_sha = _sha256_file(path)
        if actual_sha != expected_sha:
            problems.append(
                ReleaseAssuranceEvidenceProblem(
                    evidence_id,
                    rel_path,
                    f"sha256 mismatch: recorded {expected_sha}, actual {actual_sha}",
                )
            )
        expected_manifest_sha = leaf.get("expected_sha256")
        if expected_manifest_sha and expected_manifest_sha != expected_sha:
            problems.append(
                ReleaseAssuranceEvidenceProblem(
                    evidence_id,
                    rel_path,
                    (
                        "manifest expected_sha256 does not match evidence sha256: "
                        f"{expected_manifest_sha} != {expected_sha}"
                    ),
                )
            )
    return problems


def write_release_assurance_case(payload: dict[str, Any], path: Path) -> Path:
    """Write ``payload`` as stable, pretty JSON.

    Returns:
        Path that was written.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_release_assurance_case(path: str | Path) -> dict[str, Any]:
    """Load a release assurance case JSON document.

    Returns:
        Parsed release assurance case object.
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Release assurance case must be a JSON object: {path}")
    return payload


def build_release_assurance_case_from_paths(
    *,
    manifest_path: Path,
    gate_spec_path: Path | None = None,
    release_gate_report_path: Path | None = None,
    generated_at_utc: str | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Load path inputs and build a release assurance case payload.

    Returns:
        Release assurance case payload.
    """

    manifest = load_release_manifest(manifest_path)
    return build_release_assurance_case(
        manifest,
        gate_spec_path=gate_spec_path,
        release_gate_report_path=release_gate_report_path,
        generated_at_utc=generated_at_utc,
        repo_root=repo_root,
    )
