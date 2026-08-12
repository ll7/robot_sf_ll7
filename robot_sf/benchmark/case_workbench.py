"""Deterministic case discovery and explanation workbench.

This module owns the machine proposal ledger.  It intentionally stops at a
digest-bound proposal/admission package: author approval is an explicit overlay,
not an implicit side effect of ranking.
"""

# The workbench is intentionally a compact orchestration boundary.  Its input
# adapters keep optional analytics dependencies lazy, and the package emits
# reviewable return payloads rather than public library objects.
# ruff: noqa: DOC201, PLC0415

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.analysis_trace import (
    canonical_json,
    trace_artifact_sha256,
    trace_coverage,
)
from robot_sf.benchmark.parquet_export import (
    ParquetDependencyError,
    derive_episode_metrics,
    export_campaign_result_store_v2,
    is_comparison_compatible,
)
from robot_sf.benchmark.termination_reason import canonical_outcome_flags
from robot_sf.common.optional_import import try_import

SCHEMA_VERSION = "case-workbench.v1"
METRIC_PROFILE_VERSION = "case-workbench-metrics.v1"
ADMISSION_SCHEMA_VERSION = "case-admission-overlay.v1"
SOURCE_GATE_SCHEMA_VERSION = "case-source-integrity-gate.v1"
SOURCE_GATE_REGISTRY_SCHEMA_VERSION = "case-source-integrity-registry.v1"
TRUSTED_SOURCE_GATE_REGISTRY = (
    Path(__file__).resolve().parents[2] / "configs/analysis/source_gate_registry.v1.json"
)
INELIGIBLE_STATUSES = {
    "fallback",
    "degraded",
    "failed",
    "failure",
    "error",
    "truncated",
    "terminated",
    "unavailable",
    "partial",
    "partial_failure",
    "diagnostic_only",
    "diagnostic_stub",
    "adapter",
}


def load_workbench_config(path: str | Path) -> dict[str, Any]:  # noqa: C901, PLR0912
    """Load and validate the compact workbench configuration."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"configuration must declare {SCHEMA_VERSION}")
    portfolio = payload.get("portfolio")
    if not isinstance(portfolio, dict) or not isinstance(portfolio.get("roles"), list):
        raise ValueError("portfolio.roles must be a list")
    if portfolio.get("require_trace_coverage", "complete") != "complete":
        raise ValueError("portfolio.require_trace_coverage must be 'complete'")
    if portfolio.get("allow_shared_prefix_false") is not True:
        raise ValueError("portfolio.allow_shared_prefix_false must be true for v1")
    interestingness = payload.get("interestingness", {})
    if interestingness and not isinstance(interestingness, dict):
        raise ValueError("interestingness must be a mapping")
    weights = interestingness.get("weights", {}) if isinstance(interestingness, dict) else {}
    if weights and not isinstance(weights, dict):
        raise ValueError("interestingness.weights must be a mapping")
    for name, value in weights.items():
        if not _finite_number(value):
            raise ValueError(f"interestingness.weights.{name} must be finite")
    comparison = payload.get("comparison", {})
    if comparison and not isinstance(comparison, dict):
        raise ValueError("comparison must be a mapping")
    for name in ("require_matching_initial_state", "require_matching_config_digest"):
        if name in comparison and not isinstance(comparison[name], bool):
            raise ValueError(f"comparison.{name} must be boolean")
        if comparison.get(name) is not True:
            raise ValueError(f"comparison.{name} must be true for v1")
    if comparison.get("shared_prefix", False) is not False:
        raise ValueError("comparison.shared_prefix must be false for case-workbench.v1")
    publication = payload.get("publication", {})
    if publication and not isinstance(publication, dict):
        raise ValueError("publication must be a mapping")
    for name in ("include_difference_curve", "include_normalized_duration"):
        if publication.get(name, False) is not False:
            raise ValueError(f"publication.{name} is forbidden in case-workbench.v1")
    return payload


def _load_source_gate_receipt(  # noqa: C901
    source: Path,
    receipt_path: str | Path | None,
) -> dict[str, Any]:
    """Resolve the exact-source gate without promoting a guessed package."""

    expected_source_sha = _sha256_file(source) if source.is_file() else _sha256_directory(source)
    blocked = {
        "schema_version": SOURCE_GATE_SCHEMA_VERSION,
        "status": "blocked_pending_exact_source_restore",
        "reason": "source_gate_receipt_missing",
        "source_sha256": expected_source_sha,
        "robot_sf_issues": [
            "https://github.com/ll7/robot_sf_ll7/issues/6792",
            "https://github.com/ll7/robot_sf_ll7/issues/6814",
        ],
        "dissertation_issue": "https://github.com/ll7/diss/issues/698",
    }
    registry_path = TRUSTED_SOURCE_GATE_REGISTRY
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {**blocked, "reason": "source_gate_registry_unavailable"}
    if (
        not isinstance(registry, Mapping)
        or registry.get("schema_version") != SOURCE_GATE_REGISTRY_SCHEMA_VERSION
    ):
        return {**blocked, "reason": "source_gate_registry_invalid"}
    registry_digest = _sha256_file(registry_path)
    approved_sources = registry.get("approved_sources")
    if not isinstance(approved_sources, list):
        return {
            **blocked,
            "reason": "source_gate_registry_invalid",
            "registry_sha256": registry_digest,
        }
    if receipt_path is None:
        return {
            **blocked,
            "reason": "source_gate_receipt_missing",
            "registry_sha256": registry_digest,
        }
    path = Path(receipt_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            **blocked,
            "reason": "source_gate_receipt_unreadable",
            "registry_sha256": registry_digest,
        }
    if not isinstance(payload, Mapping):
        return {
            **blocked,
            "reason": "source_gate_receipt_invalid",
            "registry_sha256": registry_digest,
        }
    if payload.get("schema_version") != SOURCE_GATE_SCHEMA_VERSION:
        return {
            **blocked,
            "reason": "source_gate_schema_mismatch",
            "receipt_sha256": _sha256_file(path),
            "registry_sha256": registry_digest,
        }
    status = str(payload.get("status") or "").lower()
    supplied_sha = str(payload.get("source_sha256") or payload.get("digest") or "")
    if status != "passed":
        return {
            **blocked,
            "reason": f"source_gate_status:{status or 'missing'}",
            "receipt_sha256": _sha256_file(path),
            "registry_sha256": registry_digest,
        }
    if not re.fullmatch(r"[0-9a-f]{64}", supplied_sha) or supplied_sha != expected_source_sha:
        return {
            **blocked,
            "reason": "source_gate_digest_mismatch",
            "receipt_sha256": _sha256_file(path),
            "supplied_source_sha256": supplied_sha or None,
        }
    matching_entry = next(
        (
            entry
            for entry in approved_sources
            if isinstance(entry, Mapping)
            and entry.get("source_sha256") == expected_source_sha
            and entry.get("approval_id") == payload.get("approval_id")
        ),
        None,
    )
    if not isinstance(matching_entry, Mapping):
        return {
            **blocked,
            "reason": "source_gate_source_not_approved",
            "receipt_sha256": _sha256_file(path),
            "registry_sha256": registry_digest,
            "supplied_source_sha256": supplied_sha,
        }
    return {
        "schema_version": SOURCE_GATE_SCHEMA_VERSION,
        "status": "passed",
        "source_sha256": expected_source_sha,
        "receipt_sha256": _sha256_file(path),
        "registry_sha256": registry_digest,
        "approval_id": str(matching_entry["approval_id"]),
        "robot_sf_issues": [
            "https://github.com/ll7/robot_sf_ll7/issues/6792",
            "https://github.com/ll7/robot_sf_ll7/issues/6814",
        ],
        "dissertation_issue": "https://github.com/ll7/diss/issues/698",
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    """Read one JSON object at a package mutation boundary."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"JSON package input is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON package input must be an object: {path}")
    return payload


def _source_gate_is_trusted(gate: Mapping[str, Any]) -> bool:
    """Verify a stored gate against the repository-controlled approval registry."""

    if gate.get("status") != "passed":
        return False
    registry_path = TRUSTED_SOURCE_GATE_REGISTRY
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if (
        not isinstance(registry, Mapping)
        or registry.get("schema_version") != SOURCE_GATE_REGISTRY_SCHEMA_VERSION
    ):
        return False
    if gate.get("registry_sha256") != _sha256_file(registry_path):
        return False
    approved_sources = registry.get("approved_sources")
    if not isinstance(approved_sources, list):
        return False
    return any(
        isinstance(entry, Mapping)
        and entry.get("approval_id") == gate.get("approval_id")
        and entry.get("source_sha256") == gate.get("source_sha256")
        for entry in approved_sources
    )


def _portable_source_gate(gate: Mapping[str, Any]) -> dict[str, Any]:
    """Return only digest-bound gate metadata suitable for package storage."""

    # The gate deliberately contains no receipt path or raw receipt payload.  Besides
    # making packages reproducible across machines, this prevents an author-supplied
    # path or receipt contents from being copied into a review artifact.
    return dict(gate)


def analyze_cases(  # noqa: C901
    *,
    config_path: str | Path,
    result_store: str | Path,
    output: str | Path,
    check_determinism: bool = False,
    source_gate_receipt: str | Path | None = None,
) -> dict[str, Any]:
    """Build a deterministic proposal/admission package from a result store."""

    config = load_workbench_config(config_path)
    input_path = Path(result_store)
    output_path = Path(output)
    if input_path.is_dir():
        resolved_input = input_path.resolve()
        resolved_output = output_path.resolve()
        if resolved_output == resolved_input or resolved_input in resolved_output.parents:
            raise ValueError("output package must not be the result-store directory or its child")
    records = _load_records(input_path)
    source_gate = _portable_source_gate(_load_source_gate_receipt(input_path, source_gate_receipt))
    interestingness_weights = config.get("interestingness", {}).get("weights", {})
    candidates = [
        _candidate(record, interestingness_weights=interestingness_weights) for record in records
    ]
    proposal = _build_proposal(candidates, config=config)
    if check_determinism:
        repeat = _build_proposal(candidates, config=config)
        if canonical_json(proposal) != canonical_json(repeat):
            raise RuntimeError("case-workbench selection is not deterministic")

    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"output package must be empty: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    if input_path.is_file():
        try:
            export_campaign_result_store_v2(
                input_path,
                output_path / "campaign-result-store.v2",
                study_id="case-workbench",
                command="analyze-cases --result-store <result-store>",
                overwrite=True,
            )
        except (ImportError, ParquetDependencyError) as exc:
            # The proposal still remains useful in a lean environment; the
            # package states that the normalized store could not be materialized.
            (output_path / "campaign-result-store.v2.unavailable").write_text(
                f"campaign-result-store.v2 unavailable: {exc}\n",
                encoding="utf-8",
            )
    elif (input_path / "episodes.parquet").is_file():
        # A v2 directory is already the canonical normalized store.  Copy it
        # byte-for-byte so the package remains bound to the verified source
        # manifest/checksum set instead of silently re-exporting a sidecar
        # JSONL with different provenance.
        shutil.copytree(input_path, output_path / "campaign-result-store.v2")
    elif (input_path / "episodes.jsonl").is_file() or (input_path / "records.jsonl").is_file():
        source_jsonl = next(
            candidate
            for candidate in (input_path / "episodes.jsonl", input_path / "records.jsonl")
            if candidate.is_file()
        )
        try:
            export_campaign_result_store_v2(
                source_jsonl,
                output_path / "campaign-result-store.v2",
                study_id="case-workbench",
                command="analyze-cases --result-store <result-store>",
                overwrite=True,
            )
        except (ImportError, ParquetDependencyError) as exc:
            (output_path / "campaign-result-store.v2.unavailable").write_text(
                f"campaign-result-store.v2 unavailable: {exc}\n", encoding="utf-8"
            )
    elif input_path.is_dir():
        destination = output_path / "campaign-result-store.v2"
        shutil.copytree(input_path, destination)
    (output_path / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (output_path / "proposal.json").write_text(
        json.dumps(proposal, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    overlay = {
        "schema_version": ADMISSION_SCHEMA_VERSION,
        "proposal_sha256": _sha256_json(proposal),
        "status": str(config.get("admission", {}).get("status", "proposed")),
        "decisions": [],
    }
    (output_path / "admission_overlay.json").write_text(
        json.dumps(overlay, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_case_files(output_path, proposal)
    _write_viewer_blueprint(output_path, proposal)
    _write_audit_dossier(output_path, proposal)
    (output_path / "source_integrity_gate.json").write_text(
        json.dumps(source_gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_publication_preview(output_path, proposal, source_gate=source_gate)
    (output_path / "review_memo.md").write_text(_review_memo(proposal), encoding="utf-8")
    manifest = _manifest(output_path, proposal, input_path, config_path, source_gate=source_gate)
    (output_path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_path / "SHA256SUMS").write_text(_checksums(output_path), encoding="utf-8")
    return proposal


def apply_admission_overlay(  # noqa: C901, PLR0912, PLR0915
    proposal: Mapping[str, Any], overlay: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply a digest-bound author overlay while retaining machine recommendations."""

    if overlay.get("schema_version") != ADMISSION_SCHEMA_VERSION:
        raise ValueError(f"overlay must declare {ADMISSION_SCHEMA_VERSION}")
    expected_digest = _sha256_json(proposal)
    if overlay.get("proposal_sha256") != expected_digest:
        raise ValueError("admission overlay proposal_sha256 does not match the proposal")
    overlay_status = str(overlay.get("status") or "proposed").lower()
    if overlay_status not in {"proposed", "admitted", "overridden", "rejected"}:
        raise ValueError(f"unsupported admission overlay status: {overlay_status}")
    decisions = overlay.get("decisions", [])
    if not isinstance(decisions, list):
        raise ValueError("admission overlay decisions must be a list")
    if any(not isinstance(item, Mapping) for item in decisions):
        raise ValueError("admission overlay decisions must be objects")
    machine_portfolio = [
        dict(case) for case in proposal.get("portfolio", []) if isinstance(case, Mapping)
    ]
    machine_portfolio = json.loads(canonical_json(machine_portfolio))
    working_portfolio = json.loads(canonical_json(machine_portfolio))
    by_id = {str(case.get("case_id")): case for case in working_portfolio}
    final_portfolio = working_portfolio
    admission_records: list[dict[str, Any]] = []
    replacement_artifacts: dict[str, str] = {}
    normalized_decisions = sorted(
        decisions,
        key=lambda item: (str(item.get("case_id")), str(item.get("decision"))),
    )
    decision_ids = [str(item.get("case_id") or "") for item in normalized_decisions]
    if len(decision_ids) != len(set(decision_ids)):
        raise ValueError("admission overlay contains duplicate case decisions")
    if overlay_status == "admitted":
        proposed_ids = set(by_id)
        if set(decision_ids) != proposed_ids:
            missing = sorted(proposed_ids - set(decision_ids))
            extra = sorted(set(decision_ids) - proposed_ids)
            raise ValueError(
                "an admitted overlay must decide every proposed case "
                f"(missing={missing}, unknown={extra})"
            )
    for decision in normalized_decisions:
        case_id = str(decision.get("case_id") or "")
        action = str(decision.get("decision") or "").lower()
        rationale = str(decision.get("rationale") or "").strip()
        if case_id not in by_id:
            raise ValueError(f"admission overlay references unknown proposed case: {case_id}")
        if action not in {"approve", "reject", "replace"}:
            raise ValueError(f"unsupported admission decision for {case_id}: {action}")
        if not rationale:
            raise ValueError(f"admission rationale is required for {case_id}")
        if action == "replace":
            replacement = decision.get("replacement")
            if not isinstance(replacement, Mapping) or not replacement.get("case_id"):
                raise ValueError(f"replacement case is required for {case_id}")
            replacement_case = _validate_replacement_case(
                replacement,
                case_id=case_id,
                artifact_inventory=proposal.get("artifact_inventory"),
                portfolio=proposal.get("portfolio"),
            )
            replacement_case["author_status"] = "replacement"
            replacement_case["machine_recommendation"] = case_id
            replacement_case["author_rationale"] = rationale
            final_portfolio = [
                case for case in final_portfolio if str(case.get("case_id")) != case_id
            ]
            replacement_id = str(replacement_case["case_id"])
            if any(str(case.get("case_id")) == replacement_id for case in final_portfolio):
                raise ValueError(
                    f"replacement case id is already in the portfolio: {replacement_id}"
                )
            final_portfolio.append(replacement_case)
            replacement_artifacts[replacement_id] = str(
                replacement_case["provenance"]["artifact_sha256"]
            )
        else:
            selected = by_id[case_id]
            selected["author_status"] = "approved" if action == "approve" else "rejected"
            selected["author_rationale"] = rationale
            if action == "reject":
                final_portfolio = [
                    case for case in final_portfolio if str(case.get("case_id")) != case_id
                ]
        admission_records.append(
            {
                "case_id": case_id,
                "decision": action,
                "rationale": rationale,
            }
        )
    result = json.loads(canonical_json(proposal))
    result["artifact_inventory"] = {
        **(
            dict(proposal.get("artifact_inventory"))
            if isinstance(proposal.get("artifact_inventory"), Mapping)
            else {}
        ),
        **replacement_artifacts,
    }
    result["machine_portfolio"] = machine_portfolio
    result["portfolio"] = final_portfolio
    result["author_admission"] = {
        "schema_version": ADMISSION_SCHEMA_VERSION,
        "status": overlay_status,
        "overlay_sha256": _sha256_json(overlay),
        "machine_proposal_sha256": expected_digest,
        "decisions": admission_records,
    }
    return result


def admit_package(package: str | Path, overlay_path: str | Path) -> dict[str, Any]:
    """Apply an author overlay and refresh the package's digest receipts."""

    package_path = Path(package)
    proposal_path = package_path / "proposal.json"
    manifest_path = package_path / "manifest.json"
    if not proposal_path.is_file() or not manifest_path.is_file():
        raise ValueError("case-workbench package is missing proposal.json or manifest.json")
    from robot_sf.benchmark.case_publication_figure import _verify_package_integrity

    _verify_package_integrity(package_path)
    proposal = _read_json_object(proposal_path)
    overlay = _read_json_object(Path(overlay_path))
    manifest = _read_json_object(manifest_path)
    machine_digest = _sha256_json(proposal)
    if manifest.get("proposal_sha256") != machine_digest:
        raise ValueError("package manifest proposal digest does not match proposal.json")
    if overlay.get("proposal_sha256") != machine_digest:
        raise ValueError("admission overlay is not bound to the machine proposal")
    gate = manifest.get("source_integrity_gate")
    source_manifest = manifest.get("source")
    if (
        not isinstance(gate, Mapping)
        or not _source_gate_is_trusted(gate)
        or not isinstance(source_manifest, Mapping)
        or gate.get("source_sha256") != source_manifest.get("sha256")
    ):
        raise ValueError("author admission is blocked by the source-integrity gate")
    result = apply_admission_overlay(proposal, overlay)
    if str(overlay.get("status") or "").lower() == "admitted" and not overlay.get("decisions"):
        raise ValueError("an admitted package requires at least one author decision")
    result["author_admission"]["machine_proposal_sha256"] = machine_digest
    proposal_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (package_path / "admission_overlay.json").write_text(
        json.dumps(overlay, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    publication = package_path / "publication"
    if publication.is_dir():
        for stale in publication.iterdir():
            if stale.is_file():
                stale.unlink()
    _clear_case_files(package_path)
    _write_case_files(package_path, result)
    _write_viewer_blueprint(package_path, result)
    _write_audit_dossier(package_path, result)
    (package_path / "review_memo.md").write_text(_review_memo(result), encoding="utf-8")
    manifest["machine_proposal_sha256"] = machine_digest
    manifest["proposal_sha256"] = _sha256_json(result)
    manifest["evidence_status"] = (
        "admitted" if str(overlay.get("status") or "").lower() == "admitted" else "reviewed"
    )
    manifest["files"] = sorted(
        str(path.relative_to(package_path))
        for path in package_path.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (package_path / "SHA256SUMS").write_text(_checksums(package_path), encoding="utf-8")
    return result


def _validate_replacement_case(  # noqa: C901, PLR0912
    replacement: Mapping[str, Any],
    *,
    case_id: str,
    artifact_inventory: Any,
    portfolio: Any,
) -> dict[str, Any]:
    """Require an author replacement to carry a complete, self-hashed trace."""

    replacement_case = dict(replacement)
    replacement_id = str(replacement_case.get("case_id") or "").strip()
    scenario_id = str(replacement_case.get("scenario_id") or "").strip()
    planner = str(replacement_case.get("planner") or "").strip()
    if not replacement_id or not scenario_id or not planner:
        raise ValueError(f"replacement case identity is incomplete for {case_id}")
    if replacement_case.get("seed") is None:
        raise ValueError(f"replacement case seed is required for {case_id}")
    provenance = replacement_case.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"replacement case provenance is required for {case_id}")
    artifact_sha = str(provenance.get("artifact_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", artifact_sha):
        raise ValueError(f"replacement case requires a full SHA-256 artifact hash for {case_id}")
    if not isinstance(artifact_inventory, Mapping):
        raise ValueError("replacement cases require a source artifact inventory")
    if artifact_inventory.get(replacement_id) != artifact_sha:
        raise ValueError(f"replacement case is not a registered source artifact: {replacement_id}")
    trace = replacement_case.get("trace")
    if not isinstance(trace, Mapping):
        metadata = replacement_case.get("algorithm_metadata")
        trace = metadata.get("analysis_trace") if isinstance(metadata, Mapping) else None
    if not isinstance(trace, Mapping):
        raise ValueError(f"replacement case analysis trace is required for {case_id}")
    trace = json.loads(canonical_json(trace))
    embedded_sha = trace.get("artifact_sha256")
    if embedded_sha != artifact_sha or trace_artifact_sha256(trace) != artifact_sha:
        raise ValueError(f"replacement case artifact hash is not bound to its trace for {case_id}")
    coverage_record = {
        "scenario_id": scenario_id,
        "algo": planner,
        "provenance": dict(provenance),
        "algorithm_metadata": {"analysis_trace": trace},
    }
    coverage = trace_coverage(coverage_record)
    if coverage.get("status") != "complete":
        raise ValueError(f"replacement case trace coverage is unavailable for {case_id}")
    supplied_coverage = replacement_case.get("coverage")
    if isinstance(supplied_coverage, Mapping) and supplied_coverage.get("status") == "complete":
        if supplied_coverage.get("schema_version") != coverage.get("schema_version"):
            raise ValueError(f"replacement case coverage receipt conflicts for {case_id}")
    replacement_case["scenario_id"] = scenario_id
    replacement_case["planner"] = planner
    replacement_case["trace"] = trace
    replacement_case["coverage"] = coverage
    replacement_case["provenance"] = dict(provenance)
    pair_ids = replacement_case.get("comparison_pair_ids")
    if pair_ids:
        if not isinstance(pair_ids, list) or len(pair_ids) != 2:
            raise ValueError(
                f"replacement comparison pair must contain exactly two cases for {case_id}"
            )
        pair_ids = [str(value) for value in pair_ids]
        if replacement_id not in pair_ids or len(set(pair_ids)) != 2:
            raise ValueError(f"replacement comparison pair is not bound to {replacement_id}")
        if not isinstance(portfolio, list):
            raise ValueError("replacement comparison pair requires the machine portfolio")
        counterpart = next(
            (
                candidate
                for candidate in portfolio
                if isinstance(candidate, Mapping)
                and str(candidate.get("case_id") or "") in set(pair_ids)
                and str(candidate.get("case_id") or "") != replacement_id
            ),
            None,
        )
        if not isinstance(counterpart, Mapping):
            raise ValueError(
                "replacement comparison pair counterpart is not in the machine portfolio"
            )
        replacement_raw = {
            "episode_id": replacement_id,
            "scenario_id": scenario_id,
            "algo": planner,
            "seed": replacement_case.get("seed"),
            "provenance": dict(provenance),
            "algorithm_metadata": {"analysis_trace": trace},
        }
        counterpart_raw = {
            "episode_id": counterpart.get("case_id"),
            "scenario_id": counterpart.get("scenario_id"),
            "algo": counterpart.get("planner"),
            "seed": counterpart.get("seed"),
            "provenance": counterpart.get("provenance", {}),
            "algorithm_metadata": {"analysis_trace": counterpart.get("trace")},
        }
        if not is_comparison_compatible(replacement_raw, counterpart_raw):
            raise ValueError("replacement comparison pair is not physically compatible")
        replacement_case["comparison_compatibility"] = {
            "status": "compatible",
            "pair_ids": sorted(pair_ids),
            "shared_prefix": False,
        }
    return replacement_case


def _load_records(path: Path) -> list[dict[str, Any]]:  # noqa: C901
    """Load source records from JSONL or a v2 Parquet result store."""

    if path.is_file():
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number} must contain a JSON object")
                rows.append(row)
        _validate_episode_ids(rows, path)
        return rows
    if not path.is_dir():
        raise FileNotFoundError(f"result store does not exist: {path}")
    episodes_path = path / "episodes.parquet"
    if episodes_path.is_file():
        rows = _load_v2_records(path, episodes_path)
        _validate_episode_ids(rows, path)
        return rows
    jsonl_candidates = [path / "episodes.jsonl", path / "records.jsonl"]
    for candidate in jsonl_candidates:
        if candidate.is_file():
            return _load_records(candidate)
    if not episodes_path.is_file():
        raise ValueError(f"result store has no episodes.jsonl or episodes.parquet: {path}")
    rows = _load_v2_records(path, episodes_path)
    _validate_episode_ids(rows, path)
    return rows


def _validate_episode_ids(rows: list[dict[str, Any]], source: Path) -> None:
    """Reject blank or duplicate episode identities before selection."""

    seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        episode_id = str(row.get("episode_id") or "").strip()
        if not episode_id:
            raise ValueError(f"{source}:row {index} has no episode_id")
        if episode_id in seen:
            raise ValueError(f"{source}:duplicate episode_id {episode_id}")
        seen.add(episode_id)


def _load_v2_records(  # noqa: C901, PLR0912, PLR0915
    store: Path, episodes_path: Path
) -> list[dict[str, Any]]:
    """Rehydrate v2 episode, step, actor, event, and feature tables."""

    episode_rows = _read_parquet_rows(episodes_path)
    integrity_errors = _v2_integrity_errors(store)
    required_tables = ("steps", "actors", "events", "features", "cells", "comparisons")
    missing_tables = {name for name in required_tables if not (store / f"{name}.parquet").is_file()}
    step_rows = _read_parquet_rows(store / "steps.parquet") if "steps" not in missing_tables else []
    actor_rows = (
        _read_parquet_rows(store / "actors.parquet") if "actors" not in missing_tables else []
    )
    event_rows = (
        _read_parquet_rows(store / "events.parquet") if "events" not in missing_tables else []
    )
    feature_rows = (
        _read_parquet_rows(store / "features.parquet") if "features" not in missing_tables else []
    )
    cell_rows = _read_parquet_rows(store / "cells.parquet") if "cells" not in missing_tables else []
    comparison_rows = (
        _read_parquet_rows(store / "comparisons.parquet")
        if "comparisons" not in missing_tables
        else []
    )
    integrity_errors.extend(
        _v2_row_count_errors(
            store,
            {
                "episodes": len(episode_rows),
                "steps": len(step_rows),
                "actors": len(actor_rows),
                "events": len(event_rows),
                "features": len(feature_rows),
                "cells": len(cell_rows),
                "comparisons": len(comparison_rows),
            },
        )
    )
    by_episode_steps: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_episode_actors: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    by_episode_events: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_episode_features: dict[str, dict[str, float]] = defaultdict(dict)
    by_cell: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    by_episode_comparisons: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in step_rows:
        by_episode_steps[str(row.get("episode_id"))].append(row)
    for row in actor_rows:
        by_episode_actors[(str(row.get("episode_id")), int(row.get("step") or 0))].append(row)
    for row in event_rows:
        by_episode_events[str(row.get("episode_id"))].append(row)
    for row in feature_rows:
        value = row.get("value_number")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            by_episode_features[str(row.get("episode_id"))][str(row.get("feature_name"))] = float(
                value
            )
    for row in cell_rows:
        key = (
            str(row.get("planner") or ""),
            str(row.get("scenario_id") or ""),
            str(row.get("config_hash") or ""),
            str(row.get("config_digest") or ""),
            str(row.get("scenario_digest") or ""),
            str(row.get("map_digest") or ""),
        )
        by_cell[key] = {
            "cell_id": row.get("cell_id"),
            "outcome_counts": _decode_json(row.get("outcome_counts_json")),
            "config_digest": row.get("config_digest"),
            "entropy": row.get("entropy"),
            "seed_count": row.get("seed_count"),
            "uncertainty": _decode_json(row.get("uncertainty_json")),
            "boundary_context": _decode_json(row.get("boundary_context_json")),
            "representative_episode_id": row.get("representative_episode_id"),
            "representative_status": row.get("representative_status"),
            "boundary_status": row.get("boundary_status"),
            "outlier_status": row.get("outlier_status"),
        }
    for row in comparison_rows:
        left_id = str(row.get("left_episode_id") or "")
        right_id = str(row.get("right_episode_id") or "")
        if left_id:
            by_episode_comparisons[left_id].append(dict(row))
        if right_id:
            by_episode_comparisons[right_id].append(dict(row))
    result: list[dict[str, Any]] = []
    for row in episode_rows:
        episode_id = str(row.get("episode_id") or "")
        trace_steps: list[dict[str, Any]] = []
        for step in sorted(
            by_episode_steps.get(episode_id, []), key=lambda item: int(item.get("step") or 0)
        ):
            robot = {
                "actor_id": "robot",
                "position": [step.get("robot_x"), step.get("robot_y")],
                "heading": step.get("heading_rad"),
                "velocity": [step.get("robot_vx"), step.get("robot_vy")],
                "radius_m": None,
            }
            pedestrians = []
            for actor in by_episode_actors.get((episode_id, int(step.get("step") or 0)), []):
                if str(actor.get("actor_kind")) == "robot":
                    robot["radius_m"] = actor.get("radius_m")
                    continue
                pedestrians.append(
                    {
                        "actor_id": actor.get("actor_id"),
                        "id": _viewer_actor_id(actor.get("actor_id")),
                        "position": [actor.get("x"), actor.get("y")],
                        "velocity": [actor.get("vx"), actor.get("vy")],
                        "radius_m": actor.get("radius_m"),
                    }
                )
            trace_steps.append(
                {
                    "step": step.get("step"),
                    "time_s": step.get("time_s"),
                    "robot": robot,
                    "pedestrians": pedestrians,
                    "controls": {
                        "requested": {
                            "linear_m_s": step.get("requested_linear_m_s"),
                            "turn_rate_rad_s": step.get("requested_turn_rate_rad_s"),
                        },
                        "applied": {
                            "linear_m_s": step.get("applied_linear_m_s"),
                            "turn_rate_rad_s": step.get("applied_turn_rate_rad_s"),
                        },
                    },
                    "events": [],
                }
            )
        stored_coverage = _decode_json(row.get("trace_coverage_json"))
        if not isinstance(stored_coverage, Mapping):
            stored_coverage = {"status": "unavailable", "reason": "coverage_receipt_invalid"}
        provenance = _decode_json(row.get("provenance_json"))
        if not isinstance(provenance, Mapping):
            provenance = {}
        else:
            provenance = dict(provenance)
        artifact_sha = row.get("artifact_sha256")
        if artifact_sha and provenance.get("artifact_sha256") in (None, ""):
            provenance["artifact_sha256"] = artifact_sha
        sorted_step_rows = sorted(
            by_episode_steps.get(episode_id, []),
            key=lambda item: int(item.get("step") or 0),
        )
        first_step_row = sorted_step_rows[0] if sorted_step_rows else None
        units = (
            _decode_json(first_step_row.get("units_json"))
            if isinstance(first_step_row, dict)
            else None
        )
        analysis_trace = {
            "schema_version": "analysis-trace.v1",
            "scenario_id": row.get("scenario_id"),
            "planner": row.get("planner"),
            "map_digest": provenance.get("map_digest"),
            "map_file": provenance.get("map_file"),
            "scenario_digest": provenance.get("scenario_digest"),
            "config_hash": row.get("config_hash") or provenance.get("config_hash"),
            "config_digest": provenance.get("config_digest"),
            "git_hash": provenance.get("git_hash"),
            "planner_commit": provenance.get("planner_commit"),
            "dt": provenance.get("dt"),
            "horizon": provenance.get("horizon"),
            "actor_geometry": provenance.get("actor_geometry"),
            "actor_id_source": provenance.get("actor_id_source"),
            "artifact_sha256": artifact_sha,
            "coordinate_frame": (
                first_step_row.get("coordinate_frame") if isinstance(first_step_row, dict) else None
            ),
            "units": units,
            "steps": trace_steps,
            "events": by_episode_events.get(episode_id, []),
        }
        stored_trace = _decode_json(row.get("analysis_trace_json"))
        if isinstance(stored_trace, Mapping):
            # The exact source envelope is the replay authority.  The flattened
            # tables remain queryable projections, but must not be reserialized
            # into a different artifact identity during v2 adaptation.
            analysis_trace = json.loads(canonical_json(stored_trace))
        reconstructed = {
            "episode_id": episode_id,
            "scenario_id": row.get("scenario_id"),
            "seed": row.get("seed"),
            "algo": row.get("planner"),
            "status": row.get("execution_status"),
            "row_status": row.get("row_status"),
            "outcome": _decode_json(row.get("outcome_json")) or {},
            "provenance": provenance,
            "config_hash": row.get("config_hash") or provenance.get("config_hash"),
            "config_digest": provenance.get("config_digest"),
            "git_hash": provenance.get("git_hash"),
            "metrics": by_episode_features.get(episode_id, {}),
            "algorithm_metadata": {"analysis_trace": analysis_trace},
            "cell_context": by_cell.get(
                (
                    str(row.get("planner") or ""),
                    str(row.get("scenario_id") or ""),
                    str(row.get("config_hash") or provenance.get("config_hash") or ""),
                    str(row.get("config_digest") or provenance.get("config_digest") or ""),
                    str(row.get("scenario_digest") or provenance.get("scenario_digest") or ""),
                    str(row.get("map_digest") or provenance.get("map_digest") or ""),
                )
            ),
            "comparison_receipts": by_episode_comparisons.get(episode_id, []),
        }
        computed_coverage = trace_coverage(reconstructed)
        if integrity_errors or missing_tables or stored_coverage.get("status") != "complete":
            computed_coverage = dict(stored_coverage)
            if integrity_errors or missing_tables:
                computed_coverage = {
                    **computed_coverage,
                    "status": "unavailable",
                    "reason": "analysis_trace_fields_incomplete",
                    "missing_tables": sorted(missing_tables),
                    "integrity_errors": integrity_errors,
                }
        reconstructed["trace_coverage"] = computed_coverage
        result.append(reconstructed)
    return result


def _v2_integrity_errors(store: Path) -> list[str]:  # noqa: C901, PLR0912
    """Validate the v2 checksum receipt before trusting reconstructed rows."""

    manifest_path = store / "manifest.json"
    if not manifest_path.is_file():
        return ["manifest_missing"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ["manifest_unreadable"]
    if not isinstance(manifest, Mapping):
        return ["manifest_invalid"]
    if manifest.get("schema_version") != "campaign-result-store.v2":
        return ["manifest_schema_mismatch"]
    tables = manifest.get("tables")
    expected_tables = {"episodes", "steps", "actors", "events", "features", "cells", "comparisons"}
    if not isinstance(tables, Mapping) or set(tables) != expected_tables:
        return ["manifest_table_contract_invalid"]
    manifest_errors: list[str] = []
    for table_name in sorted(expected_tables):
        table = tables.get(table_name)
        expected_file = f"{table_name}.parquet"
        if not isinstance(table, Mapping) or table.get("file") != expected_file:
            manifest_errors.append(f"manifest_table_file_invalid:{table_name}")
        elif not (store / expected_file).is_file():
            manifest_errors.append(f"manifest_table_missing:{table_name}")
        elif not isinstance(table.get("rows"), int) or table.get("rows") < 0:
            manifest_errors.append(f"manifest_table_rows_invalid:{table_name}")

    checksum_path = store / "SHA256SUMS"
    if not checksum_path.is_file():
        return [*manifest_errors, "checksum_receipt_missing"]
    errors: list[str] = list(manifest_errors)
    seen_files: set[str] = set()
    try:
        lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return [*errors, "checksum_receipt_unreadable"]
    for line in lines:
        if not line.strip():
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError:
            errors.append("checksum_receipt_malformed")
            continue
        if not re.fullmatch(r"[0-9a-f]{64}", expected) or Path(relative).is_absolute():
            errors.append(f"checksum_entry_invalid:{relative}")
            continue
        if ".." in Path(relative).parts:
            errors.append(f"checksum_entry_invalid:{relative}")
            continue
        seen_files.add(relative)
        path = store / relative
        if not path.is_file():
            errors.append(f"checksum_missing:{relative}")
            continue
        if _sha256_file(path) != expected:
            errors.append(f"checksum_mismatch:{relative}")
    expected_files = {f"{name}.parquet" for name in expected_tables} | {"manifest.json"}
    for relative in sorted(expected_files - seen_files):
        errors.append(f"checksum_entry_missing:{relative}")
    for relative in sorted(seen_files - expected_files):
        errors.append(f"checksum_entry_unexpected:{relative}")
    return errors


def _v2_row_count_errors(store: Path, actual_counts: Mapping[str, int]) -> list[str]:
    """Compare manifest row counts with the tables actually read."""

    try:
        manifest = json.loads((store / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ["manifest_unreadable"]
    tables = manifest.get("tables") if isinstance(manifest, Mapping) else None
    if not isinstance(tables, Mapping):
        return ["manifest_table_contract_invalid"]
    errors: list[str] = []
    for name, actual in actual_counts.items():
        table = tables.get(name)
        expected = table.get("rows") if isinstance(table, Mapping) else None
        if isinstance(expected, int) and expected != actual:
            errors.append(f"manifest_row_count_mismatch:{name}")
    return errors


def _viewer_actor_id(value: Any) -> Any:
    """Preserve numeric actor ids for legacy viewer adapters when possible."""

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    text = str(value or "")
    if text.startswith("pedestrian-"):
        suffix = text.removeprefix("pedestrian-")
        try:
            return int(suffix)
        except ValueError:
            pass
    return value


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    """Read a Parquet table through DuckDB or pandas."""

    duckdb = try_import("duckdb")
    if duckdb is not None:
        connection = duckdb.connect(database=":memory:")
        try:
            result = connection.execute("SELECT * FROM read_parquet(?)", [str(path)])
            columns = [str(item[0]) for item in result.description]
            return [dict(zip(columns, row, strict=True)) for row in result.fetchall()]
        finally:
            connection.close()
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Parquet analysis requires duckdb or pandas") from exc
    return pd.read_parquet(path).to_dict(orient="records")


def _decode_json(value: Any) -> Any:
    """Decode a stored JSON column."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return value


def _row_from_v2_store(row: Mapping[str, Any], store: Path) -> dict[str, Any]:
    """Rehydrate the episode-level fields required by proposal selection."""

    def decode(name: str) -> Any:
        value = row.get(name)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        try:
            return json.loads(str(value))
        except (TypeError, json.JSONDecodeError):
            return value

    return {
        "episode_id": row.get("episode_id"),
        "scenario_id": row.get("scenario_id"),
        "seed": row.get("seed"),
        "algo": row.get("planner"),
        "status": row.get("execution_status"),
        "row_status": row.get("row_status"),
        "outcome": decode("outcome_json") or {},
        "provenance": decode("provenance_json") or {"artifact_uri": str(store)},
        "algorithm_metadata": {"analysis_trace": None},
        "trace_coverage": decode("trace_coverage_json") or {"status": "unavailable"},
    }


def _candidate(  # noqa: C901, PLR0912, PLR0915
    record: Mapping[str, Any],
    *,
    interestingness_weights: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize one episode into an eligibility and metric candidate."""

    raw_record = dict(record)
    computed_coverage = trace_coverage(raw_record)
    supplied_coverage = record.get("trace_coverage")
    coverage = computed_coverage
    provenance = record.get("provenance") if isinstance(record.get("provenance"), Mapping) else {}
    row_status = str(record.get("row_status") or record.get("status") or "native")
    blockers: list[str] = []
    for status_value in (record.get("row_status"), record.get("status")):
        normalized_status = str(status_value or "").lower()
        if normalized_status in INELIGIBLE_STATUSES:
            blockers.append(f"execution_status:{normalized_status}")
    metadata = record.get("algorithm_metadata")
    if isinstance(metadata, Mapping):
        for key in ("status", "readiness_status", "preflight_status", "execution_mode"):
            nested = str(metadata.get(key) or "").lower()
            if nested in INELIGIBLE_STATUSES:
                blockers.append(f"execution_metadata:{key}={nested}")
        if metadata.get("evidence_eligible") is False:
            blockers.append("execution_metadata:evidence_eligible=false")
        foresight = metadata.get("foresight_prediction")
        if isinstance(foresight, Mapping) and foresight.get("evidence_eligible") is False:
            blockers.append("execution_metadata:foresight.evidence_eligible=false")
        if isinstance(foresight, Mapping) and str(foresight.get("status") or "").lower() in {
            "fallback",
            "degraded",
            "unavailable",
        }:
            blockers.append(
                f"execution_metadata:foresight.status={str(foresight.get('status')).lower()}"
            )
        if isinstance(metadata.get("algorithm_metadata"), Mapping):
            nested_metadata = metadata["algorithm_metadata"]
            if nested_metadata.get("evidence_eligible") is False:
                blockers.append("execution_metadata:evidence_eligible=false")
    if isinstance(record.get("evidence_eligible"), bool) and not record["evidence_eligible"]:
        blockers.append("evidence_eligible=false")
    if isinstance(record.get("integrity"), Mapping):
        contradictions = record["integrity"].get("contradictions")
        if isinstance(contradictions, list) and contradictions:
            blockers.append("integrity:contradictions_present")
    if isinstance(supplied_coverage, Mapping) and supplied_coverage.get("status") == "complete":
        if computed_coverage.get("status") != "complete":
            blockers.append("trace_coverage:stored_complete_receipt_conflict")
    elif isinstance(supplied_coverage, Mapping) and supplied_coverage.get("status") != "complete":
        # A normalized v2 store carries a coverage receipt for the source
        # tables.  Do not let a complete embedded trace overrule an
        # unavailable receipt caused by missing or tampered projections.
        supplied_reason = str(supplied_coverage.get("reason") or "analysis_trace_fields_incomplete")
        blockers.append(f"trace_coverage:{supplied_reason}")
    if coverage.get("status") != "complete":
        blockers.append(f"trace_coverage:{coverage.get('reason') or 'incomplete'}")
    trace = metadata.get("analysis_trace") if isinstance(metadata, Mapping) else None
    artifact_sha = provenance.get("artifact_sha256")
    if not isinstance(artifact_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", artifact_sha):
        blockers.append("provenance:artifact_sha256_missing")
    if isinstance(trace, Mapping):
        embedded_sha = trace.get("artifact_sha256")
        if not isinstance(embedded_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", embedded_sha):
            blockers.append("provenance:artifact_sha256_invalid")
        elif artifact_sha != embedded_sha:
            blockers.append("provenance:artifact_sha256_mismatch")
        elif embedded_sha != trace_artifact_sha256(trace):
            blockers.append("provenance:artifact_sha256_invalid")
    else:
        blockers.append("provenance:analysis_trace_missing")
    episode_id = str(record.get("episode_id") or "").strip()
    scenario_id = str(record.get("scenario_id") or record.get("scenario") or "").strip()
    planner = str(record.get("algo") or record.get("planner") or "").strip()
    if not episode_id:
        blockers.append("identity:episode_id_missing")
    if not scenario_id:
        blockers.append("identity:scenario_id_missing")
    if not planner:
        blockers.append("identity:planner_missing")
    outcome = record.get("outcome") if isinstance(record.get("outcome"), Mapping) else {}
    success, collision = canonical_outcome_flags(outcome)
    return {
        "episode_id": episode_id,
        "scenario_id": scenario_id,
        "planner": planner,
        "seed": _int(record.get("seed")),
        "row_status": row_status,
        "coverage": dict(coverage),
        "provenance": dict(provenance),
        "cell_context": (
            dict(record.get("cell_context"))
            if isinstance(record.get("cell_context"), Mapping)
            else None
        ),
        "comparison_receipts": (
            list(record.get("comparison_receipts"))
            if isinstance(record.get("comparison_receipts"), list)
            else []
        ),
        "outcome": {"success": success, "collision": collision, "label": _outcome_label(record)},
        "metrics": _episode_metrics(record),
        "eligible": not blockers,
        "exclusion_reasons": blockers,
        "interestingness_score": _interestingness(record, weights=interestingness_weights),
        "raw": dict(record),
    }


def _episode_metrics(record: Mapping[str, Any]) -> dict[str, float | None]:
    """Extract interpretable metrics without fabricating missing values."""

    metrics = record.get("metrics") if isinstance(record.get("metrics"), Mapping) else {}
    derived = derive_episode_metrics(record)

    def value(*keys: str) -> float | None:
        """Prefer recorded aggregate values, then v2 trace-derived features."""

        recorded = _first_number(metrics, *keys)
        return recorded if recorded is not None else _first_number(derived, *keys)

    return {
        "surface_clearance_min": value(
            "surface_clearance_min", "min_surface_clearance", "min_separation"
        ),
        "progress": value("progress", "route_progress", "distance_travelled"),
        "control_effort": value("control_effort", "action_effort"),
        "applied_linear_control_effort": value("applied_linear_control_effort"),
        "applied_turn_control_effort": value("applied_turn_control_effort"),
        "event_time": value("event_time", "first_collision_time"),
        "ttc_min": value("ttc_min"),
        "cpa_min": value("cpa_min"),
        "closing_speed_max": value("closing_speed_max"),
        "braking_response_time": value("braking_response_time"),
        "turning_response_time": value("turning_response_time"),
        "critical_duration_integral": value("critical_duration_integral"),
        "stall_duration": value("stall_duration"),
        "reversal_count": value("reversal_count"),
        "detour_ratio": value("detour_ratio"),
        "clipping_steps": value("clipping_steps"),
        "fallback_steps": value("fallback_steps"),
        "outcome_score": value("outcome_score"),
    }


def _interestingness(
    record: Mapping[str, Any], *, weights: Mapping[str, Any] | None = None
) -> float:
    """Use a transparent scalar only for broad exploratory triage."""

    configured = weights if isinstance(weights, Mapping) else {}

    def weight(name: str, default: float) -> float:
        """Read a finite configured weight without letting config break triage."""

        value = configured.get(name, default)
        return float(value) if _finite_number(value) else default

    metrics = _episode_metrics(record)
    clearance = metrics.get("surface_clearance_min")
    score = 0.0
    if clearance is not None:
        score += weight("surface_clearance_min", 1.0) * max(0.0, 2.0 - float(clearance))
    outcome = record.get("outcome") if isinstance(record.get("outcome"), Mapping) else {}
    success, collision = canonical_outcome_flags(outcome)
    outcome_salience = 1.0 if collision else (0.25 if success else 0.0)
    score += weight("outcome_salience", 1.0) * outcome_salience
    effort = metrics.get("control_effort")
    if effort is not None:
        score += weight("control_effort", 0.25) * min(1.0, abs(float(effort)) / 10.0)
    return round(score, 9)


def _build_proposal(
    candidates: list[dict[str, Any]], *, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Build the complete proposed portfolio and reason ledger."""

    eligible = [item for item in candidates if item["eligible"]]
    roles = [str(role) for role in config.get("portfolio", {}).get("roles", [])]
    max_cases = int(config.get("portfolio", {}).get("max_cases", 12))
    role_candidates = {role: _role_candidates(role, eligible) for role in roles}
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    unavailable_roles: list[dict[str, Any]] = []
    for role in roles:
        options = role_candidates[role]
        if not options:
            unavailable_roles.append(
                {"role": role, "reason": _role_unavailable_reason(role, candidates)}
            )
            continue
        pareto = _pareto_front(options)
        if role in {"seed_sensitivity", "planner_upset"}:
            pair = _comparison_pair(role, options)
            if pair is None:
                unavailable_roles.append({"role": role, "reason": "no_compatible_pair"})
                continue
            pair_ids = [str(item["episode_id"]) for item in pair]
            for chosen in pair:
                if chosen["episode_id"] in selected_ids:
                    continue
                selected_ids.add(chosen["episode_id"])
                selected.append(
                    _selection_record(
                        role,
                        chosen,
                        options,
                        pareto,
                        comparison_pair_ids=pair_ids,
                    )
                )
            continue
        chosen = sorted(pareto, key=lambda item: (-_role_score(role, item), item["episode_id"]))[0]
        if chosen["episode_id"] in selected_ids:
            continue
        selected_ids.add(chosen["episode_id"])
        selected.append(_selection_record(role, chosen, options, pareto))
    selected = _truncate_portfolio_atomically(selected, max_cases)
    selected_ids = {item["case_id"] for item in selected}
    excluded = []
    for candidate in candidates:
        if candidate["episode_id"] in selected_ids:
            continue
        reason = candidate["exclusion_reasons"] or ["not_selected_after_role_coverage"]
        excluded.append(
            {
                "case_id": candidate["episode_id"],
                "eligible": candidate["eligible"],
                "reasons": reason,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "metric_profile_version": METRIC_PROFILE_VERSION,
        "selection_authority": "machine_proposal_plus_author_admission_overlay",
        "claim_boundary": "descriptive observed cases; no causal divergence or planner ranking",
        "portfolio": selected,
        "excluded": sorted(excluded, key=lambda item: item["case_id"]),
        "unavailable_roles": unavailable_roles,
        "runner_ups": _runner_up_ledger(role_candidates, selected),
        "artifact_inventory": {
            item["episode_id"]: item["provenance"].get("artifact_sha256")
            for item in candidates
            if item.get("episode_id") and item.get("provenance", {}).get("artifact_sha256")
        },
        "candidate_count": len(candidates),
        "eligible_count": len(eligible),
    }


def _runner_up_ledger(
    role_candidates: Mapping[str, list[dict[str, Any]]],
    selected: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Return stable runner-up explanations for every configured role."""

    selected_by_role: dict[str, set[str]] = defaultdict(set)
    selected_case_by_role: dict[str, Mapping[str, Any]] = {}
    for item in selected:
        role = str(item.get("primary_role"))
        selected_by_role[role].add(str(item.get("case_id")))
        selected_case_by_role.setdefault(
            role,
            {
                "episode_id": item.get("case_id"),
                "interestingness_score": item.get("selection_reason", {}).get(
                    "interestingness_score"
                ),
            },
        )
    ledger: dict[str, list[dict[str, Any]]] = {}
    for role, options in role_candidates.items():
        chosen_ids = selected_by_role.get(role, set())
        chosen = selected_case_by_role.get(role, {})
        ledger[role] = [
            _runner_up(item, chosen)
            for item in sorted(
                options, key=lambda item: (-_role_score(role, item), item["episode_id"])
            )
            if item["episode_id"] not in chosen_ids
        ][:3]
    return ledger


def _role_candidates(role: str, candidates: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return candidates satisfying a role's local predicate."""

    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_scenario[candidate["scenario_id"]].append(candidate)
    if role == "seed_sensitivity":
        return [
            item
            for group in by_scenario.values()
            if len({entry["seed"] for entry in group}) > 1
            for item in group
        ]
    if role == "planner_upset":
        return [
            item
            for group in by_scenario.values()
            if len({entry["planner"] for entry in group}) > 1
            for item in group
        ]
    if role == "safety_boundary":
        return [
            item
            for item in candidates
            if _relation_metric(item, "boundary_context", "safety_boundary")
        ]
    if role == "metric_disagreement":
        return [
            item
            for item in candidates
            if _relation_metric(item, "metric_disagreement", "metric_disagreement_score")
        ]
    if role == "cross_cell_inversion":
        return [
            item
            for item in candidates
            if _relation_metric(item, "cross_cell_inversion", "cross_cell_inversion_score")
        ]
    if role == "representative_control":
        return [
            item
            for item in candidates
            if _relation_metric(item, "representative_control", "cell_representative_status")
        ]
    return []


def _relation_metric(candidate: Mapping[str, Any], *keys: str) -> bool:
    """Return true only for an explicit relation/context receipt."""

    raw = candidate.get("raw")
    metrics = raw.get("metrics") if isinstance(raw, Mapping) else None
    contexts = [metrics]
    cell_context = candidate.get("cell_context")
    if isinstance(cell_context, Mapping):
        contexts.append(cell_context)
    raw_cell_context = raw.get("cell_context") if isinstance(raw, Mapping) else None
    if isinstance(raw_cell_context, Mapping):
        contexts.append(raw_cell_context)
    for context in contexts:
        for key in keys:
            value = context.get(key)
            if isinstance(value, Mapping):
                value = value.get("status") or value.get("value")
            if isinstance(value, bool):
                if value:
                    return True
            elif _finite_number(value) and float(value) > 0.0:
                return True
            elif isinstance(value, str) and value in {
                "available",
                "observed",
                "medoid",
                "boundary",
            }:
                return True
    return False


def _comparison_pair(role: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    """Choose a stable two-member comparison pair for a pair-valued role."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate["scenario_id"]].append(candidate)
    pair_options: list[tuple[float, tuple[str, str], str, list[dict[str, Any]]]] = []
    dimension = "seed" if role == "seed_sensitivity" else "planner"
    for scenario_id, group in grouped.items():
        ranked = sorted(group, key=lambda item: (-_role_score(role, item), item["episode_id"]))
        for left_index, left in enumerate(ranked):
            for right in ranked[left_index + 1 :]:
                if not _pair_matches_role(role, left, right, dimension):
                    continue
                if not is_comparison_compatible(left.get("raw", {}), right.get("raw", {})):
                    continue
                selected = sorted([left, right], key=lambda item: item["episode_id"])
                ids = (str(selected[0]["episode_id"]), str(selected[1]["episode_id"]))
                pair_options.append(
                    (_role_score(role, left) + _role_score(role, right), ids, scenario_id, selected)
                )
    if not pair_options:
        return None
    return sorted(pair_options, key=lambda item: (-item[0], item[1], item[2]))[0][3]


def _pair_matches_role(
    role: str, left: Mapping[str, Any], right: Mapping[str, Any], dimension: str
) -> bool:
    """Return whether a pair changes only the dimension named by its role."""

    if left[dimension] == right[dimension]:
        return False
    if dimension == "seed" and (left["seed"] is None or right["seed"] is None):
        return False
    if role == "seed_sensitivity":
        return left["planner"] == right["planner"]
    if role == "planner_upset":
        return left["seed"] == right["seed"]
    return True


def _truncate_portfolio_atomically(
    selected: list[dict[str, Any]], max_cases: int
) -> list[dict[str, Any]]:
    """Apply the portfolio bound without splitting a declared comparison pair."""

    if max_cases <= 0:
        return []
    kept: list[dict[str, Any]] = []
    kept_ids: set[str] = set()
    for item in selected:
        if item["case_id"] in kept_ids:
            continue
        pair_ids = [str(value) for value in item.get("comparison_pair_ids", [])]
        pair_ids = [value for value in pair_ids if value != str(item["case_id"])]
        if pair_ids:
            pair = [
                item,
                *[candidate for candidate in selected if candidate["case_id"] in pair_ids],
            ]
            if len(kept) + len(pair) > max_cases:
                continue
            kept.extend(pair)
            kept_ids.update(candidate["case_id"] for candidate in pair)
            continue
        if len(kept) >= max_cases:
            break
        kept.append(item)
        kept_ids.add(item["case_id"])
    return kept


def _pareto_front(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute a stable nondominated front over role-local scalar dimensions."""

    if len(candidates) < 2:
        return list(candidates)
    front: list[dict[str, Any]] = []
    for candidate in candidates:
        dominated = False
        for other in candidates:
            if other is candidate:
                continue
            a = _vector(candidate)
            b = _vector(other)
            if all(x <= y for x, y in zip(a, b, strict=True)) and any(
                x < y for x, y in zip(a, b, strict=True)
            ):
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return sorted(front, key=lambda item: item["episode_id"])


def _vector(candidate: Mapping[str, Any]) -> tuple[float, float, float, float]:
    """Return comparable dimensions (lower clearance is more interesting)."""

    metrics = candidate["metrics"]
    clearance = metrics.get("surface_clearance_min")
    effort = metrics.get("control_effort")
    progress = metrics.get("progress")
    outcome = 1.0 if candidate["outcome"]["collision"] else 0.0
    return (
        float(clearance) if clearance is not None else 1.0e9,
        -outcome,
        -float(progress) if progress is not None else 1.0e9,
        -float(effort) if effort is not None else 1.0e9,
    )


def _role_score(role: str, candidate: Mapping[str, Any]) -> float:
    """Rank candidates locally after Pareto filtering."""

    metrics = candidate["metrics"]
    clearance = metrics.get("surface_clearance_min")
    score = float(candidate.get("interestingness_score", 0.0))
    if clearance is not None:
        score += max(0.0, 1.0 - float(clearance))
    if role == "seed_sensitivity":
        score += 0.5 if candidate["outcome"]["collision"] else 0.0
    if role == "planner_upset":
        score += 0.5
    return score


def _selection_record(
    role: str,
    chosen: Mapping[str, Any],
    options: list[dict[str, Any]],
    pareto: list[dict[str, Any]],
    *,
    comparison_pair_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Build a selection ledger row with runner-up explanation."""

    ranked = sorted(options, key=lambda item: (-_role_score(role, item), item["episode_id"]))
    runner_up = next((item for item in ranked if item["episode_id"] != chosen["episode_id"]), None)
    raw_trace = (
        chosen.get("raw", {}).get("algorithm_metadata", {}).get("analysis_trace")
        if isinstance(chosen.get("raw"), Mapping)
        and isinstance(chosen.get("raw", {}).get("algorithm_metadata"), Mapping)
        else {}
    )
    return {
        "case_id": chosen["episode_id"],
        "primary_role": role,
        "scenario_id": chosen["scenario_id"],
        "planner": chosen["planner"],
        "seed": chosen["seed"],
        "selection_reason": {
            "role": role,
            "pareto_status": "nondominated"
            if chosen["episode_id"] in {item["episode_id"] for item in pareto}
            else "dominated",
            "interestingness_score": chosen["interestingness_score"],
            "why_selected_over_runner_up": (
                (
                    "paired role coverage across distinct dimensions; stable episode-id tie-break"
                    if comparison_pair_ids
                    else "higher role-local score; stable episode-id tie-break"
                )
                if runner_up is not None
                else (
                    "paired role coverage across distinct dimensions"
                    if comparison_pair_ids
                    else "only eligible candidate for this role"
                )
            ),
            "comparison_pair_ids": comparison_pair_ids or [],
        },
        "outcome": chosen["outcome"],
        "metrics": chosen["metrics"],
        "coverage": chosen["coverage"],
        "provenance": chosen["provenance"],
        "config_hash": raw_trace.get("config_hash"),
        "config_digest": raw_trace.get("config_digest"),
        "scenario_digest": raw_trace.get("scenario_digest"),
        "map_digest": raw_trace.get("map_digest"),
        "cell_context": chosen.get("cell_context"),
        "trace": raw_trace or None,
        "shared_prefix": False,
        "comparison_pair_ids": comparison_pair_ids or [],
        "comparison_compatibility": (
            {
                "status": "compatible",
                "pair_ids": sorted(str(value) for value in comparison_pair_ids),
                "shared_prefix": False,
            }
            if comparison_pair_ids
            else None
        ),
        "author_status": "proposed",
    }


def _runner_up(candidate: Mapping[str, Any], chosen: Mapping[str, Any]) -> dict[str, Any]:
    """Return a compact runner-up explanation."""

    candidate_score = candidate.get("interestingness_score")
    chosen_score = chosen.get("interestingness_score")
    score_delta = None
    if _finite_number(candidate_score) and _finite_number(chosen_score):
        score_delta = round(float(chosen_score) - float(candidate_score), 9)
    return {
        "case_id": candidate.get("episode_id"),
        "score": candidate_score,
        "selected_case_id": chosen.get("episode_id") or None,
        "selected_score": chosen_score,
        "score_delta": score_delta,
        "reason_not_selected": "lower role-local score or stable tie-break",
    }


def _role_unavailable_reason(role: str, candidates: list[dict[str, Any]]) -> str:
    """Explain why a role has no eligible candidate."""

    if not candidates:
        return "no_candidates"
    if role in {
        "safety_boundary",
        "metric_disagreement",
        "cross_cell_inversion",
        "representative_control",
    }:
        return "required_relation_metric_unavailable"
    if any(not item["eligible"] for item in candidates):
        return "all_candidates_failed_eligibility"
    return "role_predicate_not_satisfied"


def _write_case_files(output: Path, proposal: Mapping[str, Any]) -> None:
    """Write one compact case record per proposed case."""

    cases = output / "cases"
    cases.mkdir(exist_ok=True)
    seen_targets: set[Path] = set()
    for case in proposal.get("portfolio", []):
        case_id = str(case["case_id"])
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", case_id).strip("._")
        if not safe_id or safe_id in {".", ".."}:
            raise ValueError(f"case id cannot be used as a package filename: {case_id!r}")
        target = (cases / f"{safe_id}.json").resolve()
        if cases.resolve() not in target.parents:
            raise ValueError(f"case id escapes package output: {case_id!r}")
        if target in seen_targets or target.exists():
            raise ValueError(f"case ids collide after package filename sanitization: {case_id!r}")
        seen_targets.add(target)
        target.write_text(json.dumps(case, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _clear_case_files(output: Path) -> None:
    """Remove only owned case projections before regenerating a package."""

    cases = output / "cases"
    if not cases.is_dir():
        return
    for child in cases.iterdir():
        if child.is_file() and child.suffix == ".json":
            child.unlink()


def _write_viewer_blueprint(output: Path, proposal: Mapping[str, Any]) -> None:
    """Write the stable synchronized-view blueprint consumed by the viewer adapter."""

    payload = {
        "schema_version": "case-workbench-viewer-blueprint.v1",
        "layout": "world_plus_absolute_time_tracks",
        "case_ids": [
            str(case.get("case_id"))
            for case in proposal.get("portfolio", [])
            if isinstance(case, Mapping)
        ],
        "views": [
            {"id": "world", "kind": "spatial", "coordinate_frame": "world", "synchronized": True},
            {
                "id": "clearance",
                "kind": "time_series",
                "field": "min_pedestrian_clearance_m",
                "units": "m",
                "synchronized": True,
            },
            {
                "id": "speed",
                "kind": "time_series",
                "field": "applied_linear_speed",
                "units": "m/s",
                "synchronized": True,
            },
            {
                "id": "turn_rate",
                "kind": "time_series",
                "field": "applied_turn_rate",
                "units": "rad/s",
                "synchronized": True,
            },
            {"id": "events", "kind": "event_timeline", "synchronized": True},
            {"id": "state", "kind": "state_inspector", "synchronized": True},
        ],
        "time_policy": "absolute_recorded_time; no normalized-duration alignment",
    }
    (output / "viewer_blueprint.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_audit_dossier(output: Path, proposal: Mapping[str, Any]) -> None:
    """Write a complete, machine-readable audit dossier beside the reduced figure."""

    dossier = {
        "schema_version": "case-workbench-audit-dossier.v1",
        "evidence_status": (
            "admitted"
            if isinstance(proposal.get("author_admission"), Mapping)
            and str(proposal["author_admission"].get("status") or "").lower() == "admitted"
            else "proposed_not_admitted"
        ),
        "claim_boundary": proposal.get("claim_boundary"),
        "portfolio": proposal.get("portfolio", []),
        "excluded": proposal.get("excluded", []),
        "unavailable_roles": proposal.get("unavailable_roles", []),
        "runner_ups": proposal.get("runner_ups", {}),
        "selection_authority": proposal.get("selection_authority"),
    }
    (output / "audit_dossier.json").write_text(
        json.dumps(dossier, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Case workbench audit dossier",
        "",
        "This is the full review ledger. It is not a dissertation figure and does not admit evidence.",
        "",
        f"- evidence status: `{dossier['evidence_status']}`",
        f"- candidates: `{proposal.get('candidate_count', 0)}`",
        f"- eligible: `{proposal.get('eligible_count', 0)}`",
        f"- selected: `{len(proposal.get('portfolio', []))}`",
        "- shared prefix: `false` unless a matched-start receipt is supplied",
        "",
        "## Complete proposed portfolio",
    ]
    for case in proposal.get("portfolio", []):
        if not isinstance(case, Mapping):
            continue
        lines.extend(
            [
                "",
                f"### `{case.get('case_id')}`",
                f"- role: `{case.get('primary_role')}`",
                f"- planner/seed: `{case.get('planner')}` / `{case.get('seed')}`",
                f"- selection reason: {case.get('selection_reason', {}).get('why_selected_over_runner_up')}",
                f"- provenance: `{case.get('provenance', {}).get('artifact_sha256')}`",
                f"- metrics: `{json.dumps(case.get('metrics', {}), sort_keys=True)}`",
            ]
        )
    lines.extend(["", "## Unavailable roles"])
    for role in proposal.get("unavailable_roles", []):
        lines.append(f"- `{role.get('role')}`: {role.get('reason')}")
    lines.extend(["", "## Exclusions"])
    for item in proposal.get("excluded", []):
        lines.append(
            f"- `{item.get('case_id')}`: {', '.join(str(reason) for reason in item.get('reasons', []))}"
        )
    (output / "audit_dossier.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_publication_preview(
    output: Path, proposal: Mapping[str, Any], *, source_gate: Mapping[str, Any]
) -> None:
    """Render a reduced diagnostic preview only after the source gate passes."""

    publication = output / "publication"
    publication.mkdir(exist_ok=True)
    if source_gate.get("status") != "passed":
        (publication / "UNAVAILABLE.json").write_text(
            json.dumps(
                {
                    "status": "unavailable",
                    "reason": str(
                        source_gate.get("reason") or "blocked_pending_exact_source_restore"
                    ),
                    "source_integrity_gate": dict(source_gate),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return
    if not proposal.get("portfolio"):
        (publication / "UNAVAILABLE.json").write_text(
            json.dumps({"status": "unavailable", "reason": "no_proposed_cases"}, indent=2) + "\n",
            encoding="utf-8",
        )
        return
    if try_import("matplotlib") is None:
        (publication / "UNAVAILABLE.json").write_text(
            json.dumps(
                {
                    "status": "unavailable",
                    "reason": "renderer_unavailable:matplotlib_missing",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return
    try:
        from robot_sf.benchmark.case_publication_figure import render_publication_figure

        render_publication_figure(
            output,
            output=publication / "figure.preview.pdf",
            output_format="pdf",
            _allow_unverified_preview=True,
        )
    except (RuntimeError, ValueError) as exc:
        (publication / "UNAVAILABLE.json").write_text(
            json.dumps(
                {
                    "status": "unavailable",
                    "reason": f"renderer_unavailable:{exc.__class__.__name__}",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def _review_memo(proposal: Mapping[str, Any]) -> str:
    """Render a deterministic author review memo."""

    lines = [
        "# Case workbench review memo",
        "",
        "Machine proposal only. Author admission is required before publication.",
        "",
        f"- candidates: {proposal.get('candidate_count', 0)}",
        f"- eligible: {proposal.get('eligible_count', 0)}",
        f"- selected: {len(proposal.get('portfolio', []))}",
        "- shared prefix: false unless a matched-start receipt is supplied",
        "",
        "## Proposed cases",
    ]
    for case in proposal.get("portfolio", []):
        lines.append(
            f"- `{case['case_id']}` ({case['primary_role']}): {case['selection_reason']['why_selected_over_runner_up']}"
        )
    lines.extend(["", "## Unavailable roles"])
    for role in proposal.get("unavailable_roles", []):
        lines.append(f"- `{role['role']}`: {role['reason']}")
    return "\n".join(lines) + "\n"


def _manifest(
    output: Path,
    proposal: Mapping[str, Any],
    source: Path,
    config: str | Path,
    *,
    source_gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Build package provenance manifest."""

    return {
        "schema_version": "case-workbench-package.v1",
        "workbench_schema_version": SCHEMA_VERSION,
        "proposal_sha256": _sha256_json(proposal),
        "machine_proposal_sha256": _sha256_json(proposal),
        "source": {
            "path": source.name,
            "sha256": _sha256_file(source)
            if source.is_file()
            else (_sha256_directory(source) if source.is_dir() else None),
        },
        "config": {"path": Path(config).name, "sha256": _sha256_file(Path(config))},
        "evidence_status": "proposed_not_admitted",
        "source_integrity_gate": dict(source_gate),
        "claim_boundary": "No causal divergence point; no planner ranking; source integrity remains separate.",
        "files": sorted(
            str(path.relative_to(output))
            for path in output.rglob("*")
            if path.is_file() and path.name != "SHA256SUMS"
        ),
    }


def _checksums(output: Path) -> str:
    """Return deterministic checksums for package files."""

    paths = sorted(
        path for path in output.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    return "\n".join(f"{_sha256_file(path)}  {path.relative_to(output)}" for path in paths) + "\n"


def _outcome_label(record: Mapping[str, Any]) -> str:
    """Return a stable descriptive outcome label."""

    outcome = record.get("outcome") if isinstance(record.get("outcome"), Mapping) else {}
    success, collision = canonical_outcome_flags(outcome)
    if collision:
        return "collision"
    if success:
        return "success"
    return str(record.get("termination_reason") or record.get("status") or "unknown")


def _first_number(mapping: Mapping[str, Any], *keys: str) -> float | None:
    """Return the first finite numeric value under the given keys."""

    for key in keys:
        value = mapping.get(key)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ):
            return float(value)
    return None


def _finite_number(value: Any) -> bool:
    """Return whether a value is a finite, non-boolean real number."""

    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _int(value: Any) -> int | None:
    """Coerce an integer-like value."""

    if isinstance(value, bool):
        return None
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _sha256_json(value: Any) -> str:
    """Hash canonical JSON."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    """Hash a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_directory(path: Path) -> str:
    """Hash a directory by sorted relative file names and bytes."""

    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(child.read_bytes())
    return digest.hexdigest()


__all__ = [
    "ADMISSION_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "SOURCE_GATE_SCHEMA_VERSION",
    "admit_package",
    "analyze_cases",
    "apply_admission_overlay",
    "load_workbench_config",
]
