#!/usr/bin/env python3
"""Validate and bundle an accepted 0.0.8 campaign after its episode run.

The canonical release runner deliberately defers publication: the predecessor
equivalence and robot-force reports require its completed episode rows. This
command works on a new copy so the accepted producer remains untouched.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from robot_sf.benchmark.release_acceptance import validate_full_benchmark_release_acceptance
from robot_sf.benchmark.release_protocol import (
    build_release_provenance,
    load_release_campaign_config,
    verify_resolved_release_identity,
)
from robot_sf.common.artifact_paths import get_repository_root
from scripts.analysis.compare_release_007_008 import (
    HISTORICAL_EFFECTIVE_CONFIG_SHA256,
    HISTORICAL_MATRIX_SHA256,
)
from scripts.analysis.compare_release_007_008 import (
    REPORT_SCHEMA as STAGE3_REPORT_SCHEMA,
)
from scripts.tools.run_benchmark_release import (
    _assert_no_historical_release_identity,
    _build_publication_payload,
    _merge_release_provenance,
    _record_publication_payload,
    _run_publication_preflight,
    _write_json,
)
from scripts.validation.check_release_metric_equivalence import (
    _read_scientific_candidate_manifest,
)

BASELINE_ARCHIVE_SHA256 = "684da7c557c426756f22ddbf5cb3270141ee8ae385669a39d36f324852a6fb2f"
BASELINE_SOURCE_SHA = "07f7e8d43084de748915e1b1eb8b2a1603357c6e"
EXPECTED_EPISODES = 20160
_SCIENTIFIC_REPORTS = (
    "reports/scientific_candidate_acceptance.json",
    "reports/metric_equivalence.json",
    "reports/robot_force_validation.json",
)
_SNQI_V2_REPORTS = (
    "reports/snqi_v2_diagnostics.json",
    "reports/snqi_v2_diagnostics.md",
    "reports/snqi_v2_family.json",
    "reports/snqi_v2_family.md",
)
_SNQI_V2_ASSET_ROLES = ("weights", "anchors", "family")
_STAGE3_DEFAULT_IDENTITY = "release/candidate_identity.json"
_STAGE3_DEFAULT_LEDGER = "reports/attribution_ledger.json"
_STAGE3_DEFAULT_REPORT = "reports/stage3_comparison.json"
_STAGE3_DEFAULT_FINDINGS = "reports/stage3_findings.jsonl"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path.name}")
    return payload


def _stage3_paths(
    campaign_root: Path,
    *,
    output_root: Path | None = None,
    candidate_identity: Path | None = None,
    attribution_ledger: Path | None = None,
    comparison_report: Path | None = None,
    comparison_findings: Path | None = None,
) -> dict[str, Path]:
    """Resolve the Stage-3 inputs and outputs for one candidate root."""
    root = campaign_root.resolve()
    output = (output_root or root).resolve()
    return {
        "candidate_identity": (candidate_identity or root / _STAGE3_DEFAULT_IDENTITY).resolve(),
        "attribution_ledger": (attribution_ledger or root / _STAGE3_DEFAULT_LEDGER).resolve(),
        "comparison_report": (comparison_report or output / _STAGE3_DEFAULT_REPORT).resolve(),
        "comparison_findings": (comparison_findings or output / _STAGE3_DEFAULT_FINDINGS).resolve(),
    }


def _require_external_stage3_outputs(producer_root: Path, paths: dict[str, Path]) -> None:
    """Keep comparator outputs outside an immutable producer tree."""
    producer = producer_root.resolve()
    for key in ("comparison_report", "comparison_findings"):
        path = paths[key]
        if path == producer or producer in path.parents:
            raise ValueError(f"Stage-3 {key} must be outside the producer root")


def _materialize_stage3_artifacts(
    candidate_root: Path, paths: dict[str, Path], receipt: dict[str, Any]
) -> dict[str, Path]:
    """Copy receipt-bound Stage-3 artifacts into the publication derivative."""
    destinations = {
        "candidate_identity": candidate_root / _STAGE3_DEFAULT_IDENTITY,
        "attribution_ledger": candidate_root / _STAGE3_DEFAULT_LEDGER,
        "comparison_report": candidate_root / _STAGE3_DEFAULT_REPORT,
        "comparison_findings": candidate_root / _STAGE3_DEFAULT_FINDINGS,
    }
    hashes = {
        "candidate_identity": receipt["candidate_identity_sha256"],
        "attribution_ledger": receipt["attribution_ledger_sha256"],
        "comparison_report": receipt["comparison_report_sha256"],
        "comparison_findings": receipt["findings_sha256"],
    }
    for key, destination in destinations.items():
        source = paths[key]
        if source.resolve() != destination.resolve():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        if _sha256(destination) != hashes[key]:
            raise ValueError(f"Stage-3 {key} changed while materializing derivative")
    return destinations


def _run_stage3_comparator(
    *,
    campaign_root: Path,
    baseline_archive: Path,
    candidate_source_root: Path,
    paths: dict[str, Path],
    log_path: Path,
) -> None:
    """Run the canonical receipt-aware 0.0.7→0.0.8 comparator."""
    _run_gate(
        [
            sys.executable,
            str(get_repository_root() / "scripts/analysis/compare_release_007_008.py"),
            "--historical-bundle",
            str(baseline_archive),
            "--candidate-root",
            str(campaign_root),
            "--candidate-source-root",
            str(candidate_source_root),
            "--candidate-identity",
            str(paths["candidate_identity"]),
            "--attribution-ledger",
            str(paths["attribution_ledger"]),
            "--report-json",
            str(paths["comparison_report"]),
            "--findings-jsonl",
            str(paths["comparison_findings"]),
            "--report-md",
            str(paths["comparison_report"].with_suffix(".md")),
        ],
        log_path,
    )


def _require_stage3_comparison(  # noqa: C901
    *,
    campaign_root: Path,
    baseline_archive: Path,
    expected_source_sha: str,
    paths: dict[str, Path],
) -> dict[str, Any]:
    """Require a complete, hash-bound comparator receipt before publication."""
    report = _read_mapping(paths["comparison_report"])
    identity = _read_mapping(paths["candidate_identity"])
    ledger = _read_mapping(paths["attribution_ledger"])
    if report.get("schema_version") != STAGE3_REPORT_SCHEMA:
        raise ValueError("Stage-3 comparison report schema is invalid")
    if report.get("comparison_passed") is not True:
        raise ValueError("Stage-3 comparison receipt is not passed")
    comparison = report.get("comparison")
    inventory = comparison.get("inventory") if isinstance(comparison, dict) else None
    if (
        not isinstance(comparison, dict)
        or comparison.get("comparison_passed") is not True
        or not isinstance(inventory, dict)
        or any(inventory.values())
        or comparison.get("row_anomalies") != []
        or comparison.get("unexplained_findings") != []
        or comparison.get("orphaned_attributions") != []
    ):
        raise ValueError("Stage-3 comparison receipt has inventory, row, or attribution findings")
    if report.get("read_anomalies") != [] or report.get("attribution_anomalies") != []:
        raise ValueError("Stage-3 comparison receipt has read or attribution anomalies")
    if identity.get("source_sha") != expected_source_sha:
        raise ValueError("Stage-3 candidate identity source differs from finalizer source")
    if ledger.get("candidate_source_sha") != expected_source_sha:
        raise ValueError("Stage-3 attribution ledger source differs from finalizer source")
    if _sha256(baseline_archive) != BASELINE_ARCHIVE_SHA256:
        raise ValueError("frozen 0.0.7 archive checksum mismatch")
    expected_hashes = {
        "candidate_source_sha": expected_source_sha,
        "historical_source_sha": BASELINE_SOURCE_SHA,
        "historical_effective_config_sha256": HISTORICAL_EFFECTIVE_CONFIG_SHA256,
        "historical_matrix_sha256": HISTORICAL_MATRIX_SHA256,
        "historical_bundle_sha256": BASELINE_ARCHIVE_SHA256,
        "candidate_identity_sha256": _sha256(paths["candidate_identity"]),
        "attribution_ledger_sha256": _sha256(paths["attribution_ledger"]),
        "findings_sha256": _sha256(paths["comparison_findings"]),
    }
    for field, expected in expected_hashes.items():
        if report.get(field) != expected:
            raise ValueError(f"Stage-3 comparison {field} is not hash-bound")
    if not campaign_root.is_dir():
        raise ValueError("Stage-3 candidate root is missing")
    return {
        "report": paths["comparison_report"],
        "candidate_identity": paths["candidate_identity"],
        "attribution_ledger": paths["attribution_ledger"],
        "findings": paths["comparison_findings"],
        **expected_hashes,
        "comparison_report_sha256": _sha256(paths["comparison_report"]),
    }


def _promote_scientific_candidate_result(
    candidate_root: Path, stage3_receipt: dict[str, Any]
) -> dict[str, Any]:
    """Promote a pending candidate only after the Stage-3 receipt passes."""
    result_path = candidate_root / "release/scientific_candidate_result.json"
    result = _read_mapping(result_path)
    if result.get("status") not in {"stage3_pending", "accepted_pre_publication"}:
        raise ValueError("scientific candidate is not waiting at the Stage-3 boundary")
    source_sha = result.get("source_sha")
    if source_sha is not None and source_sha != stage3_receipt["candidate_source_sha"]:
        raise ValueError("scientific candidate source differs from Stage-3 receipt")
    result.update(
        {
            "status": "accepted_pre_publication",
            "stage3_status": "passed",
            "stage3_comparison_sha256": stage3_receipt["comparison_report_sha256"],
            "stage3_candidate_identity_sha256": stage3_receipt["candidate_identity_sha256"],
            "stage3_attribution_ledger_sha256": stage3_receipt["attribution_ledger_sha256"],
            "stage3_findings_sha256": stage3_receipt["findings_sha256"],
        }
    )
    _write_json(result_path, result)
    return result


def _promote_scientific_candidate_if_present(
    candidate_root: Path, stage3_receipt: dict[str, Any]
) -> dict[str, Any] | None:
    """Require paired science identity/result files and promote them after Stage 3."""
    identity_path = candidate_root / "release/scientific_candidate.json"
    result_path = candidate_root / "release/scientific_candidate_result.json"
    if not identity_path.exists() and not result_path.exists():
        return None
    if not identity_path.is_file() or not result_path.is_file():
        raise ValueError("scientific candidate identity and result must be paired")
    return _promote_scientific_candidate_result(candidate_root, stage3_receipt)


def _require_producer(producer_root: Path, source_sha: str) -> dict[str, Any]:
    result = _read_mapping(producer_root / "release" / "release_result.json")
    resolved = _read_mapping(producer_root / "release" / "release_manifest.resolved.json")
    campaign = _read_mapping(producer_root / "campaign_manifest.json")
    provenance = resolved.get("provenance")
    release_provenance = result.get("benchmark_release")
    acceptance = result.get("release_acceptance")
    if (
        result.get("release_benchmark_success") is not True
        or result.get("release_status") != "ok"
        or result.get("release_exit_code") != 0
        or not isinstance(acceptance, dict)
        or acceptance.get("status") != "valid"
        or result.get("publication_requested") is not False
        or result.get("publication_preflight_status") != "not_requested"
        or not isinstance(provenance, dict)
        or resolved.get("source_sha") != source_sha
        or provenance.get("source_sha") != source_sha
        or not isinstance(release_provenance, dict)
        or release_provenance.get("source_commit") != source_sha
        or release_provenance.get("release_tag") != resolved.get("release_tag")
        or release_provenance.get("version_doi") != provenance.get("version_doi")
        or campaign.get("git_hash") != source_sha
    ):
        raise ValueError("producer is not an accepted, unpublished exact-source release campaign")
    return result


def _require_producer_identity(
    producer_root: Path, source_sha: str, manifest: Any
) -> dict[str, Any]:
    """Match all producer science declarations to the verified resolved identity."""
    result = _require_producer(producer_root, source_sha)
    resolved = _read_mapping(producer_root / "release" / "release_manifest.resolved.json")
    if resolved != manifest.resolved_manifest_payload:
        raise ValueError("producer resolved manifest differs from the verified release identity")
    provenance = result["benchmark_release"]
    if (
        provenance["release_tag"] != manifest.release_tag
        or provenance["version_doi"] != manifest.version_doi
    ):
        raise ValueError("producer release identity differs from the verified candidate identity")
    return result


def _require_copyable_producer(producer_root: Path) -> None:
    """Reject symlinked and already-inventoried sources before making a candidate copy."""
    if producer_root.is_symlink() or (producer_root / "SHA256SUMS").exists():
        raise ValueError("producer root checksum inventory is already frozen")
    if any(path.is_symlink() for path in producer_root.rglob("*")):
        raise ValueError("producer campaign contains a symlink")


def _raw_episode_hashes(root: Path) -> dict[str, str]:
    """Hash only immutable episode rows, keyed by relative artifact path."""
    paths = sorted((root / "runs").glob("*/episodes.jsonl"))
    if len(paths) != 14:
        raise ValueError("0.0.8 candidate must contain exactly 14 raw episode files")
    return {path.relative_to(root).as_posix(): _sha256(path) for path in paths}


def _require_scientific_report_identities(  # noqa: C901, PLR0912
    producer_root: Path,
    source_sha: str,
    identity: dict[str, Any],
) -> None:
    """Require report bytes and their declared scientific identities to agree.

    The candidate result stores checksums, but a caller could otherwise replace a report and
    update that result together.  Recheck the report's source, matrix, predecessor, and raw-arm
    identities before any derivative copy or publication export is allowed.
    """
    report_paths = {name: producer_root / name for name in _SCIENTIFIC_REPORTS}
    reports = {name: _read_mapping(path) for name, path in report_paths.items()}
    acceptance = reports[_SCIENTIFIC_REPORTS[0]]
    if (
        acceptance.get("status") != "valid"
        or acceptance.get("benchmark_success") is not True
        or acceptance.get("expected_planner_arms") != 14
        or acceptance.get("expected_episode_cells") != EXPECTED_EPISODES
        or acceptance.get("observed_episode_rows") != EXPECTED_EPISODES
        or acceptance.get("unique_episode_identities") != EXPECTED_EPISODES
        or acceptance.get("source_commits") != [source_sha]
        or acceptance.get("blockers") != []
        or acceptance.get("forbidden_status_counts") != {}
    ):
        raise ValueError("scientific candidate acceptance report identity is invalid")

    equivalence = reports[_SCIENTIFIC_REPORTS[1]]
    if (
        equivalence.get("status") not in {"pass", "mismatch"}
        or equivalence.get("baseline_archive_sha256") != BASELINE_ARCHIVE_SHA256
        or equivalence.get("baseline_source_sha") != BASELINE_SOURCE_SHA
        or equivalence.get("candidate_source_sha") != source_sha
        or equivalence.get("expected_rows") != EXPECTED_EPISODES
        or equivalence.get("baseline_rows") != EXPECTED_EPISODES
        or equivalence.get("candidate_rows") != EXPECTED_EPISODES
        or equivalence.get("paired_rows") != EXPECTED_EPISODES
        or not isinstance(equivalence.get("scientific_manifest_differences"), list)
    ):
        raise ValueError("scientific candidate equivalence report identity is invalid")
    force_metrics = equivalence.get("robot_force_metrics")
    if (
        not isinstance(force_metrics, dict)
        or force_metrics.get("status") != "pass"
        or force_metrics.get("checked_rows") != EXPECTED_EPISODES
        or force_metrics.get("failed_rows") != 0
    ):
        raise ValueError("scientific candidate equivalence force report identity is invalid")

    force = reports[_SCIENTIFIC_REPORTS[2]]
    expected_episode_hashes = identity.get("raw_episode_sha256")
    sources = force.get("sources")
    if (
        force.get("classification") != "release_robot_force_validation"
        or force.get("episodes") != EXPECTED_EPISODES
        or force.get("source_commit") != source_sha
        or force.get("equivalence_report_sha256") != _sha256(report_paths[_SCIENTIFIC_REPORTS[1]])
        or not isinstance(expected_episode_hashes, dict)
        or not isinstance(sources, list)
    ):
        raise ValueError("scientific candidate robot-force report identity is invalid")
    observed_sources: dict[str, dict[str, Any]] = {}
    for source in sources:
        if not isinstance(source, dict) or not isinstance(source.get("artifact_path"), str):
            raise ValueError("scientific candidate robot-force source identity is invalid")
        artifact_path = source["artifact_path"]
        if artifact_path in observed_sources:
            raise ValueError("scientific candidate robot-force source identity is duplicated")
        observed_sources[artifact_path] = source
    if set(observed_sources) != set(expected_episode_hashes):
        raise ValueError("scientific candidate robot-force source set differs from raw custody")
    for artifact_path, expected_hash in expected_episode_hashes.items():
        source = observed_sources[artifact_path]
        if source.get("sha256") != expected_hash or type(source.get("rows")) is not int:
            raise ValueError("scientific candidate robot-force source checksum is invalid")

    science = identity.get("scientific_manifest")
    metrics = science.get("metrics") if isinstance(science, dict) else None
    if not isinstance(metrics, dict) or any(
        not isinstance(metrics.get(f"snqi_v2_{role}_path"), str)
        or not metrics[f"snqi_v2_{role}_path"]
        or not isinstance(metrics.get(f"snqi_v2_{role}_sha256"), str)
        or len(metrics[f"snqi_v2_{role}_sha256"]) != 64
        or any(char not in "0123456789abcdef" for char in metrics[f"snqi_v2_{role}_sha256"])
        for role in _SNQI_V2_ASSET_ROLES
    ):
        raise ValueError("scientific candidate SNQI-v2 metrics are incomplete")
    asset_root = get_repository_root().resolve()
    for role in _SNQI_V2_ASSET_ROLES:
        raw_path = Path(metrics[f"snqi_v2_{role}_path"])
        asset_path = raw_path if raw_path.is_absolute() else asset_root / raw_path
        try:
            asset_path = asset_path.resolve()
        except OSError as exc:
            raise ValueError(f"scientific candidate SNQI-v2 asset path is invalid: {role}") from exc
        if not asset_path.is_relative_to(asset_root) or not asset_path.is_file():
            raise ValueError(f"scientific candidate SNQI-v2 asset is missing: {role}")
        if _sha256(asset_path) != metrics[f"snqi_v2_{role}_sha256"]:
            raise ValueError(f"scientific candidate SNQI-v2 asset checksum is invalid: {role}")
    for relative in _SNQI_V2_REPORTS:
        path = producer_root / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"scientific candidate SNQI-v2 report is missing: {relative}")
    for relative in ("reports/snqi_v2_diagnostics.json", "reports/snqi_v2_family.json"):
        report = _read_mapping(producer_root / relative)
        provenance = report.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError(f"scientific candidate SNQI-v2 report identity is invalid: {relative}")
        for role in ("weights", "anchors", "family"):
            if provenance.get(f"snqi_v2_{role}_path") != metrics.get(
                f"snqi_v2_{role}_path"
            ) or provenance.get(f"snqi_v2_{role}_sha256") != metrics.get(f"snqi_v2_{role}_sha256"):
                raise ValueError(
                    f"scientific candidate SNQI-v2 report identity is invalid: {relative}"
                )
        if report.get("episode_count") != EXPECTED_EPISODES:
            raise ValueError(
                f"scientific candidate SNQI-v2 report row count is invalid: {relative}"
            )
    family = _read_mapping(producer_root / "reports/snqi_v2_family.json")
    diagnostics = _read_mapping(producer_root / "reports/snqi_v2_diagnostics.json")
    if (
        family.get("family") != "V2-F"
        or not isinstance(family.get("vectors"), list)
        or len(family["vectors"]) != 2013
        or family.get("stratified_count") != 2011
        or diagnostics.get("family_report") != "snqi_v2_family.json"
    ):
        raise ValueError("scientific candidate SNQI-v2 family report identity is invalid")


def _require_scientific_candidate(  # noqa: C901, PLR0912
    producer_root: Path, source_sha: str, manifest: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reverify accepted pre-publication custody and DOI-bound science identity."""
    identity_path = producer_root / "release/scientific_candidate.json"
    result_path = producer_root / "release/scientific_candidate_result.json"
    identity = _read_mapping(identity_path)
    result = _read_mapping(result_path)
    if (
        identity.get("schema_version") != "benchmark-scientific-candidate.v1"
        or result.get("schema_version") != "benchmark-scientific-candidate-result.v1"
        or result.get("status") not in {"stage3_pending", "accepted_pre_publication"}
        or identity.get("source_sha") != source_sha
        or result.get("source_sha") != source_sha
        or manifest.source_sha != source_sha
        or result.get("identity_file_sha256") != _sha256(identity_path)
        or identity.get("baseline_archive_sha256") != BASELINE_ARCHIVE_SHA256
        or result.get("baseline_archive_sha256") != BASELINE_ARCHIVE_SHA256
    ):
        raise ValueError(
            "producer is not an accepted exact-source scientific candidate or Stage-3 pending candidate"
        )
    unsigned = dict(identity)
    identity_digest = unsigned.pop("scientific_identity_sha256", None)
    canonical = (
        json.dumps(
            unsigned,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
            separators=(",", ": "),
        )
        + "\n"
    )
    if hashlib.sha256(canonical.encode()).hexdigest() != identity_digest or (
        result.get("scientific_identity_sha256") != identity_digest
    ):
        raise ValueError("scientific candidate identity hash changed")
    science = identity.get("scientific_manifest")
    resolved = manifest.resolved_manifest_payload
    if not isinstance(science, dict) or not isinstance(resolved, dict):
        raise ValueError("candidate or resolved scientific manifest is missing")
    for field in ("matrix", "scenario", "seed_policy", "planners", "kinematics", "metrics"):
        if science.get(field) != resolved.get(field):
            raise ValueError(f"DOI-bound {field} differs from accepted candidate science")
    if resolved.get("canonical_campaign_config_sha256") != identity.get("campaign_template_sha256"):
        raise ValueError("DOI-bound campaign template differs from candidate")
    if resolved.get("canonical_campaign_config") != identity.get("campaign_template_path"):
        raise ValueError("DOI-bound campaign template path differs from candidate")
    if identity.get("model_registry_sha256") != _sha256(
        get_repository_root() / "model/registry.yaml"
    ):
        raise ValueError("model registry bytes changed since scientific acceptance")
    campaign_manifest = _read_mapping(producer_root / "campaign_manifest.json")
    if identity.get(
        "scientific_config_hash_schema"
    ) != "camera-ready-publication-free.v1" or campaign_manifest.get("config_hash") != identity.get(
        "scientific_config_hash"
    ):
        raise ValueError("scientific campaign config hash differs from producer")
    if _raw_episode_hashes(producer_root) != identity.get("raw_episode_sha256"):
        raise ValueError("accepted candidate raw episode bytes changed")
    sidecars = identity.get("producer_sidecar_sha256")
    expected_sidecars = {
        "campaign_manifest.json",
        "manifest.json",
        "run_meta.json",
        "reports/campaign_summary.json",
        "reports/campaign_integrity.json",
    }
    expected_sidecars.update(
        name
        for raw_name in identity["raw_episode_sha256"]
        for name in (
            f"{Path(raw_name).parent.as_posix()}/episodes.jsonl.provenance.json",
            f"{Path(raw_name).parent.as_posix()}/summary.json",
        )
    )
    if (
        not isinstance(sidecars, dict)
        or set(sidecars) != expected_sidecars
        or any(
            _sha256(producer_root / name) != sidecar_digest
            for name, sidecar_digest in sidecars.items()
        )
    ):
        raise ValueError("accepted candidate producer sidecar bytes changed")
    required_reports = {
        "full_acceptance_sha256": "reports/scientific_candidate_acceptance.json",
        "metric_equivalence_sha256": "reports/metric_equivalence.json",
        "robot_force_validation_sha256": "reports/robot_force_validation.json",
    }
    for key, name in required_reports.items():
        report_path = producer_root / name
        report = _read_mapping(report_path)
        accepted = (
            report.get("classification") == "release_robot_force_validation"
            and report.get("episodes") == EXPECTED_EPISODES
            if key == "robot_force_validation_sha256"
            else report.get("status") in {"valid", "pass", "mismatch"}
        )
        if _sha256(report_path) != result.get(key) or not accepted:
            raise ValueError(f"scientific candidate gate is not accepted: {name}")
    for key, name in (
        ("metric_equivalence_log_sha256", "reports/scientific_candidate_equivalence.log"),
        ("robot_force_log_sha256", "reports/scientific_candidate_force.log"),
    ):
        if _sha256(producer_root / name) != result.get(key):
            raise ValueError(f"scientific candidate gate log changed: {name}")
    _require_scientific_report_identities(producer_root, source_sha, identity)
    if result.get("scientific_identity_sha256") != identity_digest:
        raise ValueError("scientific candidate result is detached from identity")
    return identity, result


def _run_gate(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            command,
            cwd=get_repository_root(),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise ValueError(f"release validation gate failed (exit {result.returncode}): {log_path}")


def _publish_copy(candidate_root: Path, producer_result: dict[str, Any], manifest: Any) -> Path:
    result_path = candidate_root / "release" / "release_result.json"
    producer_path = candidate_root / "release" / "producer_release_result.json"
    if not producer_path.is_file() or _read_mapping(producer_path) != producer_result:
        raise ValueError("candidate producer result snapshot is missing or changed")
    publication_output = _publication_output(candidate_root)
    if publication_output.exists() or publication_output.is_symlink():
        raise FileExistsError(f"candidate publication output already exists: {publication_output}")
    result = dict(producer_result)
    result.update(
        {
            "finalization_status": "pass",
            "publication_requested": True,
            "publication_preflight_status": "pass",
            "publication_preflight_violations": [],
        }
    )
    try:
        # The first export must see the validated candidate state, not the
        # fail-closed provisional result left by the copy phase.
        _write_json(result_path, result)
        publication_payload = _build_publication_payload(
            campaign_root=candidate_root,
            release_tag=manifest.release_tag,
            doi=manifest.doi,
            repository_url=manifest.repository_url,
            output_dir=publication_output,
        )
        for _ in range(5):
            result["publication_bundle"] = publication_payload
            _record_publication_payload(candidate_root, publication_payload)
            _write_json(result_path, result)
            refreshed = _build_publication_payload(
                campaign_root=candidate_root,
                release_tag=manifest.release_tag,
                doi=manifest.doi,
                repository_url=manifest.repository_url,
                output_dir=publication_output,
            )
            if refreshed == publication_payload:
                break
            publication_payload = refreshed
        else:
            raise ValueError("publication bundle descriptor did not stabilize")
        bundle_dir = _publication_path(publication_payload, "bundle_dir")
        archive = _publication_path(publication_payload, "archive_path")
        if not bundle_dir.is_dir() or not archive.is_file():
            raise ValueError("publication export did not leave a bundle directory and archive")
        _assert_no_historical_release_identity(bundle_dir)
        _run_publication_preflight(bundle_dir)
    except BaseException as exc:
        result.update(
            {
                "publication_bundle": None,
                "publication_preflight_status": "fail",
                "publication_preflight_violations": [str(exc) or type(exc).__name__],
                "release_benchmark_success": False,
                "release_status": "publication_preflight_failed",
                "release_status_reason": "post-run publication bundle failed validation",
                "release_exit_code": 2,
            }
        )
        try:
            _write_json(result_path, result)
        finally:
            _remove_owned_output(candidate_root)
        raise
    return archive


def _publication_output(candidate_root: Path) -> Path:
    """Keep unverified exports in a candidate-owned namespace until receipt exists."""
    return candidate_root.parent / f"{candidate_root.name}.publication_candidate"


def _remove_owned_output(candidate_root: Path) -> None:
    """Remove only the export namespace reserved for this candidate."""
    publication_output = _publication_output(candidate_root)
    if publication_output.is_symlink():
        publication_output.unlink()
    elif publication_output.exists():
        shutil.rmtree(publication_output)


def _receipt_paths(candidate_root: Path) -> tuple[Path, Path]:
    """Return the atomic pending and final candidate receipt paths."""
    parent = candidate_root.parent
    name = candidate_root.name
    return (
        parent / f"{name}.finalization_receipt.pending.json",
        parent / f"{name}.finalization_receipt.json",
    )


def _publication_path(payload: dict[str, Any], key: str) -> Path:
    """Require publication paths to remain inside this exact source checkout."""
    raw = payload.get(key)
    if not isinstance(raw, str) or not raw or raw.startswith("<external>/"):
        raise ValueError(f"publication {key} is not a repository path")
    root = get_repository_root().resolve()
    candidate = (root / raw).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"publication {key} leaves the source checkout")
    return candidate


def _mark_candidate_failure(candidate_root: Path, stage: str) -> None:
    """Never leave a failed derivative looking like an accepted producer campaign."""
    result_path = candidate_root / "release" / "release_result.json"
    source_path = (
        result_path
        if result_path.is_file()
        else (candidate_root / "release" / "producer_release_result.json")
    )
    result = _read_mapping(source_path) if source_path.is_file() else {}
    result.update(
        {
            "finalization_status": "fail",
            "finalization_failed_stage": stage,
            "benchmark_success": False,
            "release_benchmark_success": False,
            "release_status": "postrun_finalization_failed",
            "release_status_reason": f"post-run {stage} gate failed; inspect sibling gate log",
            "release_exit_code": 2,
            "publication_bundle": None,
            "publication_preflight_status": "fail",
        }
    )
    _write_json(result_path, result)


def _copy_producer(producer_root: Path, candidate_root: Path) -> None:
    """Stage the copy with a rejected release result before exposing its final path."""
    candidate_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = candidate_root.with_name(f"{candidate_root.name}.copying")
    if staging_root.exists() or staging_root.is_symlink():
        raise FileExistsError(f"incomplete candidate copy already exists: {staging_root}")

    def exclude_active_result(source: str, names: list[str]) -> set[str]:
        if Path(source) == producer_root / "release" and "release_result.json" in names:
            return {"release_result.json"}
        return set()

    # A partial copy must never contain the producer's accepted result under
    # the canonical filename, including if copytree is interrupted mid-file.
    shutil.copytree(producer_root, staging_root, symlinks=False, ignore=exclude_active_result)
    snapshot = staging_root / "release" / "producer_release_result.json"
    shutil.copy2(producer_root / "release" / "release_result.json", snapshot)
    _mark_candidate_failure(staging_root, "incomplete")
    staging_root.rename(candidate_root)


def _prepare_candidate(
    producer_root: Path, candidate_root: Path, source_sha: str, manifest: Any
) -> tuple[Path, Path, dict[str, Any]]:
    """Validate exact-source paths and make one fail-closed derivative copy.

    Returns:
        Resolved producer, resolved candidate, and original producer result.
    """
    if producer_root.is_symlink():
        raise ValueError("producer root is a symlink")
    producer_root = producer_root.resolve(strict=True)
    if candidate_root.is_symlink():
        raise ValueError("publication candidate must not be a symlink")
    candidate_root = candidate_root.resolve()
    if not candidate_root.is_relative_to(get_repository_root().resolve()):
        raise ValueError("publication candidate must be inside the source checkout")
    if candidate_root == producer_root or producer_root in candidate_root.parents:
        raise ValueError("publication candidate must be separate from the producer")
    producer_result = _require_producer_identity(producer_root, source_sha, manifest)
    _require_copyable_producer(producer_root)
    if candidate_root.exists() or candidate_root.is_symlink():
        raise FileExistsError(f"publication candidate already exists: {candidate_root}")
    publication_output = _publication_output(candidate_root)
    if publication_output.exists() or publication_output.is_symlink():
        raise FileExistsError("candidate publication output already exists")
    if any(path.exists() or path.is_symlink() for path in _receipt_paths(candidate_root)):
        raise FileExistsError("candidate finalization receipt already exists")
    _copy_producer(producer_root, candidate_root)
    return producer_root, candidate_root, producer_result


def finalize(  # noqa: PLR0913
    *,
    producer_root: Path,
    candidate_root: Path,
    resolved_identity: Path,
    baseline_archive: Path,
    expected_source_sha: str,
    candidate_source_root: Path | None = None,
    candidate_identity: Path | None = None,
    attribution_ledger: Path | None = None,
    comparison_report: Path | None = None,
    comparison_findings: Path | None = None,
) -> dict[str, Any]:
    """Copy, compare, validate, and publish one exact-source campaign."""
    manifest = verify_resolved_release_identity(resolved_identity)
    if manifest.source_sha != expected_source_sha or manifest.source_sha == BASELINE_SOURCE_SHA:
        raise ValueError("0.0.8 resolved identity has the wrong scientific source")
    if _sha256(baseline_archive) != BASELINE_ARCHIVE_SHA256:
        raise ValueError("frozen 0.0.7 archive checksum mismatch")
    producer_root, candidate_root, producer_result = _prepare_candidate(
        producer_root, candidate_root, expected_source_sha, manifest
    )
    report_dir = candidate_root / "reports"
    stage3_paths = _stage3_paths(
        candidate_root,
        candidate_identity=candidate_identity,
        attribution_ledger=attribution_ledger,
        comparison_report=comparison_report,
        comparison_findings=comparison_findings,
    )
    stage3_source_root = (candidate_source_root or get_repository_root()).resolve()
    stage3_log = candidate_root.parent / f"{candidate_root.name}.stage3_comparison.log"
    equivalence = report_dir / "metric_equivalence.json"
    equivalence_log = candidate_root.parent / f"{candidate_root.name}.equivalence.log"
    force_log = candidate_root.parent / f"{candidate_root.name}.robot_force.log"
    stage = "stage3_comparison"
    try:
        _run_stage3_comparator(
            campaign_root=candidate_root,
            baseline_archive=baseline_archive,
            candidate_source_root=stage3_source_root,
            paths=stage3_paths,
            log_path=stage3_log,
        )
        stage3_receipt = _require_stage3_comparison(
            campaign_root=candidate_root,
            baseline_archive=baseline_archive,
            expected_source_sha=expected_source_sha,
            paths=stage3_paths,
        )
        _materialize_stage3_artifacts(candidate_root, stage3_paths, stage3_receipt)
        stage = "scientific_candidate_promotion"
        _promote_scientific_candidate_if_present(candidate_root, stage3_receipt)
        stage = "metric_equivalence"
        _run_gate(
            [
                sys.executable,
                str(
                    get_repository_root() / "scripts/validation/check_release_metric_equivalence.py"
                ),
                "--baseline-archive",
                str(baseline_archive),
                "--baseline-sha256",
                BASELINE_ARCHIVE_SHA256,
                "--baseline-source-sha",
                BASELINE_SOURCE_SHA,
                "--candidate-root",
                str(candidate_root),
                "--candidate-source-sha",
                expected_source_sha,
                "--expected-rows",
                str(EXPECTED_EPISODES),
                "--diagnostic",
                "--require-robot-force-metrics",
                "--output",
                str(equivalence),
            ],
            equivalence_log,
        )
        stage = "robot_force_validation"
        _run_gate(
            [
                sys.executable,
                str(
                    get_repository_root() / "scripts/analysis/issue_9668_robot_force_validation.py"
                ),
                "--campaign-root",
                str(candidate_root),
                "--expected-source-sha",
                expected_source_sha,
                "--expected-episodes",
                str(EXPECTED_EPISODES),
                "--equivalence-report",
                str(equivalence),
                "--output",
                str(report_dir / "robot_force_validation.json"),
            ],
            force_log,
        )
        stage = "full_release_acceptance"
        acceptance = validate_full_benchmark_release_acceptance(
            candidate_root,
            manifest=manifest,
            campaign_config=load_release_campaign_config(manifest),
        )
        if acceptance.get("status") != "valid":
            raise ValueError("post-copy full release acceptance failed")
        stage = "publication_bundle"
        archive = _publish_copy(candidate_root, producer_result, manifest)
        stage = "finalization_receipt"
        receipt = {
            "schema_version": "benchmark-data-v008-publication-finalization.v1",
            "source_sha": expected_source_sha,
            "resolved_identity_sha256": _sha256(resolved_identity),
            "baseline_archive_sha256": BASELINE_ARCHIVE_SHA256,
            "producer_release_result_sha256": _sha256(
                producer_root / "release" / "release_result.json"
            ),
            "candidate_release_result_sha256": _sha256(
                candidate_root / "release" / "release_result.json"
            ),
            "equivalence_report_sha256": _sha256(equivalence),
            "equivalence_log_sha256": _sha256(equivalence_log),
            "stage3_comparison_sha256": stage3_receipt["comparison_report_sha256"],
            "stage3_candidate_identity_sha256": stage3_receipt["candidate_identity_sha256"],
            "stage3_attribution_ledger_sha256": stage3_receipt["attribution_ledger_sha256"],
            "stage3_findings_sha256": stage3_receipt["findings_sha256"],
            "robot_force_validation_sha256": _sha256(report_dir / "robot_force_validation.json"),
            "robot_force_log_sha256": _sha256(force_log),
            "publication_archive_sha256": _sha256(archive),
            "publication_archive": str(archive),
        }
        pending_receipt, receipt_path = _receipt_paths(candidate_root)
        _write_json(pending_receipt, receipt)
        pending_receipt.rename(receipt_path)
        return receipt
    except BaseException:
        try:
            _mark_candidate_failure(candidate_root, stage)
        finally:
            try:
                if stage == "finalization_receipt":
                    _remove_owned_output(candidate_root)
            finally:
                for path in _receipt_paths(candidate_root):
                    path.unlink(missing_ok=True)
        raise


def finalize_pre_doi_candidate(  # noqa: C901, PLR0912, PLR0913, PLR0915
    *,
    producer_root: Path,
    candidate_root: Path,
    resolved_identity: Path,
    baseline_archive: Path,
    expected_source_sha: str,
    candidate_source_root: Path | None = None,
    candidate_identity: Path | None = None,
    attribution_ledger: Path | None = None,
    comparison_report: Path | None = None,
    comparison_findings: Path | None = None,
) -> dict[str, Any]:
    """After Stage-3 approval, bind DOI metadata in a candidate derivative."""
    manifest = verify_resolved_release_identity(resolved_identity)
    if manifest.source_sha != expected_source_sha or manifest.source_sha == BASELINE_SOURCE_SHA:
        raise ValueError("resolved DOI identity has the wrong scientific source")
    if _sha256(baseline_archive) != BASELINE_ARCHIVE_SHA256:
        raise ValueError("frozen 0.0.7 archive checksum mismatch")
    if producer_root.is_symlink() or candidate_root.is_symlink():
        raise ValueError("candidate source and destination must be regular directories")
    producer_root = producer_root.resolve(strict=True)
    candidate_root = candidate_root.resolve()
    trusted_root = get_repository_root().resolve()
    if not candidate_root.is_relative_to(trusted_root):
        raise ValueError("publication derivative must be inside the exact source checkout")
    if candidate_root == producer_root or producer_root in candidate_root.parents:
        raise ValueError("publication derivative must be separate from its producer")
    _require_copyable_producer(producer_root)
    stage3_output_root = candidate_root.parent / f"{candidate_root.name}.stage3"
    if stage3_output_root.exists() or stage3_output_root.is_symlink():
        raise FileExistsError(f"Stage-3 output directory already exists: {stage3_output_root}")
    stage3_paths = _stage3_paths(
        producer_root,
        output_root=stage3_output_root,
        candidate_identity=candidate_identity,
        attribution_ledger=attribution_ledger,
        comparison_report=comparison_report,
        comparison_findings=comparison_findings,
    )
    _require_external_stage3_outputs(producer_root, stage3_paths)
    stage3_source_root = (candidate_source_root or get_repository_root()).resolve()
    stage3_log = producer_root.parent / f"{producer_root.name}.stage3_comparison.log"
    _run_stage3_comparator(
        campaign_root=producer_root,
        baseline_archive=baseline_archive,
        candidate_source_root=stage3_source_root,
        paths=stage3_paths,
        log_path=stage3_log,
    )
    stage3_receipt = _require_stage3_comparison(
        campaign_root=producer_root,
        baseline_archive=baseline_archive,
        expected_source_sha=expected_source_sha,
        paths=stage3_paths,
    )
    identity, _ = _require_scientific_candidate(producer_root, expected_source_sha, manifest)
    _read_scientific_candidate_manifest(producer_root, expected_source_sha, BASELINE_ARCHIVE_SHA256)
    original_raw = copy.deepcopy(identity["raw_episode_sha256"])
    if candidate_root.exists() or candidate_root.is_symlink():
        raise FileExistsError("publication derivative already exists")
    if _publication_output(candidate_root).exists():
        raise FileExistsError("publication output already exists")
    stage_root = candidate_root.with_name(f"{candidate_root.name}.copying")
    if stage_root.exists() or stage_root.is_symlink():
        raise FileExistsError("incomplete publication derivative copy already exists")
    shutil.copytree(producer_root, stage_root, symlinks=False)
    if _raw_episode_hashes(stage_root) != original_raw:
        raise ValueError("raw episode bytes changed during derivative copy")
    stage_root.rename(candidate_root)
    stage = "publication_identity"
    try:
        derivative_stage3_paths = _materialize_stage3_artifacts(
            candidate_root, stage3_paths, stage3_receipt
        )
        stage3_receipt = _require_stage3_comparison(
            campaign_root=candidate_root,
            baseline_archive=baseline_archive,
            expected_source_sha=expected_source_sha,
            paths=derivative_stage3_paths,
        )
        _promote_scientific_candidate_result(candidate_root, stage3_receipt)
        resolved = manifest.resolved_manifest_payload
        _write_json(candidate_root / "release/release_manifest.resolved.json", resolved)
        provenance = build_release_provenance(
            manifest,
            campaign_root=candidate_root,
            invoked_command="finalize_benchmark_data_v008.py --pre-doi-producer",
            source_commit=expected_source_sha,
        )
        _merge_release_provenance(candidate_root, provenance)
        if _raw_episode_hashes(candidate_root) != original_raw:
            raise ValueError("raw episode bytes changed during publication identity binding")
        stage = "full_release_acceptance"
        campaign_cfg = load_release_campaign_config(manifest)
        campaign_cfg = replace(campaign_cfg, publication_identity_mode="scientific_candidate")
        acceptance = validate_full_benchmark_release_acceptance(
            candidate_root,
            manifest=manifest,
            campaign_config=campaign_cfg,
        )
        if acceptance.get("status") != "valid":
            raise ValueError("DOI-bound derivative failed full release acceptance")
        _write_json(candidate_root / "reports/release_acceptance.json", acceptance)
        summary = _read_mapping(candidate_root / "reports/campaign_summary.json")
        campaign = summary.get("campaign")
        if not isinstance(campaign, dict) or campaign.get("benchmark_success") is not True:
            raise ValueError("DOI-bound derivative has no successful campaign summary")
        release_result = {
            **campaign,
            "benchmark_release": provenance,
            "release_acceptance": acceptance,
            "release_benchmark_success": True,
            "release_status": "ok",
            "release_exit_code": 0,
            "publication_requested": False,
            "publication_preflight_status": "not_requested",
            "publication_bundle": None,
            "scientific_candidate_result_sha256": _sha256(
                candidate_root / "release/scientific_candidate_result.json"
            ),
        }
        _write_json(candidate_root / "release/producer_release_result.json", release_result)
        _write_json(candidate_root / "release/release_result.json", release_result)
        stage = "publication_bundle"
        archive = _publish_copy(candidate_root, release_result, manifest)
        if _raw_episode_hashes(candidate_root) != original_raw:
            raise ValueError("raw episode bytes changed during publication export")
        stage = "finalization_receipt"
        receipt = {
            "schema_version": "benchmark-data-v008-pre-doi-promotion.v1",
            "source_sha": expected_source_sha,
            "scientific_identity_sha256": identity["scientific_identity_sha256"],
            "scientific_candidate_result_sha256": _sha256(
                candidate_root / "release/scientific_candidate_result.json"
            ),
            "stage3_comparison_sha256": stage3_receipt["comparison_report_sha256"],
            "stage3_candidate_identity_sha256": stage3_receipt["candidate_identity_sha256"],
            "stage3_attribution_ledger_sha256": stage3_receipt["attribution_ledger_sha256"],
            "stage3_findings_sha256": stage3_receipt["findings_sha256"],
            "publication_identity_sha256": _sha256(resolved_identity),
            "raw_episode_sha256": original_raw,
            "publication_archive_sha256": _sha256(archive),
            "publication_archive": str(archive),
        }
        pending_receipt, receipt_path = _receipt_paths(candidate_root)
        _write_json(pending_receipt, receipt)
        pending_receipt.rename(receipt_path)
        return receipt
    except BaseException:
        try:
            _mark_candidate_failure(candidate_root, stage)
        finally:
            _remove_owned_output(candidate_root)
            for path in _receipt_paths(candidate_root):
                path.unlink(missing_ok=True)
        raise


def main() -> int:
    """Run the post-campaign release finalizer from its explicit source pins."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--resolved-identity", type=Path, required=True)
    parser.add_argument("--baseline-archive", type=Path, required=True)
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument(
        "--candidate-source-root",
        type=Path,
        help="Clean checkout whose source/config/matrix pins are bound by Stage 3.",
    )
    parser.add_argument(
        "--candidate-identity",
        "--stage3-candidate-identity",
        dest="candidate_identity",
        type=Path,
        help="Stage-3 candidate identity JSON (defaults inside the producer/candidate root).",
    )
    parser.add_argument(
        "--attribution-ledger",
        "--stage3-attribution-ledger",
        dest="attribution_ledger",
        type=Path,
        help="Stage-3 attribution ledger JSON (defaults inside the producer/candidate root).",
    )
    parser.add_argument(
        "--comparison-report",
        "--stage3-comparison-report",
        dest="comparison_report",
        type=Path,
        help="Stage-3 comparison report JSON (defaults inside the producer/candidate root).",
    )
    parser.add_argument(
        "--comparison-findings",
        "--stage3-comparison-findings",
        dest="comparison_findings",
        type=Path,
        help="Stage-3 findings JSONL (defaults inside the producer/candidate root).",
    )
    parser.add_argument(
        "--pre-doi-producer",
        action="store_true",
        help="Bind a real DOI identity only in a derivative of an accepted scientific candidate",
    )
    args = parser.parse_args()
    finalizer = finalize_pre_doi_candidate if args.pre_doi_producer else finalize
    receipt = finalizer(
        producer_root=args.producer_root,
        candidate_root=args.candidate_root,
        resolved_identity=args.resolved_identity,
        baseline_archive=args.baseline_archive,
        expected_source_sha=args.expected_source_sha,
        candidate_source_root=args.candidate_source_root,
        candidate_identity=args.candidate_identity,
        attribution_ledger=args.attribution_ledger,
        comparison_report=args.comparison_report,
        comparison_findings=args.comparison_findings,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
