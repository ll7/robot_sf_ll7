#!/usr/bin/env python3
"""Run a benchmark release workflow on top of the camera-ready campaign stack.

Exit codes follow the wrapped campaign semantics for non-success benchmark outcomes:
- 0: benchmark-success release
- 2: unexpected failure or missing required release artifacts
- 3: accepted-unavailable-only campaign outcome (non-success, fail-closed)
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from robot_sf.adversarial.public_projection import find_offending_paths
from robot_sf.benchmark.artifact_publication import (
    PublicationPreflightError,
    export_publication_bundle,
    verify_publication_bundle_preflight,
)
from robot_sf.benchmark.camera_ready_campaign import (
    load_campaign_config,
    prepare_campaign_preflight,
    run_campaign,
    write_campaign_report,
)
from robot_sf.benchmark.checkpoint_staging_receipt import (
    CheckpointStagingReceiptError,
    validate_checkpoint_staging_receipt,
)
from robot_sf.benchmark.identity.hash_utils import sha256_file
from robot_sf.benchmark.orca_preflight import OrcaRvo2PreflightError, check_orca_rvo2_preflight
from robot_sf.benchmark.release_acceptance import (
    validate_diagnostic_stress_smoke_acceptance,
    validate_full_benchmark_release_acceptance,
)
from robot_sf.benchmark.release_protocol import (
    HISTORICAL_ZENODO_CONCEPT_DOIS,
    build_release_provenance,
    build_resolved_release_manifest,
    is_diagnostic_stress_smoke,
    load_release_manifest,
    parse_release_args,
    resolve_campaign_artifact_path,
    validate_release_manifest,
    validate_release_planner_roster,
    validate_stress_smoke_runtime_identity,
)
from robot_sf.benchmark.release_resume_admission import (
    ReleaseResumeAdmissionError,
    campaign_has_prior_execution,
    validate_release_resume_admission,
)
from robot_sf.benchmark.runtime_smoke_admission import (
    RuntimeSmokeAdmissionError,
    validate_runtime_smoke_result,
)
from robot_sf.common.artifact_paths import get_artifact_category_path, get_repository_root

if TYPE_CHECKING:
    from collections.abc import Sequence


HISTORICAL_RELEASE_IDENTITY_TOKENS = frozenset({"0.0.3.post1", *HISTORICAL_ZENODO_CONCEPT_DOIS})
_TEXT_ARTIFACT_SUFFIXES = frozenset(
    {".cff", ".csv", ".html", ".json", ".jsonl", ".md", ".tex", ".tsv", ".txt", ".yaml", ".yml"}
)
_CAMPAIGN_LOCAL_PATH_FIELDS = frozenset(
    {
        "campaign_root",
        "summary_json",
        "table_csv",
        "table_md",
        "report_md",
        "snqi_diagnostics_json",
        "snqi_diagnostics_md",
        "snqi_sensitivity_csv",
        "assurance_fragment_json",
        "assurance_fragment_md",
        "assurance_fragment_svg",
        "matrix_summary_json",
        "matrix_summary_csv",
        "seed_variability_json",
        "seed_variability_csv",
        "seed_episode_rows_csv",
        "statistical_sufficiency_json",
        "actuation_envelope_json",
        "actuation_envelope_md",
        "publication_bundle",
    }
)
_CAMPAIGN_PUBLIC_RESULT_FIELDS = frozenset(
    {
        "campaign_id",
        "total_runs",
        "successful_runs",
        "non_success_runs",
        "accepted_unavailable_runs",
        "unexpected_failed_runs",
        "campaign_execution_status",
        "evidence_status",
        "row_status_summary",
        "benchmark_success",
        "status",
        "status_reason",
        "exit_code",
        "benchmark_success_basis",
        "core_successful_runs",
        "core_total_runs",
        "total_episodes",
        "runtime_sec",
        "campaign_integrity",
        "warnings",
        "soft_contract_warning",
    }
)
_PUBLIC_RELEASE_ACCEPTANCE_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "benchmark_success",
        "claim_boundary",
        "expected_planner_arms",
        "successful_planner_arms",
        "expected_scenario_count",
        "expected_seed_count",
        "expected_episode_cells",
        "observed_episode_rows",
        "unique_episode_identities",
        "missing_episode_identities",
        "unexpected_episode_identities",
        "source_commits",
        "forbidden_status_counts",
        "blockers",
    }
)


class ReleaseArtifactIdentityError(ValueError):
    """Raised when a campaign artifact still carries a predecessor identity."""


class ReleaseResultPrivacyError(ValueError):
    """Raised when a release result could expose machine-local filesystem state."""


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON object to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _public_campaign_result(run_payload: dict[str, Any]) -> dict[str, Any]:
    """Project a campaign runner result onto its public, path-free release fields."""
    unknown = set(run_payload) - _CAMPAIGN_LOCAL_PATH_FIELDS - _CAMPAIGN_PUBLIC_RESULT_FIELDS
    if unknown:
        raise ReleaseResultPrivacyError("campaign runner returned unsupported result fields")
    projected = {
        key: run_payload[key] for key in _CAMPAIGN_PUBLIC_RESULT_FIELDS if key in run_payload
    }
    if find_offending_paths(projected):
        raise ReleaseResultPrivacyError("campaign runner result contains private filesystem data")
    return projected


def _print_public_result_file(path: Path) -> None:
    """Emit the persisted public result after a second private-path check."""
    payload = _read_json(path)
    if find_offending_paths(payload):
        raise ReleaseResultPrivacyError("persisted release result contains private filesystem data")
    print(json.dumps(payload, indent=2))


def _public_release_acceptance(acceptance: dict[str, Any]) -> dict[str, Any]:
    """Project validator output onto its path-free publication contract."""
    unknown = set(acceptance) - _PUBLIC_RELEASE_ACCEPTANCE_FIELDS
    projected = {
        key: acceptance[key] for key in _PUBLIC_RELEASE_ACCEPTANCE_FIELDS if key in acceptance
    }
    if unknown or find_offending_paths(projected):
        return {
            "schema_version": str(acceptance.get("schema_version", "unknown")),
            "status": "invalid",
            "benchmark_success": False,
            "claim_boundary": "release acceptance diagnostics must be public and path-free",
            "blockers": ["release acceptance diagnostics contained non-public fields"],
        }
    return projected


def _campaign_summary_path(campaign_root: Path) -> Path:
    """Resolve the campaign summary before any release read or merge write."""
    return resolve_campaign_artifact_path(campaign_root, "reports/campaign_summary.json")


def _repo_relative(path: Path) -> str:
    """Return a repository-relative path string when possible."""
    resolved = path.resolve()
    repo_root = get_repository_root().resolve()
    try:
        return resolved.relative_to(repo_root).as_posix()
    except ValueError:
        return f"<external>/{resolved.name}"


def _required_repo_relative(path: Path) -> str:
    """Return a repository-relative release input path or fail without exposing its location."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(get_repository_root().resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("release input must be inside the repository worktree") from exc


def _public_release_invocation(manifest_argument: str, mode: str) -> str:
    """Return a reproducible public entrypoint without machine-local launch paths.

    The complete scheduler invocation remains in private campaign provenance.  The
    publication record intentionally omits operational arguments such as output
    roots and receipt locations because those values identify local filesystems.
    """
    manifest_path = _repo_relative(Path(manifest_argument))
    return shlex.join(
        [
            "python",
            "scripts/tools/run_benchmark_release.py",
            "--manifest",
            manifest_path,
            "--mode",
            mode,
        ]
    )


def _current_source_commit() -> str:
    """Return the exact checked-out source commit or fail closed."""
    completed = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD^{commit}"],
        cwd=get_repository_root(),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    commit = completed.stdout.strip().lower()
    if (
        completed.returncode != 0
        or len(commit) != 40
        or any(c not in "0123456789abcdef" for c in commit)
    ):
        raise ReleaseResumeAdmissionError("unable to resolve exact release source commit")
    return commit


def _current_worktree_clean() -> bool:
    """Return whether the release checkout has no tracked or untracked changes."""
    completed = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=get_repository_root(),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if completed.returncode != 0:
        raise ReleaseResumeAdmissionError("unable to inspect release source worktree state")
    return not bool(completed.stdout.strip())


def _private_stress_launch() -> bool:
    """Identify private/SLURM execution without treating local tests as cluster launches."""
    return bool(
        os.environ.get("SLURM_JOB_ID")
        or "SLURM_EXPECTED_PUBLIC_COMMIT" in os.environ
        or os.environ.get("RELEASE_PRIVATE_LAUNCH")
    )


def _fixed_campaign_root(*, output_root: Path | None, campaign_id: str) -> Path:
    """Resolve a fixed campaign directory without allowing path traversal."""
    base = (
        output_root.resolve()
        if output_root is not None
        else (get_artifact_category_path("benchmarks") / "camera_ready").resolve()
    )
    candidate = (base / campaign_id).resolve()
    if not candidate.is_relative_to(base):
        raise ReleaseResumeAdmissionError("campaign_id resolves outside the campaign output root")
    return candidate


def _admit_release_resume(
    *,
    args: Any,
    cfg: Any,
    campaign_config_path: Path,
    checkpoint_receipt_path: Path,
) -> dict[str, Any] | None:
    """Admit a fresh run or validate an infrastructure-only same-ID resume.

    Returns:
        Sanitized resume-receipt metadata, or ``None`` for a fresh campaign.
    """
    if args.campaign_id is None:
        if args.resume_receipt is not None:
            raise ReleaseResumeAdmissionError(
                "resume receipt requires an explicit fixed campaign_id"
            )
        return None
    campaign_root = _fixed_campaign_root(
        output_root=args.output_root,
        campaign_id=args.campaign_id,
    )
    prior_execution = campaign_has_prior_execution(campaign_root)
    if not prior_execution and args.resume_receipt is None:
        return None
    receipt = validate_release_resume_admission(
        campaign_root=campaign_root,
        campaign_id=args.campaign_id,
        campaign_config_path=campaign_config_path,
        checkpoint_receipt_path=checkpoint_receipt_path,
        current_source_commit=_current_source_commit(),
        resume_enabled=bool(cfg.resume),
        resume_receipt_path=args.resume_receipt,
        max_age_hours=args.resume_receipt_max_age_hours,
    )
    if receipt is None:
        return None
    receipt["path"] = _required_repo_relative(args.resume_receipt)
    return receipt


def _merge_release_provenance(campaign_root: Path, release_provenance: dict[str, Any]) -> None:
    """Inject release provenance into campaign artifacts and refresh the markdown report."""
    # Campaign summary JSON and its human-readable markdown report.
    summary_path = _campaign_summary_path(campaign_root)
    report_md_path = campaign_root / "reports" / "campaign_report.md"
    # Campaign and benchmark manifests that describe the run contract.
    manifest_path = campaign_root / "campaign_manifest.json"
    benchmark_manifest_path = campaign_root / "manifest.json"
    # Run metadata consumed by downstream automation and provenance checks.
    run_meta_path = campaign_root / "run_meta.json"

    summary = _read_json(summary_path)
    summary["benchmark_release"] = dict(release_provenance)
    campaign_block = summary.get("campaign")
    if not isinstance(campaign_block, dict):
        campaign_block = {}
        summary["campaign"] = campaign_block
    # Inject release identity and manifest pointers into the campaign metadata.
    campaign_block.update(
        {
            "benchmark_protocol_version": release_provenance["benchmark_protocol_version"],
            "benchmark_release_id": release_provenance["release_id"],
            "benchmark_release_tag": release_provenance["release_tag"],
            "benchmark_release_manifest_path": release_provenance["manifest_path"],
            "benchmark_release_manifest_sha256": release_provenance["manifest_sha256"],
            "canonical_release_config": release_provenance["canonical_campaign_config"],
            "release_tag": release_provenance["release_tag"],
            "doi": release_provenance.get("doi", "10.5281/zenodo.<record-id>"),
            "doi_url": (
                f"https://doi.org/{release_provenance.get('doi', '10.5281/zenodo.<record-id>')}"
            ),
            "release_url": (
                f"{release_provenance.get('repository_url', 'https://github.com/ll7/robot_sf_ll7').rstrip('/')}/releases/"
                f"tag/{release_provenance['release_tag']}"
            ),
            "release_asset_url": (
                f"{release_provenance.get('repository_url', 'https://github.com/ll7/robot_sf_ll7').rstrip('/')}/releases/download/"
                f"{release_provenance['release_tag']}/{campaign_root.name}_publication_bundle.tar.gz"
            ),
        }
    )
    _write_json(summary_path, summary)
    write_campaign_report(report_md_path, summary)

    # Stamp every provenance-facing artifact with the benchmark_release payload.
    for path in (manifest_path, benchmark_manifest_path, run_meta_path):
        payload = _read_json(path)
        payload["benchmark_release"] = dict(release_provenance)
        _write_json(path, payload)


def _assert_no_historical_release_identity(campaign_root: Path) -> None:
    """Reject text artifacts that retain a superseded release tag or DOI.

    The campaign runner writes release metadata before this guard runs.  A
    stale fixed-campaign directory can still contain predecessor fields in
    files that the runner does not own, however.  Rejecting that tree keeps
    the publication bundle internally authoritative instead of silently
    mixing two release identities.
    """
    offenders: list[str] = []
    token_bytes = {token: token.encode("ascii") for token in HISTORICAL_RELEASE_IDENTITY_TOKENS}
    overlap = max(len(value) for value in token_bytes.values()) - 1
    for candidate in sorted(campaign_root.rglob("*")):
        if (
            not candidate.is_file()
            or candidate.is_symlink()
            or candidate.suffix.lower() not in _TEXT_ARTIFACT_SUFFIXES
        ):
            continue
        try:
            with candidate.open("rb") as handle:
                matched: set[str] = set()
                tail = b""
                while chunk := handle.read(1024 * 1024):
                    searchable = tail + chunk
                    for token, token_value in token_bytes.items():
                        if token_value in searchable:
                            matched.add(token)
                    if matched == HISTORICAL_RELEASE_IDENTITY_TOKENS:
                        break
                    tail = searchable[-overlap:]
        except OSError as exc:
            raise ReleaseArtifactIdentityError(
                "unable to inspect campaign artifact for release identity"
            ) from exc
        if matched:
            relative = candidate.relative_to(campaign_root).as_posix()
            offenders.append(f"{relative} ({', '.join(sorted(matched))})")
    if offenders:
        raise ReleaseArtifactIdentityError(
            "campaign artifacts retain a historical release identity; refusing release: "
            + "; ".join(offenders[:8])
        )


def _required_artifacts_missing(campaign_root: Path, required_paths: tuple[str, ...]) -> list[str]:
    """Return missing or unsafe required artifact paths from the campaign root."""
    missing: list[str] = []
    for relative_path in required_paths:
        try:
            resolve_campaign_artifact_path(campaign_root, relative_path)
        except (OSError, ValueError):
            missing.append(relative_path)
    return missing


def _build_publication_payload(
    *,
    campaign_root: Path,
    release_tag: str,
    doi: str,
    repository_url: str,
) -> dict[str, Any]:
    """Export a benchmark publication bundle and return a JSON-safe payload."""
    result = export_publication_bundle(
        campaign_root,
        get_artifact_category_path("benchmarks") / "publication",
        bundle_name=f"{campaign_root.name}_publication_bundle",
        include_videos=False,
        repository_url=repository_url,
        release_tag=release_tag,
        doi=doi,
        overwrite=True,
    )
    return {
        "bundle_dir": _repo_relative(result.bundle_dir),
        "archive_path": _repo_relative(result.archive_path),
        "manifest_path": _repo_relative(result.manifest_path),
        "checksums_path": _repo_relative(result.checksums_path),
        "file_count": result.file_count,
        "total_bytes": result.total_bytes,
    }


def _run_publication_preflight(bundle_dir: Path) -> None:
    """Run the final publication preflight over an exported bundle directory.

    A missing bundle directory is treated as "not exported here" and is skipped
    rather than failing (the export step owns the hard failure when it runs).

    Raises:
        PublicationPreflightError: If a built publication bundle is internally
            self-inconsistent (issue #5530). The release must not be treated as
            release-valid when this fails.
    """
    resolved = Path(bundle_dir).resolve()
    if not resolved.is_dir():
        return
    verify_publication_bundle_preflight(resolved)


def _record_publication_payload(campaign_root: Path, publication_payload: dict[str, Any]) -> None:
    """Record the exported bundle descriptor in the campaign summary and report."""
    summary_path = _campaign_summary_path(campaign_root)
    summary = _read_json(summary_path)
    summary["publication_bundle"] = publication_payload
    _write_json(summary_path, summary)
    write_campaign_report(campaign_root / "reports" / "campaign_report.md", summary)


def _record_release_acceptance(campaign_root: Path, acceptance: dict[str, Any]) -> None:
    """Persist the full-release gate beside the campaign summary and report."""
    summary_path = _campaign_summary_path(campaign_root)
    summary = _read_json(summary_path)
    summary["full_release_acceptance"] = acceptance
    _write_json(summary_path, summary)
    write_campaign_report(campaign_root / "reports" / "campaign_report.md", summary)


def _record_diagnostic_stress_smoke_acceptance(
    campaign_root: Path, acceptance: dict[str, Any]
) -> None:
    """Persist diagnostic stress admission without using the full-release field."""
    summary_path = _campaign_summary_path(campaign_root)
    summary = _read_json(summary_path)
    summary["diagnostic_stress_smoke_acceptance"] = acceptance
    _write_json(summary_path, summary)
    write_campaign_report(campaign_root / "reports" / "campaign_report.md", summary)


def _print_publication_identity_rejection() -> None:
    """Emit the fail-closed status without logging campaign-derived values."""
    print(
        json.dumps(
            {
                "mode": "run",
                "publication_bundle": None,
                "publication_preflight_status": "fail",
                "release_benchmark_success": False,
                "release_status": "publication_identity_rejected",
                "release_status_reason": (
                    "publication bundle retained a historical release identity"
                ),
                "release_exit_code": 2,
            },
            indent=2,
        )
    )


def _normalize_repository_input(path: Path, *, field_name: str) -> Path:
    """Resolve one release input from the repository root and reject external paths."""
    repository_root = get_repository_root().resolve()
    candidate = path if path.is_absolute() else repository_root / path
    resolved = candidate.resolve()
    try:
        resolved.relative_to(repository_root)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be inside the repository worktree") from exc
    return resolved


def _checkpoint_arm_identities(
    payload: dict[str, Any], *, label: str
) -> list[tuple[str, str, str, str, bool, str]]:
    """Return the path-independent checkpoint identity for every validated receipt arm."""
    arms = payload.get("arms")
    if not isinstance(arms, list) or not arms:
        raise ValueError(f"{label} checkpoint receipt has no arms")

    identities: list[tuple[str, str, str, str, bool, str]] = []
    for index, arm in enumerate(arms):
        if not isinstance(arm, dict):
            raise ValueError(f"{label} checkpoint receipt arm {index} is not an object")
        fields = tuple(arm.get(field) for field in ("planner_key", "algo", "kind", "value"))
        if not all(isinstance(field, str) and field for field in fields):
            raise ValueError(f"{label} checkpoint receipt arm {index} has incomplete identity")
        implicit = arm.get("implicit")
        if not isinstance(implicit, bool):
            raise ValueError(f"{label} checkpoint receipt arm {index} has invalid implicit flag")
        checkpoint_sha256 = arm.get("checkpoint_sha256")
        if not isinstance(checkpoint_sha256, str):
            raise ValueError(f"{label} checkpoint receipt arm {index} has no checkpoint hash")
        checkpoint_sha256 = checkpoint_sha256.lower()
        if len(checkpoint_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in checkpoint_sha256
        ):
            raise ValueError(
                f"{label} checkpoint receipt arm {index} has an invalid checkpoint hash"
            )
        identities.append((*fields, implicit, checkpoint_sha256))

    if len(set(identities)) != len(identities):
        raise ValueError(f"{label} checkpoint receipt contains duplicate arm identities")
    return sorted(identities)


def _load_runtime_smoke_checkpoint_receipt(runtime_smoke_receipt: Path) -> dict[str, Any]:
    """Load and verify the checkpoint receipt embedded in a runtime-smoke result."""
    try:
        result = _read_json(runtime_smoke_receipt)
        descriptor = result.get("checkpoint_staging_receipt")
        if not isinstance(descriptor, dict):
            raise ValueError("runtime smoke checkpoint staging receipt descriptor is missing")
        raw_path = descriptor.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError("runtime smoke checkpoint staging receipt path is missing")
        checkpoint_relative = Path(raw_path)
        if checkpoint_relative.is_absolute():
            raise ValueError("runtime smoke checkpoint staging receipt path is absolute")
        checkpoint_path = _normalize_repository_input(
            checkpoint_relative,
            field_name="runtime smoke checkpoint staging receipt",
        )
        if not checkpoint_path.is_file():
            raise ValueError("runtime smoke checkpoint staging receipt is not a file")
        declared_sha256 = descriptor.get("sha256")
        if not isinstance(declared_sha256, str) or declared_sha256.lower() != sha256_file(
            checkpoint_path
        ):
            raise ValueError("runtime smoke checkpoint staging receipt hash is invalid")
        return _read_json(checkpoint_path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            "runtime smoke checkpoint receipt could not be loaded or verified"
        ) from exc


def _compare_rehearsal_checkpoint_identities(
    checkpoint_receipt: dict[str, Any],
    runtime_smoke_receipt: Path,
    *,
    release_receipt_sha256: str,
    runtime_smoke_receipt_sha256: str,
) -> tuple[dict[str, Any], bool]:
    """Compare shared checkpoint identities across config-bound rehearsal receipts."""
    try:
        runtime_checkpoint_receipt = _load_runtime_smoke_checkpoint_receipt(runtime_smoke_receipt)
        staged_identities = _checkpoint_arm_identities(checkpoint_receipt, label="staged")
        runtime_identities = _checkpoint_arm_identities(
            runtime_checkpoint_receipt,
            label="runtime smoke",
        )
    except (OSError, TypeError, ValueError, KeyError):
        return (
            {
                "schema_version": "benchmark-release-rehearsal-checkpoint-identity-admission.v1",
                "status": "rejected",
                "release_receipt_sha256": release_receipt_sha256,
                "runtime_smoke_receipt_sha256": runtime_smoke_receipt_sha256,
                "blockers": ["checkpoint arm identity could not be compared"],
            },
            False,
        )

    identities_match = staged_identities == runtime_identities
    identity_admission = {
        "schema_version": "benchmark-release-rehearsal-checkpoint-identity-admission.v1",
        "status": "admitted" if identities_match else "rejected",
        "release_receipt_sha256": release_receipt_sha256,
        "runtime_smoke_receipt_sha256": runtime_smoke_receipt_sha256,
        "arm_count": len(staged_identities),
        "identity_fields": [
            "planner_key",
            "algo",
            "kind",
            "value",
            "implicit",
            "checkpoint_sha256",
        ],
        "blockers": []
        if identities_match
        else ["staged and runtime-smoke checkpoint arm identities do not match"],
    }
    return identity_admission, identities_match


def _normalize_rehearsal_args(args: Any) -> None:
    """Normalize all read-only rehearsal inputs before any admission is attempted."""
    args.manifest = _normalize_repository_input(args.manifest, field_name="manifest")
    if args.checkpoint_receipt is not None:
        args.checkpoint_receipt = _normalize_repository_input(
            args.checkpoint_receipt,
            field_name="checkpoint receipt",
        )
    if args.runtime_smoke_receipt is not None:
        args.runtime_smoke_receipt = _normalize_repository_input(
            args.runtime_smoke_receipt,
            field_name="runtime smoke receipt",
        )


def _rehearsal_failure(
    *,
    status: str,
    reason: str,
    evidence: dict[str, Any] | None = None,
) -> int:
    """Emit a path-free fail-closed rehearsal result."""
    result: dict[str, Any] = {
        "mode": "rehearsal",
        "status": status,
        "status_reason": reason,
        "benchmark_success": False,
        "release_benchmark_success": False,
        "campaign_execution_status": "not_started",
        "campaign_output_created": False,
        "publication_requested": False,
        "scheduler_requested": False,
        "evidence_status": "blocked",
        "release_exit_code": 2,
    }
    if evidence:
        result.update(evidence)
    if find_offending_paths(result):
        result["status_reason"] = "release rehearsal admission failed without public path details"
        for key in tuple(result):
            if key not in {
                "mode",
                "status",
                "status_reason",
                "benchmark_success",
                "release_benchmark_success",
                "campaign_execution_status",
                "campaign_output_created",
                "publication_requested",
                "scheduler_requested",
                "evidence_status",
                "release_exit_code",
            }:
                result.pop(key)
    print(json.dumps(result, indent=2))
    return 2


def _run_release_rehearsal(args: Any) -> int:  # noqa: C901, PLR0912
    """Run all release admissions and stop before campaign execution or output creation."""
    unsupported = {
        "output_root": args.output_root,
        "label": args.label,
        "campaign_id": args.campaign_id,
        "resume_receipt": args.resume_receipt,
    }
    supplied = sorted(name.replace("_", "-") for name, value in unsupported.items() if value)
    if supplied:
        return _rehearsal_failure(
            status="unsupported_combination",
            reason=(
                "rehearsal mode does not accept campaign allocation or resume options: "
                + ", ".join(supplied)
            ),
        )

    try:
        manifest = load_release_manifest(args.manifest)
    except (OSError, TypeError, ValueError, KeyError) as exc:
        return _rehearsal_failure(
            status="manifest_rejected",
            reason="release manifest could not be admitted: " + str(exc),
        )
    if getattr(manifest, "schema_version", None) != "benchmark-release-manifest.v0.2":
        return _rehearsal_failure(
            status="unsupported_manifest",
            reason="rehearsal mode requires benchmark-release-manifest.v0.2",
        )

    try:
        cfg = load_campaign_config(manifest.canonical_campaign_config_path)
        source_commit = _current_source_commit()
        worktree_clean = _current_worktree_clean()
    except (OSError, TypeError, ValueError, KeyError, ReleaseResumeAdmissionError) as exc:
        return _rehearsal_failure(
            status="startup_admission_failed",
            reason="release rehearsal startup admission failed: " + str(exc),
        )

    startup_admission = {
        "schema_version": "benchmark-release-rehearsal-startup-admission.v1",
        "status": "valid",
        "source_commit": source_commit,
        "worktree_clean": worktree_clean,
        "blockers": [],
    }
    if not worktree_clean:
        startup_admission["status"] = "invalid"
        startup_admission["blockers"] = ["release source worktree is not clean"]
    if source_commit != getattr(manifest, "source_sha", None):
        startup_admission["status"] = "invalid"
        startup_admission["blockers"] = [
            "checked-out source SHA does not match manifest source_sha"
        ]
    if startup_admission["status"] != "valid":
        return _rehearsal_failure(
            status="startup_admission_failed",
            reason=str(startup_admission["blockers"][0]),
            evidence={"startup_admission": startup_admission},
        )

    try:
        check_orca_rvo2_preflight(cfg)
    except OrcaRvo2PreflightError as exc:
        return _rehearsal_failure(
            status="startup_admission_failed",
            reason="ORCA runtime preflight failed: " + str(exc),
            evidence={"startup_admission": {**startup_admission, "status": "invalid"}},
        )

    try:
        manifest_validation = validate_release_manifest(manifest, campaign_config=cfg)
        planner_admission = validate_release_planner_roster(manifest, cfg)
        evidence: dict[str, Any] = {
            "startup_admission": startup_admission,
            "manifest_validation": manifest_validation,
            "planner_roster_admission": planner_admission,
            "release_inputs": {
                "source_commit": source_commit,
                "manifest_path": _required_repo_relative(manifest.path),
                "manifest_sha256": sha256_file(manifest.path),
                "canonical_campaign_config": _required_repo_relative(
                    manifest.canonical_campaign_config_path
                ),
                "canonical_campaign_config_sha256": sha256_file(
                    manifest.canonical_campaign_config_path
                ),
            },
        }
    except (OSError, TypeError, ValueError, KeyError) as exc:
        return _rehearsal_failure(
            status="manifest_rejected",
            reason="release manifest admission failed: " + str(exc),
            evidence={"startup_admission": startup_admission},
        )
    if manifest_validation.get("status") != "valid":
        return _rehearsal_failure(
            status="manifest_rejected",
            reason="release manifest validation failed",
            evidence=evidence,
        )
    if planner_admission["status"] != "valid":
        return _rehearsal_failure(
            status="planner_roster_rejected",
            reason=str(planner_admission["blockers"][0]),
            evidence=evidence,
        )

    if args.checkpoint_receipt is None:
        return _rehearsal_failure(
            status="checkpoint_receipt_missing",
            reason="rehearsal mode requires an enforced-staged checkpoint receipt",
            evidence=evidence,
        )
    if args.runtime_smoke_receipt is None:
        return _rehearsal_failure(
            status="runtime_smoke_receipt_missing",
            reason="rehearsal mode requires an exact-source runtime smoke receipt",
            evidence=evidence,
        )

    try:
        checkpoint_receipt = validate_checkpoint_staging_receipt(
            cfg,
            args.checkpoint_receipt,
            campaign_config_path=manifest.canonical_campaign_config_path,
            max_age_hours=args.checkpoint_receipt_max_age_hours,
            repo_root=get_repository_root(),
        )
        if checkpoint_receipt.get("submit_safe") is not True:
            raise CheckpointStagingReceiptError("checkpoint receipt is not submit-safe")
        evidence["checkpoint_staging_admission"] = {
            "schema_version": "benchmark-release-rehearsal-checkpoint-admission.v1",
            "status": "admitted",
            "path": _required_repo_relative(args.checkpoint_receipt),
            "sha256": sha256_file(args.checkpoint_receipt),
            "generated_at_utc": checkpoint_receipt.get("generated_at_utc"),
            "submit_safe": True,
            "arm_count": len(checkpoint_receipt.get("arms", [])),
        }
    except (OSError, TypeError, ValueError, KeyError, CheckpointStagingReceiptError) as exc:
        evidence["checkpoint_staging_admission"] = {
            "schema_version": "benchmark-release-rehearsal-checkpoint-admission.v1",
            "status": "rejected",
            "blockers": [str(exc)],
        }
        return _rehearsal_failure(
            status="checkpoint_receipt_rejected",
            reason="checkpoint staging admission failed: " + str(exc),
            evidence=evidence,
        )

    try:
        smoke_admission = validate_runtime_smoke_result(
            args.runtime_smoke_receipt,
            repo_root=get_repository_root(),
            expected_source_commit=source_commit,
            expected_planner_keys=tuple(manifest.planner_keys),
            max_age_hours=args.runtime_smoke_receipt_max_age_hours,
        )
    except (OSError, TypeError, ValueError, KeyError, RuntimeSmokeAdmissionError) as exc:
        evidence["runtime_smoke_admission"] = {
            "schema_version": "benchmark-release-rehearsal-runtime-smoke-admission.v1",
            "status": "rejected",
            "blockers": [str(exc)],
        }
        return _rehearsal_failure(
            status="runtime_smoke_receipt_rejected",
            reason="runtime smoke admission failed: " + str(exc),
            evidence=evidence,
        )
    evidence["runtime_smoke_admission"] = {
        "schema_version": "benchmark-release-rehearsal-runtime-smoke-admission.v1",
        **smoke_admission,
    }
    release_receipt_sha256 = str(
        evidence["checkpoint_staging_admission"].get("sha256") or ""
    ).lower()
    runtime_smoke_receipt_sha256 = str(
        smoke_admission.get("checkpoint_receipt_sha256") or ""
    ).lower()
    identity_admission, identities_match = _compare_rehearsal_checkpoint_identities(
        checkpoint_receipt,
        args.runtime_smoke_receipt,
        release_receipt_sha256=release_receipt_sha256,
        runtime_smoke_receipt_sha256=runtime_smoke_receipt_sha256,
    )
    evidence["checkpoint_identity_admission"] = identity_admission
    if not identities_match:
        unable_to_compare = not identity_admission.get("arm_count")
        return _rehearsal_failure(
            status="checkpoint_identity_mismatch",
            reason=(
                "release and runtime-smoke checkpoint identities could not be compared"
                if unable_to_compare
                else "release and runtime-smoke checkpoint identities do not match"
            ),
            evidence=evidence,
        )

    try:
        resolved_manifest = build_resolved_release_manifest(
            manifest,
            campaign_config=cfg,
            source_commit=source_commit,
        )
    except (OSError, TypeError, ValueError, KeyError) as exc:
        return _rehearsal_failure(
            status="release_identity_rejected",
            reason="release identity binding failed: " + str(exc),
            evidence=evidence,
        )

    result = {
        "mode": "rehearsal",
        "status": "release_rehearsal_passed",
        "status_reason": (
            "release admissions passed; campaign execution, episode creation, publication, "
            "and scheduler submission were intentionally not started"
        ),
        "benchmark_success": False,
        "release_benchmark_success": False,
        "campaign_execution_status": "not_started",
        "campaign_output_created": False,
        "publication_requested": False,
        "scheduler_requested": False,
        "evidence_status": "preflight_valid",
        "release_exit_code": 0,
        "resolved_manifest": resolved_manifest,
        **evidence,
    }
    if find_offending_paths(result):
        return _rehearsal_failure(
            status="release_identity_rejected",
            reason="release rehearsal evidence contained private path data",
        )
    print(json.dumps(result, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901, PLR0912, PLR0915
    """Run the benchmark release entrypoint and return a POSIX exit code."""
    raw_argv = list(argv) if argv is not None else list(sys.argv[1:])
    args = parse_release_args(raw_argv)

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    if args.mode == "rehearsal":
        unsupported_options = (
            "--output-root",
            "--label",
            "--campaign-id",
            "--resume-receipt",
            "--resume-receipt-max-age-hours",
        )
        supplied_unsupported = sorted(
            option
            for option in unsupported_options
            if any(argument == option or argument.startswith(f"{option}=") for argument in raw_argv)
        )
        if supplied_unsupported:
            return _rehearsal_failure(
                status="unsupported_combination",
                reason=(
                    "rehearsal mode does not accept campaign allocation or resume options: "
                    + ", ".join(supplied_unsupported)
                ),
            )
        try:
            _normalize_rehearsal_args(args)
        except (OSError, TypeError, ValueError) as exc:
            return _rehearsal_failure(
                status="unsupported_input",
                reason="release rehearsal input normalization failed: " + str(exc),
            )
        return _run_release_rehearsal(args)

    invoked_command = shlex.join([sys.executable, str(Path(__file__)), *raw_argv])

    manifest = load_release_manifest(args.manifest)
    cfg = load_campaign_config(manifest.canonical_campaign_config_path)
    stress_smoke = is_diagnostic_stress_smoke(manifest)
    runtime_source_commit: str | None = None
    runtime_source_admission: dict[str, Any] = {
        "schema_version": "benchmark-stress-smoke-runtime-identity.v1",
        "status": "not_applicable",
        "runtime_source_commit": None,
        "blockers": [],
    }
    if stress_smoke:
        try:
            runtime_source_commit = _current_source_commit()
            private_launch = _private_stress_launch()
            # A stress result is evidence about the exact checked-out source even
            # when it is run locally.  Never allow local edits to become part of
            # an apparently immutable stress campaign.
            worktree_clean = _current_worktree_clean()
        except (ReleaseResumeAdmissionError, ValueError) as exc:
            runtime_source_admission = {
                "schema_version": "benchmark-stress-smoke-runtime-identity.v1",
                "status": "invalid",
                "runtime_source_commit": None,
                "blockers": [str(exc)],
            }
        else:
            runtime_source_admission = validate_stress_smoke_runtime_identity(
                manifest,
                current_source_commit=runtime_source_commit,
                launch_expected_source_commit=os.environ.get("SLURM_EXPECTED_PUBLIC_COMMIT"),
                require_launch_pin=private_launch,
                worktree_clean=worktree_clean,
                require_clean_worktree=True,
            )
        if runtime_source_admission["status"] != "valid":
            result = {
                "mode": args.mode,
                "status": "stress_smoke_source_rejected",
                "status_reason": str(runtime_source_admission["blockers"][0]),
                "benchmark_success": False,
                "release_benchmark_success": False,
                "diagnostic_success": False,
                "stress_smoke_runtime_identity": runtime_source_admission,
                "release_status": "stress_smoke_source_rejected",
                "release_status_reason": str(runtime_source_admission["blockers"][0]),
                "release_exit_code": 2,
            }
            print(json.dumps(result, indent=2))
            return 2
    elif getattr(manifest, "source_sha", None) is not None:
        # Future v0.2 manifests freeze the final immutable source before the
        # campaign starts.  Do not let a planning/base checkout produce
        # artifacts that claim the frozen source identity.
        try:
            runtime_source_commit = _current_source_commit()
        except (ReleaseResumeAdmissionError, ValueError) as exc:
            result = {
                "mode": args.mode,
                "status": "release_source_rejected",
                "status_reason": str(exc),
                "benchmark_success": False,
                "release_benchmark_success": False,
                "release_status": "release_source_rejected",
                "release_status_reason": str(exc),
                "release_exit_code": 2,
            }
            print(json.dumps(result, indent=2))
            return 2
        if runtime_source_commit != manifest.source_sha:
            reason = (
                "checked-out source SHA does not match manifest source_sha; "
                "planning/base SHA cannot satisfy final source identity"
            )
            result = {
                "mode": args.mode,
                "status": "release_source_rejected",
                "status_reason": reason,
                "benchmark_success": False,
                "release_benchmark_success": False,
                "release_status": "release_source_rejected",
                "release_status_reason": reason,
                "release_exit_code": 2,
            }
            print(json.dumps(result, indent=2))
            return 2
    try:
        check_orca_rvo2_preflight(cfg)
    except OrcaRvo2PreflightError as exc:
        reason = str(exc)
        result = {
            "mode": args.mode,
            "status": "orca_preflight_failed",
            "status_reason": reason,
            "benchmark_success": False,
            "exit_code": 2,
            "campaign_execution_status": "failed",
            "evidence_status": "blocked",
            "row_status_summary": {
                "successful_evidence_rows": 0,
                "accepted_unavailable_rows": 0,
                "unexpected_failed_rows": 0,
                "fallback_or_degraded_rows": 0,
            },
            "release_status": "orca_preflight_failed",
            "release_status_reason": reason,
            "release_benchmark_success": False,
            "release_exit_code": 2,
        }
        print(json.dumps(result, indent=2))
        return 2
    validation = validate_release_manifest(manifest, campaign_config=cfg)

    resolved_manifest_kwargs: dict[str, Any] = {"campaign_config": cfg}
    if runtime_source_commit is not None:
        resolved_manifest_kwargs["source_commit"] = runtime_source_commit
    resolved_manifest = build_resolved_release_manifest(manifest, **resolved_manifest_kwargs)
    if args.mode == "preflight":
        prepared = prepare_campaign_preflight(
            cfg,
            output_root=args.output_root,
            label=args.label,
            campaign_id=args.campaign_id,
            invoked_command=invoked_command,
        )
        preflight_payload = {
            "mode": "preflight",
            "manifest_validation": validation,
            "resolved_manifest": resolved_manifest,
            "campaign_id": prepared["campaign_id"],
            "campaign_root": str(prepared["campaign_root"]),
            "validate_config_path": str(prepared["validate_config_path"]),
            "preview_scenarios_path": str(prepared["preview_scenarios_path"]),
            "matrix_summary_json": str(prepared["matrix_summary_json_path"]),
            "matrix_summary_csv": str(prepared["matrix_summary_csv_path"]),
        }
        if stress_smoke:
            preflight_payload["runtime_source_commit"] = runtime_source_commit
            preflight_payload["stress_smoke_runtime_identity"] = runtime_source_admission
        print(json.dumps(preflight_payload, indent=2))
        return 0 if validation["status"] == "valid" else 2

    result = {
        "mode": "run",
        "manifest_validation": validation,
        "resolved_manifest": resolved_manifest,
    }
    if validation["status"] != "valid":
        result["benchmark_success"] = False
        result["status"] = "invalid_manifest"
        result["campaign_execution_status"] = "failed"
        result["evidence_status"] = "invalid"
        result["row_status_summary"] = {
            "successful_evidence_rows": 0,
            "accepted_unavailable_rows": 0,
            "unexpected_failed_rows": 0,
            "fallback_or_degraded_rows": 0,
        }
        print(json.dumps(result, indent=2))
        return 2

    if args.checkpoint_receipt is None:
        result.update(
            {
                "benchmark_success": False,
                "status": "checkpoint_receipt_missing",
                "status_reason": "run mode requires an enforced-staged checkpoint receipt",
                "campaign_execution_status": "not_started",
                "evidence_status": "blocked",
            }
        )
        print(json.dumps(result, indent=2))
        return 2
    try:
        checkpoint_receipt_path = _required_repo_relative(args.checkpoint_receipt)
    except ValueError as exc:
        result.update(
            {
                "benchmark_success": False,
                "status": "checkpoint_receipt_rejected",
                "status_reason": str(exc),
                "campaign_execution_status": "not_started",
                "evidence_status": "blocked",
            }
        )
        print(json.dumps(result, indent=2))
        return 2
    try:
        checkpoint_receipt = validate_checkpoint_staging_receipt(
            cfg,
            args.checkpoint_receipt,
            campaign_config_path=manifest.canonical_campaign_config_path,
            max_age_hours=args.checkpoint_receipt_max_age_hours,
        )
    except CheckpointStagingReceiptError as exc:
        result.update(
            {
                "benchmark_success": False,
                "status": "checkpoint_receipt_rejected",
                "status_reason": str(exc),
                "campaign_execution_status": "not_started",
                "evidence_status": "blocked",
            }
        )
        print(json.dumps(result, indent=2))
        return 2
    result["checkpoint_staging_receipt"] = {
        "path": checkpoint_receipt_path,
        "sha256": sha256_file(args.checkpoint_receipt),
        "generated_at_utc": checkpoint_receipt["generated_at_utc"],
        "submit_safe": True,
    }

    if getattr(manifest, "schema_version", None) == "benchmark-release-manifest.v0.2":
        smoke_result_path = getattr(args, "runtime_smoke_receipt", None)
        if smoke_result_path is None:
            result.update(
                {
                    "benchmark_success": False,
                    "status": "runtime_smoke_receipt_missing",
                    "status_reason": "v0.2 run mode requires exact-source runtime smoke evidence",
                    "campaign_execution_status": "not_started",
                    "evidence_status": "blocked",
                }
            )
            print(json.dumps(result, indent=2))
            return 2
        try:
            smoke_path = _required_repo_relative(smoke_result_path)
            smoke_receipt = validate_runtime_smoke_result(
                smoke_result_path,
                repo_root=get_repository_root(),
                expected_source_commit=_current_source_commit(),
                expected_planner_keys=tuple(manifest.planner_keys),
                max_age_hours=getattr(args, "runtime_smoke_receipt_max_age_hours", 24.0),
            )
        except (RuntimeSmokeAdmissionError, ValueError) as exc:
            result.update(
                {
                    "benchmark_success": False,
                    "status": "runtime_smoke_receipt_rejected",
                    "status_reason": str(exc),
                    "campaign_execution_status": "not_started",
                    "evidence_status": "blocked",
                }
            )
            print(json.dumps(result, indent=2))
            return 2
        result["runtime_smoke_receipt"] = {"path": smoke_path, **smoke_receipt}

    try:
        resume_receipt = _admit_release_resume(
            args=args,
            cfg=cfg,
            campaign_config_path=manifest.canonical_campaign_config_path,
            checkpoint_receipt_path=args.checkpoint_receipt,
        )
    except (ReleaseResumeAdmissionError, ValueError) as exc:
        result.update(
            {
                "benchmark_success": False,
                "status": "resume_admission_rejected",
                "status_reason": str(exc),
                "campaign_execution_status": "not_started",
                "evidence_status": "blocked",
            }
        )
        print(json.dumps(result, indent=2))
        return 2
    result["resume_admission"] = (
        resume_receipt
        if resume_receipt is not None
        else {"status": "fresh_campaign", "resume_same_campaign": False}
    )

    run_payload = run_campaign(
        cfg,
        output_root=args.output_root,
        label=args.label,
        campaign_id=args.campaign_id,
        skip_publication_bundle=True,
        invoked_command=invoked_command,
    )
    campaign_root = Path(str(run_payload["campaign_root"])).resolve()
    try:
        result.update(_public_campaign_result(run_payload))
    except ReleaseResultPrivacyError:
        result.update(
            {
                "benchmark_success": False,
                "status": "release_result_privacy_rejected",
                "status_reason": "campaign result could not be projected without private paths",
                "campaign_execution_status": "completed",
                "evidence_status": "blocked",
                "release_status": "release_result_privacy_rejected",
                "release_status_reason": (
                    "campaign result could not be projected without private paths"
                ),
                "release_benchmark_success": False,
                "release_exit_code": 2,
            }
        )
        release_dir = campaign_root / "release"
        _write_json(release_dir / "release_result.json", result)
        print(json.dumps(result, indent=2))
        return 2

    post_manifest_validation: dict[str, Any] | None = None
    if stress_smoke:
        # Re-hash every stress input after execution.  A clean HEAD alone does
        # not protect a long campaign from an asset being replaced in-place.
        post_manifest_validation = validate_release_manifest(manifest, campaign_config=cfg)
        result["post_run_manifest_validation"] = post_manifest_validation

    post_runtime_source_admission = runtime_source_admission
    if stress_smoke:
        try:
            post_runtime_source_commit = _current_source_commit()
            private_launch = _private_stress_launch()
            post_worktree_clean = _current_worktree_clean()
            post_runtime_source_admission = validate_stress_smoke_runtime_identity(
                manifest,
                current_source_commit=post_runtime_source_commit,
                launch_expected_source_commit=os.environ.get("SLURM_EXPECTED_PUBLIC_COMMIT"),
                require_launch_pin=private_launch,
                worktree_clean=post_worktree_clean,
                require_clean_worktree=True,
            )
            if post_runtime_source_commit != runtime_source_commit:
                post_runtime_source_admission["status"] = "invalid"
                post_runtime_source_admission.setdefault("blockers", []).append(
                    "checked-out runtime source commit changed during campaign execution"
                )
        except (ReleaseResumeAdmissionError, ValueError) as exc:
            post_runtime_source_admission = {
                "schema_version": "benchmark-stress-smoke-runtime-identity.v1",
                "status": "invalid",
                "runtime_source_commit": None,
                "blockers": [str(exc)],
            }

    release_provenance_kwargs: dict[str, Any] = {
        "campaign_root": campaign_root,
        "invoked_command": _public_release_invocation(args.manifest, args.mode),
    }
    if runtime_source_commit is not None:
        release_provenance_kwargs["source_commit"] = runtime_source_commit
    release_provenance = build_release_provenance(
        manifest,
        **release_provenance_kwargs,
    )
    result["benchmark_release"] = release_provenance
    try:
        _merge_release_provenance(campaign_root, release_provenance)
        _assert_no_historical_release_identity(campaign_root)
    except ReleaseArtifactIdentityError as exc:
        reason = str(exc)
        result.update(
            {
                "benchmark_success": False,
                "status": "release_identity_rejected",
                "status_reason": reason,
                "campaign_execution_status": str(
                    run_payload.get("campaign_execution_status", "completed")
                ),
                "evidence_status": "blocked",
                "release_status": "release_identity_rejected",
                "release_status_reason": reason,
                "release_benchmark_success": False,
                "release_exit_code": 2,
            }
        )
        _write_json(campaign_root / "release" / "release_result.json", result)
        print(json.dumps(result, indent=2))
        return 2

    missing = _required_artifacts_missing(campaign_root, manifest.required_artifact_paths)
    result["required_artifact_paths"] = list(manifest.required_artifact_paths)
    result["missing_required_artifacts"] = missing

    if stress_smoke:
        diagnostic_acceptance = validate_diagnostic_stress_smoke_acceptance(
            campaign_root,
            manifest=manifest,
            campaign_config=cfg,
            expected_source_commit=runtime_source_commit or "",
        )
        if post_runtime_source_admission["status"] != "valid":
            diagnostic_acceptance["status"] = "invalid"
            diagnostic_acceptance["diagnostic_success"] = False
            diagnostic_acceptance.setdefault("blockers", []).extend(
                f"runtime identity: {blocker}"
                for blocker in post_runtime_source_admission.get("blockers", [])
            )
        if missing:
            diagnostic_acceptance["status"] = "invalid"
            diagnostic_acceptance["diagnostic_success"] = False
            diagnostic_acceptance.setdefault("blockers", []).append(
                "required campaign artifacts are missing: " + ", ".join(missing)
            )
        if post_manifest_validation is not None and post_manifest_validation["status"] != "valid":
            diagnostic_acceptance["status"] = "invalid"
            diagnostic_acceptance["diagnostic_success"] = False
            for problem in post_manifest_validation.get("problems", []):
                diagnostic_acceptance.setdefault("blockers", []).append(
                    f"post-run manifest validation: {problem}"
                )
        _record_diagnostic_stress_smoke_acceptance(campaign_root, diagnostic_acceptance)
        result["stress_smoke_runtime_identity"] = post_runtime_source_admission
        result["diagnostic_stress_smoke_acceptance"] = diagnostic_acceptance
        result["campaign_benchmark_success"] = bool(run_payload.get("benchmark_success"))
        result["benchmark_success"] = False
        result["diagnostic_success"] = bool(diagnostic_acceptance.get("diagnostic_success"))
        result["release_benchmark_success"] = False
        result["publication_requested"] = False
        result["publication_bundle"] = None
        result["publication_preflight_status"] = "not_requested"
        result["release_status"] = (
            "diagnostic_stress_smoke_passed"
            if result["diagnostic_success"]
            else "diagnostic_stress_smoke_failed"
        )
        result["release_status_reason"] = (
            "diagnostic stress smoke passed exact-source, matrix, and fail-closed admission; "
            "this is not benchmark-release success"
            if result["diagnostic_success"]
            else str(
                (diagnostic_acceptance.get("blockers") or ["diagnostic stress smoke failed"])[0]
            )
        )
        result["campaign_status"] = run_payload.get("status")
        result["campaign_status_reason"] = run_payload.get("status_reason")
        run_exit_code = int(run_payload.get("exit_code", 2))
        result["release_exit_code"] = 0 if result["diagnostic_success"] else (run_exit_code or 2)
        result["status"] = result["release_status"]
        result["status_reason"] = result["release_status_reason"]
        result["exit_code"] = result["release_exit_code"]
        release_dir = campaign_root / "release"
        _write_json(release_dir / "release_manifest.resolved.json", resolved_manifest)
        _write_json(release_dir / "release_result.json", result)
        print(json.dumps(result, indent=2))
        return int(result["release_exit_code"])

    release_acceptance = _public_release_acceptance(
        validate_full_benchmark_release_acceptance(
            campaign_root,
            manifest=manifest,
            campaign_config=cfg,
        )
    )
    _record_release_acceptance(campaign_root, release_acceptance)
    result["release_acceptance"] = release_acceptance
    full_release_acceptance_failed = (
        getattr(manifest, "schema_version", None) == "benchmark-release-manifest.v0.2"
        and release_acceptance["status"] != "valid"
    )
    if getattr(manifest, "schema_version", None) == "benchmark-release-manifest.v0.2":
        # Keep the campaign's permissive core-success counters available in the
        # nested payload, but never expose them as full-release success.
        result["campaign_benchmark_success"] = bool(run_payload.get("benchmark_success"))
        result["benchmark_success"] = bool(
            run_payload.get("benchmark_success") and not full_release_acceptance_failed
        )

    result["benchmark_release"] = release_provenance

    release_dir = campaign_root / "release"
    _write_json(release_dir / "release_manifest.resolved.json", resolved_manifest)

    release_benchmark_success = (
        bool(run_payload.get("benchmark_success"))
        and not missing
        and not full_release_acceptance_failed
    )
    result["release_benchmark_success"] = release_benchmark_success
    publication_requested = bool(getattr(cfg, "export_publication_bundle", True))
    result["publication_requested"] = publication_requested
    if release_benchmark_success and publication_requested:
        publication_payload = _build_publication_payload(
            campaign_root=campaign_root,
            release_tag=manifest.release_tag,
            doi=manifest.doi,
            repository_url=manifest.repository_url,
        )
    else:
        result["publication_bundle"] = None
        result["publication_preflight_status"] = (
            "not_requested" if not publication_requested else None
        )

    result["release_status"] = (
        "missing_required_artifacts"
        if missing
        else "full_release_acceptance_failed"
        if full_release_acceptance_failed
        else (
            "ok"
            if release_benchmark_success
            else str(run_payload.get("status", "benchmark_failed"))
        )
    )
    result["release_status_reason"] = (
        "release artifacts validated and benchmark campaign was benchmark-success"
        if release_benchmark_success
        else (
            "release is missing required benchmark artifacts"
            if missing
            else (
                "full benchmark release acceptance failed: "
                + str((release_acceptance.get("blockers") or ["unspecified"])[0])
                if full_release_acceptance_failed
                else str(run_payload.get("status_reason", "benchmark release did not succeed"))
            )
        )
    )
    result["release_exit_code"] = (
        0
        if release_benchmark_success
        else (
            2 if missing or full_release_acceptance_failed else int(run_payload.get("exit_code", 2))
        )
    )

    if release_benchmark_success and publication_requested:
        try:
            # The first export discovers the deterministic bundle descriptor.  Then write that
            # descriptor and the final release result into the source campaign before exporting
            # again.  Repeat until the descriptor is stable so the bundle contains the same
            # metadata that the release result advertises.
            for _ in range(5):
                result["publication_bundle"] = publication_payload
                _record_publication_payload(campaign_root, publication_payload)
                _write_json(release_dir / "release_result.json", result)
                refreshed_payload = _build_publication_payload(
                    campaign_root=campaign_root,
                    release_tag=manifest.release_tag,
                    doi=manifest.doi,
                    repository_url=manifest.repository_url,
                )
                if refreshed_payload == publication_payload:
                    publication_payload = refreshed_payload
                    break
                publication_payload = refreshed_payload
            else:
                raise PublicationPreflightError(
                    "publication bundle descriptor did not stabilize after final metadata write"
                )
            result["publication_bundle"] = publication_payload
            _assert_no_historical_release_identity(Path(publication_payload["bundle_dir"]))
            _run_publication_preflight(Path(publication_payload["bundle_dir"]))
        except ReleaseArtifactIdentityError as exc:
            result["publication_bundle"] = None
            result["publication_preflight_status"] = "fail"
            result["publication_preflight_violations"] = [str(exc)]
            result["release_benchmark_success"] = False
            result["release_status"] = "publication_identity_rejected"
            result["release_status_reason"] = (
                "publication bundle retained a historical release identity"
            )
            result["release_exit_code"] = 2
            _write_json(release_dir / "release_result.json", result)
            _print_publication_identity_rejection()
            return 2
        except PublicationPreflightError as exc:
            result["publication_bundle"] = None
            result["publication_preflight_status"] = "fail"
            result["publication_preflight_violations"] = [str(exc)]
            result["release_benchmark_success"] = False
            result["release_status"] = "publication_preflight_failed"
            result["release_status_reason"] = (
                "publication bundle failed the final self-consistency preflight"
            )
            result["release_exit_code"] = 2
            release_result_path = release_dir / "release_result.json"
            _write_json(release_result_path, result)
            _print_public_result_file(release_result_path)
            return 2
    else:
        _write_json(release_dir / "release_result.json", result)

    _print_public_result_file(release_dir / "release_result.json")
    return int(result["release_exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
