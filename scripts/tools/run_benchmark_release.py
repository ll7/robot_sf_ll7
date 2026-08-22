#!/usr/bin/env python3
"""Run a benchmark release workflow on top of the camera-ready campaign stack.

Exit codes follow the wrapped campaign semantics for non-success benchmark outcomes:
- 0: benchmark-success release
- 2: unexpected failure or missing required release artifacts
- 3: accepted-unavailable-only campaign outcome (non-success, fail-closed)
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

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
from robot_sf.benchmark.release_acceptance import validate_full_benchmark_release_acceptance
from robot_sf.benchmark.release_protocol import (
    HISTORICAL_ZENODO_CONCEPT_DOIS,
    build_release_provenance,
    build_resolved_release_manifest,
    load_release_manifest,
    parse_release_args,
    validate_release_manifest,
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


class ReleaseArtifactIdentityError(ValueError):
    """Raised when a campaign artifact still carries a predecessor identity."""


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
    summary_path = campaign_root / "reports" / "campaign_summary.json"
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
    """Return required artifact paths that are missing from the campaign root."""
    missing: list[str] = []
    for relative_path in required_paths:
        candidate = campaign_root / relative_path
        if not candidate.exists():
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
    summary_path = campaign_root / "reports" / "campaign_summary.json"
    summary = _read_json(summary_path)
    summary["publication_bundle"] = publication_payload
    _write_json(summary_path, summary)
    write_campaign_report(campaign_root / "reports" / "campaign_report.md", summary)


def _record_release_acceptance(campaign_root: Path, acceptance: dict[str, Any]) -> None:
    """Persist the full-release gate beside the campaign summary and report."""
    summary_path = campaign_root / "reports" / "campaign_summary.json"
    summary = _read_json(summary_path)
    summary["full_release_acceptance"] = acceptance
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


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901, PLR0912, PLR0915
    """Run the benchmark release entrypoint and return a POSIX exit code."""
    raw_argv = list(argv) if argv is not None else list(sys.argv[1:])
    args = parse_release_args(raw_argv)

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    invoked_command = shlex.join([sys.executable, str(Path(__file__)), *raw_argv])

    manifest = load_release_manifest(args.manifest)
    cfg = load_campaign_config(manifest.canonical_campaign_config_path)
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

    resolved_manifest = build_resolved_release_manifest(manifest, campaign_config=cfg)
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
    result.update(run_payload)
    campaign_root = Path(str(run_payload["campaign_root"])).resolve()

    release_provenance = build_release_provenance(
        manifest,
        campaign_root=campaign_root,
        invoked_command=invoked_command,
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

    release_acceptance = validate_full_benchmark_release_acceptance(
        campaign_root,
        manifest=manifest,
        campaign_config=cfg,
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

    missing = _required_artifacts_missing(campaign_root, manifest.required_artifact_paths)
    result["required_artifact_paths"] = list(manifest.required_artifact_paths)
    result["missing_required_artifacts"] = missing
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
            _write_json(release_dir / "release_result.json", result)
            print(json.dumps(result, indent=2))
            return 2
    else:
        _write_json(release_dir / "release_result.json", result)

    print(json.dumps(result, indent=2))
    return int(result["release_exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
