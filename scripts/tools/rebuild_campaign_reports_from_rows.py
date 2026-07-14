#!/usr/bin/env python3
"""Rebuild campaign reports/publication bundle from pre-existing arm rows, without re-simulating.

Context (2026-07-13 release/v0.0.3-h600-s30-extended reconstruction): a 14-arm camera-ready
campaign ran on a cluster. Run 1 completed 13 arms cleanly (1,440 episodes each) but the ``ppo``
arm produced 0 rows (unexpected_failure). A resume attempt using the same ``--campaign-id``
re-ran the 13 already-complete arms and appended duplicate rows onto their ``episodes.jsonl``
files (see the ``#5392``/``_resume_plan.py`` diagnosis in the accompanying report), while ``ppo``
produced a clean, fresh 1,440-row run. The cluster job was then cancelled.

This script reconstructs a release-grade campaign root from two clean data snapshots (the 13
completed arms, deduplicated to 1,440 unique episodes each, plus the freshly-completed ppo arm)
by driving the *real* release/campaign orchestration code
(:func:`robot_sf.benchmark.camera_ready_campaign.run_campaign`, mirroring
``scripts/tools/run_benchmark_release.py``'s ``run`` mode) with one injected collaborator:
``run_batch`` is replaced by :func:`_native_reuse_run_batch`, which loads each arm's *existing*
``episodes.jsonl`` (already placed under ``runs/<planner>__<kinematics>/`` before this script
runs) instead of simulating anything. Preflight, aggregation, SNQI diagnostics, fairness,
reporting, and publication-bundle export all run for real, exactly as they would for a live
campaign -- only the (expensive, GPU/CPU-bound) simulation step is skipped because the episode
data already exists on disk.

Usage:
    uv run python scripts/tools/rebuild_campaign_reports_from_rows.py \\
        --manifest configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_release_v0_0_3.yaml \\
        --campaign-id paper_experiment_matrix_v2_h600_s30_extended_release_v0_0_3_final

Preconditions:
    - ``output/benchmarks/camera_ready/<campaign-id>/runs/<arm>/episodes.jsonl`` must already
      exist for every enabled planner x kinematics arm, with the full expected episode count and
      globally-unique ``episode_id`` values (verify with a separate row-count/uniqueness check
      before running this script -- it does not deduplicate rows itself, it only refuses to
      fabricate missing ones).
    - A prior ``summary.json`` should exist alongside each arm's ``episodes.jsonl`` (a normal
      by-product of any earlier real ``run_batch`` execution for that arm); its non-count fields
      (``preflight``, ``algorithm_readiness``, ``algorithm_metadata_contract``,
      ``observation_noise*``, ``tracking_precision*``, ``synthetic_actuation_profile``,
      ``latency_stress_*``, ``workers``, ``parallel_execution``, ``provenance``) are carried
      through verbatim; the count fields (``written``, ``total_jobs``, ``failed_jobs``,
      ``skipped_jobs``, ``successful_jobs``, ``failures``) are recomputed from the actual
      on-disk row count so a stale/duplicated prior summary can't leak wrong counts forward.

This is a reconstruction tool, not a general resume mechanism: it does not implement any
new resume/dedup logic, and it deliberately fails closed (raises) rather than simulating when an
arm's ``episodes.jsonl`` is missing or empty.
"""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from robot_sf.benchmark.aggregate import read_jsonl
from robot_sf.benchmark.artifact_publication import export_publication_bundle
from robot_sf.benchmark.camera_ready._config import load_campaign_config
from robot_sf.benchmark.camera_ready._preflight import prepare_campaign_preflight
from robot_sf.benchmark.camera_ready._reporting import write_campaign_report
from robot_sf.benchmark.camera_ready._util import _utc_now

# NOTE: run_campaign is imported from the extracted campaign module directly (not from the
# robot_sf.benchmark.camera_ready_campaign legacy facade) because the facade's run_campaign
# hard-binds run_batch=robot_sf.benchmark.runner.run_batch with no override parameter. Only the
# lower-level robot_sf.benchmark.camera_ready.campaign.run_campaign accepts an injectable
# run_batch collaborator, which this script relies on to skip simulation entirely.
from robot_sf.benchmark.camera_ready.campaign import run_campaign
from robot_sf.benchmark.orca_preflight import OrcaRvo2PreflightError, check_orca_rvo2_preflight
from robot_sf.benchmark.release_protocol import (
    build_release_provenance,
    build_resolved_release_manifest,
    load_release_manifest,
    parse_release_args,
    validate_release_manifest,
)
from robot_sf.common.artifact_paths import get_artifact_category_path, get_repository_root

if TYPE_CHECKING:
    from collections.abc import Sequence


# --------------------------------------------------------------------------------------
# Injected run_batch replacement: reuse existing rows, never simulate.
# --------------------------------------------------------------------------------------


def _native_reuse_run_batch(
    scenarios: list[dict[str, Any]],
    *,
    out_path: str | Path,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Return a run_batch-shaped summary for an arm whose episodes.jsonl already exists.

    This is invoked once per enabled planner x kinematics arm by the real
    ``run_campaign`` orchestrator, in place of ``robot_sf.benchmark.runner.run_batch``.
    It never simulates: it fails closed if the arm's pre-existing ``episodes.jsonl`` is
    missing, empty, or contains duplicate ``episode_id`` values, and otherwise returns a
    summary dict shaped exactly like a real ``run_batch`` return (reusing whatever
    non-count fields are available from a prior ``summary.json`` alongside the file).

    Returns:
        A run_batch-contract summary dict with recomputed count fields and a
        ``reconstruction`` provenance block marking the arm as natively reused (no
        re-simulation).

    Raises:
        FileNotFoundError: If the arm's ``episodes.jsonl`` does not exist or is empty.
        ValueError: If the arm's ``episodes.jsonl`` contains duplicate ``episode_id``
            values (this script never deduplicates; the campaign root must already be
            reconciled before this driver runs).
    """
    out_path = Path(out_path)
    arm_dir = out_path.parent
    if not out_path.is_file() or out_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Reconstruction precondition failed: expected pre-existing episodes at {out_path} "
            "(this driver never simulates; place the real episodes.jsonl for this arm first)."
        )

    records = read_jsonl(str(out_path))
    episode_ids = [str(rec.get("episode_id")) for rec in records]
    n_rows = len(episode_ids)
    n_unique = len(set(episode_ids))
    if n_unique != n_rows:
        raise ValueError(
            f"Reconstruction precondition failed: {out_path} has {n_rows} rows but only "
            f"{n_unique} unique episode_id values (duplicate rows must be deduplicated before "
            "running this driver)."
        )

    summary_path = arm_dir / "summary.json"
    prior_summary: dict[str, Any] = {}
    if summary_path.is_file():
        try:
            loaded = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                prior_summary = loaded
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not reuse prior summary.json at {}: {}", summary_path, exc)

    summary: dict[str, Any] = dict(prior_summary)
    summary.update(
        {
            "status": "ok",
            "out_path": str(out_path),
            "total_jobs": n_rows,
            "written": n_rows,
            "failed_jobs": 0,
            "skipped_jobs": 0,
            "successful_jobs": n_rows,
            "failures": [],
        }
    )
    summary["reconstruction"] = {
        "mode": "native_reuse_no_resimulation",
        "reused_at_utc": _utc_now(),
        "episodes_path": str(out_path),
        "episode_count": n_rows,
        "prior_summary_reused": bool(prior_summary),
        "note": (
            "Rows were produced by a real run_batch execution outside this reconstruction "
            "(either the original cluster run or a targeted re-run of a single failed arm); "
            "this driver only re-ran the reporting/aggregation/publication stages."
        ),
    }
    return summary


# --------------------------------------------------------------------------------------
# The rest mirrors scripts/tools/run_benchmark_release.py's ``run`` mode, with run_batch injected.
# --------------------------------------------------------------------------------------


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
        return str(resolved)


def _merge_release_provenance(campaign_root: Path, release_provenance: dict[str, Any]) -> None:
    """Inject release provenance into campaign artifacts and refresh the markdown report."""
    summary_path = campaign_root / "reports" / "campaign_summary.json"
    report_md_path = campaign_root / "reports" / "campaign_report.md"
    manifest_path = campaign_root / "campaign_manifest.json"
    benchmark_manifest_path = campaign_root / "manifest.json"
    run_meta_path = campaign_root / "run_meta.json"

    summary = _read_json(summary_path)
    summary["benchmark_release"] = dict(release_provenance)
    campaign_block = summary.get("campaign")
    if not isinstance(campaign_block, dict):
        campaign_block = {}
        summary["campaign"] = campaign_block
    campaign_block.update(
        {
            "benchmark_protocol_version": release_provenance["benchmark_protocol_version"],
            "benchmark_release_id": release_provenance["release_id"],
            "benchmark_release_tag": release_provenance["release_tag"],
            "benchmark_release_manifest_path": release_provenance["manifest_path"],
            "benchmark_release_manifest_sha256": release_provenance["manifest_sha256"],
            "canonical_release_config": release_provenance["canonical_campaign_config"],
        }
    )
    _write_json(summary_path, summary)
    write_campaign_report(report_md_path, summary)

    for path in (manifest_path, benchmark_manifest_path, run_meta_path):
        payload = _read_json(path)
        payload["benchmark_release"] = dict(release_provenance)
        _write_json(path, payload)


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


def main(argv: Sequence[str] | None = None) -> int:
    """Run the reconstruction entrypoint and return a POSIX exit code."""
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
        }
        print(json.dumps(preflight_payload, indent=2))
        return 0 if validation["status"] == "valid" else 2

    result: dict[str, Any] = {
        "mode": "run",
        "manifest_validation": validation,
        "resolved_manifest": resolved_manifest,
    }
    if validation["status"] != "valid":
        result["benchmark_success"] = False
        result["status"] = "invalid_manifest"
        print(json.dumps(result, indent=2))
        return 2

    # The only difference from scripts/tools/run_benchmark_release.py: run_batch is replaced
    # by _native_reuse_run_batch so no simulation happens, only reporting/aggregation/export.
    run_payload = run_campaign(
        cfg,
        output_root=args.output_root,
        label=args.label,
        campaign_id=args.campaign_id,
        skip_publication_bundle=True,
        invoked_command=invoked_command,
        run_batch=_native_reuse_run_batch,
    )
    result.update(run_payload)
    campaign_root = Path(str(run_payload["campaign_root"])).resolve()

    release_provenance = build_release_provenance(
        manifest,
        campaign_root=campaign_root,
        invoked_command=invoked_command,
    )
    _merge_release_provenance(campaign_root, release_provenance)

    missing = _required_artifacts_missing(campaign_root, manifest.required_artifact_paths)
    result["required_artifact_paths"] = list(manifest.required_artifact_paths)
    result["missing_required_artifacts"] = missing
    result["benchmark_release"] = release_provenance

    release_dir = campaign_root / "release"
    _write_json(release_dir / "release_manifest.resolved.json", resolved_manifest)

    release_benchmark_success = bool(run_payload.get("benchmark_success")) and not missing
    result["release_benchmark_success"] = release_benchmark_success
    if release_benchmark_success:
        publication_payload = _build_publication_payload(
            campaign_root=campaign_root,
            release_tag=manifest.release_tag,
            doi=manifest.doi,
            repository_url=manifest.repository_url,
        )
        result["publication_bundle"] = publication_payload

        summary_path = campaign_root / "reports" / "campaign_summary.json"
        summary = _read_json(summary_path)
        summary["publication_bundle"] = publication_payload
        _write_json(summary_path, summary)
        write_campaign_report(campaign_root / "reports" / "campaign_report.md", summary)
    else:
        result["publication_bundle"] = None

    result["release_status"] = (
        "missing_required_artifacts"
        if missing
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
            else str(run_payload.get("status_reason", "benchmark release did not succeed"))
        )
    )
    result["release_exit_code"] = (
        0 if release_benchmark_success else (2 if missing else int(run_payload.get("exit_code", 2)))
    )
    _write_json(release_dir / "release_result.json", result)

    print(json.dumps(result, indent=2))
    return int(result["release_exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
