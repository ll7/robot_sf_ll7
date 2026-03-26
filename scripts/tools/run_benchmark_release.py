#!/usr/bin/env python3
"""Run a benchmark release workflow on top of the camera-ready campaign stack."""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from robot_sf.benchmark.artifact_publication import export_publication_bundle
from robot_sf.benchmark.camera_ready_campaign import (
    load_campaign_config,
    prepare_campaign_preflight,
    run_campaign,
    write_campaign_report,
)
from robot_sf.benchmark.fallback_policy import campaign_exit_code
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
        }
    )
    _write_json(summary_path, summary)
    write_campaign_report(report_md_path, summary)

    # Stamp every provenance-facing artifact with the benchmark_release payload.
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
    """Run the benchmark release entrypoint and return a POSIX exit code."""
    raw_argv = list(argv) if argv is not None else list(sys.argv[1:])
    args = parse_release_args(raw_argv)

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    invoked_command = shlex.join([sys.executable, str(Path(__file__)), *raw_argv])

    manifest = load_release_manifest(args.manifest)
    cfg = load_campaign_config(manifest.canonical_campaign_config_path)
    validation = validate_release_manifest(manifest, campaign_config=cfg)

    resolved_manifest = build_resolved_release_manifest(manifest, campaign_config=cfg)
    if args.mode == "preflight":
        prepared = prepare_campaign_preflight(
            cfg,
            output_root=args.output_root,
            label=args.label,
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
        print(json.dumps(result, indent=2))
        return 2

    run_payload = run_campaign(
        cfg,
        output_root=args.output_root,
        label=args.label,
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
    _merge_release_provenance(campaign_root, release_provenance)

    missing = _required_artifacts_missing(campaign_root, manifest.required_artifact_paths)
    result["required_artifact_paths"] = list(manifest.required_artifact_paths)
    result["missing_required_artifacts"] = missing
    result["benchmark_release"] = release_provenance

    release_dir = campaign_root / "release"
    _write_json(release_dir / "release_manifest.resolved.json", resolved_manifest)

    benchmark_success = bool(run_payload.get("benchmark_success")) and not missing
    result["benchmark_success"] = benchmark_success
    if benchmark_success:
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

    if missing and benchmark_success is False:
        result["status"] = "missing_required_artifacts"
    release_status = result.get("status")
    if not isinstance(release_status, str) or not release_status:
        result["status"] = "ok" if benchmark_success else "benchmark_failed"
    _write_json(release_dir / "release_result.json", result)

    print(json.dumps(result, indent=2))
    return campaign_exit_code(result)


if __name__ == "__main__":
    raise SystemExit(main())
