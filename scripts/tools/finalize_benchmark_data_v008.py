#!/usr/bin/env python3
"""Validate and bundle an accepted 0.0.8 campaign after its episode run.

The canonical release runner deliberately defers publication: the predecessor
equivalence and robot-force reports require its completed episode rows. This
command works on a new copy so the accepted producer remains untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from robot_sf.benchmark.release_acceptance import validate_full_benchmark_release_acceptance
from robot_sf.benchmark.release_protocol import (
    load_release_campaign_config,
    verify_resolved_release_identity,
)
from robot_sf.common.artifact_paths import get_repository_root
from scripts.tools.run_benchmark_release import (
    _assert_no_historical_release_identity,
    _build_publication_payload,
    _record_publication_payload,
    _run_publication_preflight,
    _write_json,
)

BASELINE_ARCHIVE_SHA256 = "684da7c557c426756f22ddbf5cb3270141ee8ae385669a39d36f324852a6fb2f"
BASELINE_SOURCE_SHA = "07f7e8d43084de748915e1b1eb8b2a1603357c6e"
EXPECTED_EPISODES = 20160


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
    result = _read_mapping(source_path)
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


def finalize(
    *,
    producer_root: Path,
    candidate_root: Path,
    resolved_identity: Path,
    baseline_archive: Path,
    expected_source_sha: str,
) -> dict[str, Any]:
    """Copy, validate, and publish one accepted exact-source campaign."""
    manifest = verify_resolved_release_identity(resolved_identity)
    if manifest.source_sha != expected_source_sha or manifest.source_sha == BASELINE_SOURCE_SHA:
        raise ValueError("0.0.8 resolved identity has the wrong scientific source")
    if _sha256(baseline_archive) != BASELINE_ARCHIVE_SHA256:
        raise ValueError("frozen 0.0.7 archive checksum mismatch")
    producer_root, candidate_root, producer_result = _prepare_candidate(
        producer_root, candidate_root, expected_source_sha, manifest
    )
    report_dir = candidate_root / "reports"
    equivalence = report_dir / "metric_equivalence.json"
    equivalence_log = candidate_root.parent / f"{candidate_root.name}.equivalence.log"
    force_log = candidate_root.parent / f"{candidate_root.name}.robot_force.log"
    stage = "metric_equivalence"
    try:
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


def main() -> int:
    """Run the post-campaign release finalizer from its explicit source pins."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--resolved-identity", type=Path, required=True)
    parser.add_argument("--baseline-archive", type=Path, required=True)
    parser.add_argument("--expected-source-sha", required=True)
    args = parser.parse_args()
    receipt = finalize(
        producer_root=args.producer_root,
        candidate_root=args.candidate_root,
        resolved_identity=args.resolved_identity,
        baseline_archive=args.baseline_archive,
        expected_source_sha=args.expected_source_sha,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
