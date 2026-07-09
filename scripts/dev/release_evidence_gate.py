"""Fresh-clone reproduction gate for tagged research-release evidence (issue #3205).

This gate turns the manual "download release -> regenerate tables -> verify checksums"
chain into a single repeatable, fail-closed command. Given a release archive (or an
already-extracted reports source root) plus the tracked artifact spec and reference
manifest, it:

1. verifies the archive SHA-256 against the expected value (when an archive is provided);
2. regenerates the dissertation artifact bundle from the canonical report rows using the
   existing ``benchmark_publication_bundle.py dissertation-bundle`` contract;
3. compares each regenerated artifact's SHA-256 against the tracked reference manifest;
4. emits a single citable ``release_evidence_snapshot`` JSON and exits non-zero on any
   mismatch, missing artifact, or integrity failure (fail-closed).

Reproduction here means regenerating the *promoted tables from the released canonical
rows*; it does not re-run the benchmark campaign. A passing snapshot is the evidence that
a tagged release is independently reproducible from a clean checkout and therefore citable.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robot_sf.benchmark.identity.hash_utils import sha256_file as _sha256_file

SCHEMA_VERSION = "release_evidence_snapshot.v1"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_BUNDLE_TOOL = _REPO_ROOT / "scripts" / "tools" / "benchmark_publication_bundle.py"


@dataclass(frozen=True)
class AcquisitionSpec:
    """How the canonical report rows are acquired for regeneration.

    Provide exactly one of ``archive`` (verified + extracted) or ``source_root`` (an
    already-extracted reports directory).
    """

    archive: Path | None = None
    expected_archive_sha256: str | None = None
    source_root: Path | None = None
    reports_subpath: str = "payload/reports"


@dataclass(frozen=True)
class ReleaseIdentity:
    """Citable identity fields recorded verbatim in the snapshot."""

    tag: str | None = None
    url: str | None = None
    doi: str | None = None


def _safe_extract(archive: Path, dest: Path) -> None:
    """Extract ``archive`` into ``dest``, rejecting path-traversal members (fail-closed)."""
    with tarfile.open(archive, "r:*") as tar:
        for member in tar.getmembers():
            target = (dest / member.name).resolve()
            if not target.is_relative_to(dest.resolve()):
                raise ValueError(f"Unsafe archive member path: {member.name}")
        tar.extractall(dest)


def _index_by_artifact_id(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index a dissertation artifact manifest's entries by ``artifact_id``."""
    return {entry["artifact_id"]: entry for entry in manifest.get("artifacts", [])}


def _regenerate_bundle(
    source_root: Path,
    out_dir: Path,
    *,
    bundle_name: str,
    artifact_spec: Path,
    commit: str,
) -> Path:
    """Regenerate the dissertation artifact bundle and return its bundle directory."""
    cmd = [
        sys.executable,
        str(_BUNDLE_TOOL),
        "dissertation-bundle",
        "--source-root",
        str(source_root),
        "--out-dir",
        str(out_dir),
        "--bundle-name",
        bundle_name,
        "--artifact-spec",
        str(artifact_spec),
        "--command",
        "release_evidence_gate.py: regenerate promoted tables from released canonical rows",
        "--commit",
        commit,
        "--overwrite",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out_dir / bundle_name


def run_gate(
    *,
    artifact_spec: Path,
    reference_manifest: Path,
    source_commit: str,
    acquisition: AcquisitionSpec,
    identity: ReleaseIdentity = ReleaseIdentity(),
    workdir: Path | None = None,
) -> dict[str, Any]:
    """Run the reproduction gate and return the release-evidence snapshot dict.

    Exactly one acquisition path must be provided via ``acquisition``: an ``archive``
    (verified + extracted) or a ``source_root`` (already-extracted canonical report rows).
    The snapshot's ``status`` is ``PASS`` only when archive integrity holds (when
    applicable) and every reference artifact is regenerated with a matching SHA-256.
    """
    archive = acquisition.archive
    source_root = acquisition.source_root
    expected_archive_sha256 = acquisition.expected_archive_sha256
    reports_subpath = acquisition.reports_subpath
    if (archive is None) == (source_root is None):
        raise ValueError("Provide exactly one of archive or source_root")

    reference = json.loads(reference_manifest.read_text())
    reference_index = _index_by_artifact_id(reference)
    bundle_name = reference.get("bundle_name", "release_evidence_bundle")

    failures: list[str] = []
    archive_check: dict[str, Any] | None = None

    tmp_ctx = tempfile.TemporaryDirectory(dir=str(workdir) if workdir else None)
    with tmp_ctx as tmp_name:
        tmp = Path(tmp_name)
        if archive is not None:
            actual_sha = _sha256_file(archive)
            integrity_ok = expected_archive_sha256 is None or actual_sha == expected_archive_sha256
            archive_check = {
                "archive_name": archive.name,
                "expected_sha256": expected_archive_sha256,
                "actual_sha256": actual_sha,
                "match": integrity_ok,
            }
            if not integrity_ok:
                failures.append(
                    f"archive SHA-256 mismatch: expected {expected_archive_sha256}, got {actual_sha}"
                )
            extract_root = tmp / "extracted"
            extract_root.mkdir()
            _safe_extract(archive, extract_root)
            candidates = sorted(extract_root.glob(f"*/{reports_subpath}")) or sorted(
                extract_root.glob(f"**/{reports_subpath}")
            )
            if not candidates:
                failures.append(f"reports source root '{reports_subpath}' not found in archive")
                resolved_source = None
            else:
                resolved_source = candidates[0]
        else:
            resolved_source = source_root

        artifacts_report: list[dict[str, Any]] = []
        if resolved_source is not None and not failures:
            out_dir = tmp / "regenerated"
            bundle_dir = _regenerate_bundle(
                resolved_source,
                out_dir,
                bundle_name=bundle_name,
                artifact_spec=artifact_spec,
                commit=source_commit,
            )
            regenerated = json.loads((bundle_dir / "artifact_manifest.json").read_text())
            regenerated_index = _index_by_artifact_id(regenerated)

            for artifact_id, ref_entry in reference_index.items():
                tracked_sha = ref_entry.get("sha256")
                regen_entry = regenerated_index.get(artifact_id)
                regen_sha = regen_entry.get("sha256") if regen_entry else None
                match = regen_entry is not None and regen_sha == tracked_sha
                artifacts_report.append(
                    {
                        "artifact_id": artifact_id,
                        "tracked_sha256": tracked_sha,
                        "reproduced_sha256": regen_sha,
                        "match": match,
                        "fallback_degraded_summary": ref_entry.get("fallback_degraded_summary"),
                    }
                )
                if not match:
                    reason = "not regenerated" if regen_entry is None else "checksum mismatch"
                    failures.append(f"artifact '{artifact_id}' {reason}")

    status = "PASS" if not failures else "FAIL"
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "release_tag": identity.tag,
        "release_url": identity.url,
        "doi": identity.doi,
        "source_commit": source_commit,
        "reference_manifest": str(reference_manifest),
        "artifact_spec": str(artifact_spec),
        "archive": archive_check,
        "artifacts": artifacts_report,
        "reproduced_artifact_count": sum(1 for a in artifacts_report if a["match"]),
        "expected_artifact_count": len(reference_index),
        "failures": failures,
        "policy": {
            "reproduction_scope": "regenerate promoted tables from released canonical rows; "
            "does not re-run the benchmark campaign",
            "fail_closed": True,
            "fallback_or_degraded_is_not_success": True,
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-spec", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--source-commit", type=str, required=True)
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument("--expected-archive-sha256", type=str, default=None)
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--reports-subpath", type=str, default="payload/reports")
    parser.add_argument("--release-tag", type=str, default=None)
    parser.add_argument("--release-url", type=str, default=None)
    parser.add_argument("--doi", type=str, default=None)
    parser.add_argument("--out", type=Path, default=None, help="Write the snapshot JSON here.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the gate CLI; return 0 on PASS and 1 on any reproduction failure (fail-closed)."""
    args = _build_parser().parse_args(argv)
    snapshot = run_gate(
        artifact_spec=args.artifact_spec,
        reference_manifest=args.reference_manifest,
        source_commit=args.source_commit,
        acquisition=AcquisitionSpec(
            archive=args.archive,
            expected_archive_sha256=args.expected_archive_sha256,
            source_root=args.source_root,
            reports_subpath=args.reports_subpath,
        ),
        identity=ReleaseIdentity(tag=args.release_tag, url=args.release_url, doi=args.doi),
    )
    rendered = json.dumps(snapshot, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n")
    print(rendered)
    return 0 if snapshot["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
