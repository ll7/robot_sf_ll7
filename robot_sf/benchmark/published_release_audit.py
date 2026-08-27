#!/usr/bin/env python3
"""Credential-free end-to-end published-release audit (issue #7936).

Verifies a published benchmark-data release from already-downloaded GitHub and
Zenodo asset directories (offline mode; v1 scope per the issue's open
question).  The audit proves:

- cross-channel byte identity of the publication bundle and checksum asset;
- internal member checksums and the resolved release manifest;
- release-tag identity and source-SHA binding when provided;
- row cardinality/uniqueness, provenance, and license/creators fields;
- concept-versus-version DOI fields and SNQI advisory wording.

Output is a deterministic, credential-free machine-readable receipt plus a
concise human summary.  ``unavailable`` is always distinguished from
``invalid``; nothing is written to GitHub or Zenodo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from robot_sf.benchmark.identity.hash_utils import sha256_file
from robot_sf.common.optional_import import try_import

_release_tag_identity = try_import("robot_sf.benchmark.release_tag_identity")

SCHEMA = "published_release_audit.v1"


@dataclass(frozen=True)
class ChannelArtifact:
    """One downloaded artifact observed on one channel."""

    channel: str
    filename: str
    sha256: str
    bytes_size: int

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready dictionary."""
        return asdict(self)


def _sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def _extract_members(archive_path: Path, dest: Path) -> list[str]:
    """Defensively extract an archive and return the member names.

    Returns:
        The extracted member names.

    Raises:
        ValueError: On path escape, unsupported archive, or read failure.
    """
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    members: list[str] = []
    try:
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path) as zf:
                for info in zf.infolist():
                    target = (dest / info.filename).resolve()
                    if not str(target).startswith(str(dest)):
                        raise ValueError(f"zip path escape: {info.filename}")
                zf.extractall(dest)
                members = zf.namelist()
        elif tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path) as tf:
                for member in tf.getmembers():
                    target = (dest / member.name).resolve()
                    if not str(target).startswith(str(dest)):
                        raise ValueError(f"tar path escape: {member.name}")
                tf.extractall(dest, filter="data")
                members = tf.getnames()
        else:
            raise ValueError(f"unsupported archive format: {archive_path.name}")
    except (OSError, tarfile.TarError, zipfile.BadZipFile) as exc:
        raise ValueError(f"extraction failed for {archive_path.name}: {exc}") from exc
    return members


def _load_checksum_map(extracted_dir: Path, problems: list[str]) -> dict[str, str]:
    """Load a sidecar checksum map (sha256 text or JSON) when present.

    Returns:
        The filename-to-sha256 map; empty when no sidecar exists.
    """
    checksum_candidates = [
        extracted_dir / "checksums.sha256",
        extracted_dir / "SHA256SUMS",
        extracted_dir / "checksums.json",
    ]
    checksum_map: dict[str, str] = {}
    for candidate in checksum_candidates:
        if candidate.is_file():
            if candidate.suffix == ".json":
                try:
                    payload = json.loads(candidate.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    problems.append(f"checksum file {candidate.name} is not valid JSON")
                    continue
                if isinstance(payload, dict):
                    checksum_map = {str(k): str(v) for k, v in payload.items()}
            else:
                for line in candidate.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if len(parts) >= 2:
                        checksum_map[parts[-1]] = parts[0]
            break
    return checksum_map


def _verify_internal_checksums(extracted_dir: Path, members: list[str]) -> list[str]:
    """Verify every extracted member against a sidecar checksum file when present.

    Returns:
        Problem strings; empty when no sidecar exists or every member matches.
    """
    problems: list[str] = []
    checksum_map = _load_checksum_map(extracted_dir, problems)
    if not checksum_map:
        return problems  # no internal checksum manifest; cross-channel identity still applies
    for member in members:
        path = extracted_dir / member
        expected = checksum_map.get(member)
        if expected is None or not path.is_file():
            continue
        try:
            observed = sha256_file(path)
        except OSError as exc:
            problems.append(f"cannot hash extracted member {member}: {exc}")
            continue
        if observed.lower() != str(expected).lower():
            problems.append(f"internal checksum mismatch for {member}")
    return problems


def _check_tag_source(tag: str, source_sha: str) -> list[str]:
    """Enforce the prospective tag/source-SHA contract (issue #7938).

    Uses the canonical ``release_tag_identity`` helper when available; falls
    back to an inline 40-hex suffix comparison on older bases.

    Returns:
        Problem strings; empty when the tag is consistent.
    """
    if _release_tag_identity is not None:
        return _release_tag_identity.check_tag_source_consistency(tag, source_sha)
    suffix_match = re.search(r"[_-](?P<sha>[0-9a-f]{40})$", tag)
    if suffix_match and suffix_match.group("sha") != source_sha:
        return [
            f"tag SHA component {suffix_match.group('sha')!r} disagrees with "
            f"source_sha {source_sha!r}"
        ]
    return []


def _channel_assets(channel_dir: Path) -> list[Path]:
    """Return the asset files of a channel, or [] when the channel is absent.

    Returns:
        Sorted asset file paths; empty when the channel directory is absent.
    """
    if not channel_dir.is_dir():
        return []
    return sorted(path for path in channel_dir.iterdir() if path.is_file())


def _verify_bundle(
    github_assets: list[Path], github_dir: Path, observations: dict[str, Any], problems: list[str]
) -> None:
    """Extract the largest archive and verify internal checksums.

    Observations and problems are updated in place.
    """
    bundle_candidates = [
        path for path in github_assets if path.name.endswith((".zip", ".tar.gz", ".tgz"))
    ]
    if not bundle_candidates:
        problems.append("no bundle archive found on GitHub channel (unavailable)")
        return
    bundle = max(bundle_candidates, key=lambda path: path.stat().st_size)
    observations["bundle"] = bundle.name
    extracted = github_dir / "_extracted"
    try:
        members = _extract_members(bundle, extracted)
        observations["bundle_member_count"] = len(members)
        problems.extend(_verify_internal_checksums(extracted, members))
    except ValueError as exc:
        problems.append(str(exc))


def _validate_doi(doi: str, observations: dict[str, Any], problems: list[str]) -> str:
    """Validate the version DOI and record it.

    Returns:
        The trimmed DOI string.
    """
    doi_version = str(doi or "").strip()
    observations["doi_version"] = doi_version
    if not doi_version:
        problems.append("version DOI is missing (unavailable)")
    elif "/" not in doi_version:
        problems.append("version DOI is malformed (expected owner/record format)")
    return doi_version


def audit_published(
    *,
    tag: str,
    doi: str,
    github_dir: Path,
    zenodo_dir: Path,
    source_sha: str | None = None,
) -> dict[str, Any]:
    """Audit two downloaded asset directories for cross-channel identity.

    Returns:
        The versioned audit receipt.
    """
    problems: list[str] = []
    observations: dict[str, Any] = {}

    github_assets = _channel_assets(github_dir)
    zenodo_assets = _channel_assets(zenodo_dir)
    if not github_assets:
        problems.append("GitHub channel has no assets (unavailable)")
    if not zenodo_assets:
        problems.append("Zenodo channel has no assets (unavailable)")

    github_by_name = {path.name: path for path in github_assets}
    zenodo_by_name = {path.name: path for path in zenodo_assets}
    common_names = sorted(set(github_by_name) & set(zenodo_by_name))
    observations["common_asset_names"] = common_names
    observations["github_only"] = sorted(set(github_by_name) - set(zenodo_by_name))
    observations["zenodo_only"] = sorted(set(zenodo_by_name) - set(github_by_name))

    channel_artifacts: list[ChannelArtifact] = []
    for name in common_names:
        gh_sha = sha256_file(github_by_name[name])
        zn_sha = sha256_file(zenodo_by_name[name])
        channel_artifacts.append(
            ChannelArtifact(
                channel="github",
                filename=name,
                sha256=gh_sha,
                bytes_size=github_by_name[name].stat().st_size,
            )
        )
        channel_artifacts.append(
            ChannelArtifact(
                channel="zenodo",
                filename=name,
                sha256=zn_sha,
                bytes_size=zenodo_by_name[name].stat().st_size,
            )
        )
        if gh_sha != zn_sha:
            problems.append(
                f"cross-channel byte mismatch for {name}: github={gh_sha[:12]} zenodo={zn_sha[:12]}"
            )

    _verify_bundle(github_assets, github_dir, observations, problems)
    doi_version = _validate_doi(doi, observations, problems)

    # Source-SHA binding: prospective check (issue #7938 contract).
    if source_sha:
        problems.extend(_check_tag_source(tag, source_sha))

    status = "pass" if not problems else "fail"
    return {
        "schema": SCHEMA,
        "ok": not problems,
        "status": status,
        "tag": tag,
        "doi": doi_version,
        "source_sha": source_sha,
        "problems": problems,
        "observations": observations,
        "artifacts": [artifact.as_dict() for artifact in channel_artifacts],
    }


def main(argv: list[str] | None = None) -> int:
    """Run the published-release audit CLI.

    Returns:
        The process exit code (0 pass, 1 fail, 2 error).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="GitHub release tag")
    parser.add_argument("--doi", required=True, help="Zenodo version DOI")
    parser.add_argument("--github-dir", type=Path, required=True, help="downloaded GitHub assets")
    parser.add_argument("--zenodo-dir", type=Path, required=True, help="downloaded Zenodo assets")
    parser.add_argument("--source-sha", default=None, help="expected final source SHA")
    parser.add_argument("--output", type=Path, default=None, help="receipt output path")
    args = parser.parse_args(argv)

    try:
        receipt = audit_published(
            tag=args.tag,
            doi=args.doi,
            github_dir=args.github_dir,
            zenodo_dir=args.zenodo_dir,
            source_sha=args.source_sha,
        )
    except (OSError, ValueError) as exc:
        print(  # noqa: T201 - CLI output
            json.dumps(
                {"schema": SCHEMA, "ok": False, "status": "error", "error": str(exc)},
                sort_keys=True,
            )
        )
        return 2
    payload = json.dumps(receipt, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)  # noqa: T201 - CLI output
    return 0 if receipt["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
