#!/usr/bin/env python3
"""Fail closed unless a queued worker can read every staged source byte.

Use this immediately before queue dispatch.  The JSON manifest records only
immutable relative paths and SHA-256 values; the worker host and its absolute
staging root are runtime arguments or private-operations state, never tracked
repository data.

Manifest shape::

    {
        "schema_version": "cross_host_source_staging.v1",
        "sources": [{"relative_path": "reports/summary.csv", "sha256": "..."}],
    }

The worker is probed over SSH with ``BatchMode=yes``.  Missing, unreadable,
non-file, mismatched, malformed, or unreachable inputs block dispatch.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, NamedTuple

SCHEMA_VERSION = "cross_host_source_staging.v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CONTROL_CHARACTER_RE = re.compile(r"[\x00-\x1f\x7f]")


class ContractError(ValueError):
    """Raised when a source-staging manifest cannot safely be probed."""


class Source(NamedTuple):
    """One immutable source expected beneath the worker staging root."""

    relative_path: str
    sha256: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot load manifest: {exc}") from exc
    _require(isinstance(payload, dict), "manifest must be a JSON object")
    return payload


def parse_manifest(manifest: Mapping[str, Any]) -> list[Source]:
    """Validate a portable manifest and return its unique expected sources."""
    _require(manifest.get("schema_version") == SCHEMA_VERSION, "schema_version mismatch")
    raw_sources = manifest.get("sources")
    _require(isinstance(raw_sources, list) and raw_sources, "sources must be a non-empty list")

    sources: list[Source] = []
    seen_paths: set[str] = set()
    for index, raw_source in enumerate(raw_sources):
        _require(isinstance(raw_source, Mapping), f"sources[{index}] must be a mapping")
        relative_path = raw_source.get("relative_path")
        _require(isinstance(relative_path, str), f"sources[{index}].relative_path must be a string")
        path = Path(relative_path)
        _require(
            relative_path == path.as_posix(), f"sources[{index}].relative_path is not portable"
        )
        _require("\\" not in relative_path, f"sources[{index}].relative_path is not portable")
        _require(not path.is_absolute(), f"sources[{index}].relative_path must be relative")
        _require(
            path.parts and ".." not in path.parts, f"sources[{index}].relative_path escapes root"
        )
        _require(
            CONTROL_CHARACTER_RE.search(relative_path) is None,
            "source path has control character",
        )
        _require(relative_path not in seen_paths, f"duplicate source path: {relative_path}")

        sha256 = raw_source.get("sha256")
        _require(isinstance(sha256, str) and SHA256_RE.fullmatch(sha256), "invalid SHA-256 digest")
        sources.append(Source(relative_path=relative_path, sha256=sha256))
        seen_paths.add(relative_path)
    return sources


def _remote_paths(staging_root: Path, sources: list[Source]) -> list[str]:
    _require(staging_root.is_absolute(), "staging_root must be absolute on the worker")
    root_text = str(staging_root)
    _require(
        CONTROL_CHARACTER_RE.search(root_text) is None,
        "staging_root has control character",
    )
    return [str(staging_root / source.relative_path) for source in sources]


def build_remote_probe(paths: list[str]) -> str:
    """Build a quoted POSIX-shell probe with a stable tab-delimited protocol."""
    quoted_paths = " ".join(shlex.quote(path) for path in paths)
    return "\n".join(
        [
            "set -u",
            f"for path in {quoted_paths}; do",
            '  if [ ! -e "$path" ]; then',
            '    printf "missing\\t%s\\n" "$path"',
            '  elif [ ! -f "$path" ]; then',
            '    printf "not_file\\t%s\\n" "$path"',
            '  elif [ ! -r "$path" ]; then',
            '    printf "unreadable\\t%s\\n" "$path"',
            '  elif digest=$(sha256sum -- "$path" 2>/dev/null); then',
            '    printf "ok\\t%s\\t%s\\n" "${digest%% *}" "$path"',
            "  else",
            '    printf "unreadable\\t%s\\n" "$path"',
            "  fi",
            "done",
        ]
    )


def _parse_probe_output(
    output: str, expected_paths: list[str]
) -> dict[str, tuple[str, str | None]]:
    expected = set(expected_paths)
    observations: dict[str, tuple[str, str | None]] = {}
    for line in output.splitlines():
        fields = line.split("\t")
        _require(
            fields and fields[0] in {"ok", "missing", "not_file", "unreadable"},
            "invalid probe response",
        )
        status = fields[0]
        if status == "ok":
            _require(
                len(fields) == 3 and SHA256_RE.fullmatch(fields[1]) is not None,
                "invalid digest response",
            )
            digest, path = fields[1], fields[2]
        else:
            _require(len(fields) == 2, "invalid probe response")
            digest, path = None, fields[1]
        _require(path in expected, f"probe returned unexpected path: {path}")
        _require(path not in observations, f"probe returned duplicate path: {path}")
        observations[path] = (status, digest)
    _require(set(observations) == expected, "probe omitted one or more requested paths")
    return observations


def check_worker_staging(
    manifest: Mapping[str, Any],
    worker_host: str,
    staging_root: Path,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Probe one worker and return a dispatch decision without copying sources."""
    _require(worker_host and not worker_host.startswith("-"), "worker_host is unsafe")
    _require(CONTROL_CHARACTER_RE.search(worker_host) is None, "worker_host has control character")
    sources = parse_manifest(manifest)
    paths = _remote_paths(staging_root, sources)
    completed = run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            worker_host,
            f"sh -c {shlex.quote(build_remote_probe(paths))}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return {
            "status": "blocked",
            "schema_version": SCHEMA_VERSION,
            "source_count": len(sources),
            "verified_source_count": 0,
            "dispatch_allowed": False,
            "blockers": ["worker_unreachable_or_probe_failed"],
            "probe_error": completed.stderr.strip(),
        }

    observations = _parse_probe_output(completed.stdout, paths)
    files: list[dict[str, Any]] = []
    for source, path in zip(sources, paths, strict=True):
        status, observed_sha256 = observations[path]
        if status == "ok" and observed_sha256 != source.sha256:
            status = "checksum_mismatch"
        files.append(
            {
                "relative_path": source.relative_path,
                "status": status,
                "expected_sha256": source.sha256,
                "observed_sha256": observed_sha256,
            }
        )

    verified = sum(item["status"] == "ok" for item in files)
    blockers = sorted({f"source_{item['status']}" for item in files if item["status"] != "ok"})
    return {
        "status": "ready" if not blockers else "blocked",
        "schema_version": SCHEMA_VERSION,
        "source_count": len(files),
        "verified_source_count": verified,
        "dispatch_allowed": not blockers,
        "blockers": blockers,
        "files": files,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, required=True, help="Portable source manifest JSON."
    )
    parser.add_argument(
        "--worker-host", required=True, help="SSH destination for the queued worker."
    )
    parser.add_argument(
        "--staging-root",
        type=Path,
        required=True,
        help="Absolute source root as seen by the worker (kept out of tracked manifests).",
    )
    parser.add_argument("--json", action="store_true", help="Emit the machine-readable report.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the worker-readability preflight and return 0 only when dispatch is safe."""
    args = _parser().parse_args(argv)
    try:
        report = check_worker_staging(
            _load_manifest(args.manifest), args.worker_host, args.staging_root
        )
    except ContractError as exc:
        report = {"status": "malformed", "schema_version": SCHEMA_VERSION, "error": str(exc)}

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            "cross-host source staging: "
            f"{report['status']} "
            f"({report.get('verified_source_count', 0)}/{report.get('source_count', 0)} verified)"
        )
    return 0 if report["status"] == "ready" else 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
