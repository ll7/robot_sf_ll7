#!/usr/bin/env python3
"""Validate the durable h600 source-report acquisition contract for issue #5164."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

DEFAULT_MANIFEST = Path(
    "docs/context/evidence/issue_3810_h600_interpretation_2026-07/h600_source_reports_manifest.json"
)
SCHEMA_VERSION = "issue_5164_h600_source_reports.v1"
EXPECTED_FILES = frozenset(
    {"scenario_breakdown.csv", "scenario_family_breakdown.csv", "seed_episode_rows.csv"}
)
EXPECTED_RUNS = {
    "13268": {
        "run_label": "confirm",
        "campaign_id": "issue3810_h600_longhorizon_confirm_run_20260702",
        "source_git_hash": "1cb7dc31a018fdf21892beb0e74ca47699e41d9a",
        "reports_dir": (
            "docs/context/evidence/issue_3810_h600_interpretation_2026-07/source_reports/13268"
        ),
        "sha256": {
            "scenario_breakdown.csv": (
                "bab81ea5ed6517c7537ff41954b4b2ecefc2dc84754c6f43efca8d36f9bac2a1"
            ),
            "scenario_family_breakdown.csv": (
                "f2d7e3128eb956721a2724a2d8d9ca9645eced6df1a3219f1e3a4e597f8965b9"
            ),
            "seed_episode_rows.csv": (
                "4493c8ebee0942c1c32b0d51e511f4fd98f35ee2be8d24d366564f8979be7d23"
            ),
        },
    },
    "13273": {
        "run_label": "extended_roster",
        "campaign_id": "issue3810_h600_extroster_run_20260702",
        "source_git_hash": "4da0879fa897783fe35f65d52ef488d14e526ccc",
        "reports_dir": (
            "docs/context/evidence/issue_3810_h600_interpretation_2026-07/source_reports/13273"
        ),
        "sha256": {
            "scenario_breakdown.csv": (
                "49ef88f08ccc66e73f01cf6221a52fd685a6ca9d3aa4f1179aa0ca1532827e1f"
            ),
            "scenario_family_breakdown.csv": (
                "1a925a9fc4f0cecba834573597e1c8995a350a5b3454fec0cb4280256eb4a156"
            ),
            "seed_episode_rows.csv": (
                "f2e5960c2439f28745b4bd2f7cd19f4c94cabd86cb1387693095a656ac859e81"
            ),
        },
    },
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ContractError(ValueError):
    """Raised when the acquisition manifest itself is unsafe or malformed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot load manifest: {exc}") from exc
    _require(isinstance(payload, dict), "manifest must be a JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_source_reports(
    manifest: Mapping[str, Any], repo_root: Path | None = None
) -> dict[str, Any]:
    """Validate the manifest and classify all six durable source reports."""
    root = (repo_root or _repo_root()).resolve()
    _require(manifest.get("schema_version") == SCHEMA_VERSION, "schema_version mismatch")
    _require(manifest.get("issue") == 5164, "issue must be 5164")
    _require(manifest.get("contract_status") == "active", "contract_status must be active")

    policy = _mapping(manifest.get("artifact_policy"), "artifact_policy")
    _require(policy.get("local_output_is_durable") is False, "output/ must not be durable")
    _require(policy.get("copy_without_transformation") is True, "sources must be copied verbatim")

    runs = manifest.get("required_runs")
    _require(isinstance(runs, list), "required_runs must be a list")
    _require(len(runs) == len(EXPECTED_RUNS), "required_runs must contain exactly two runs")

    observations: list[dict[str, Any]] = []
    seen_jobs: set[str] = set()
    for run_index, raw_run in enumerate(runs):
        run = _mapping(raw_run, f"required_runs[{run_index}]")
        job_id = str(run.get("job_id", ""))
        _require(job_id in EXPECTED_RUNS, f"unexpected job_id: {job_id}")
        _require(job_id not in seen_jobs, f"duplicate job_id: {job_id}")
        seen_jobs.add(job_id)

        expected_run = EXPECTED_RUNS[job_id]
        for field in ("run_label", "campaign_id", "source_git_hash"):
            _require(run.get(field) == expected_run[field], f"{job_id}: {field} mismatch")
        reports_dir = run.get("reports_dir")
        _require(reports_dir == expected_run["reports_dir"], f"{job_id}: reports_dir mismatch")
        reports_path = Path(str(reports_dir))
        _require(not reports_path.is_absolute(), f"{job_id}: reports_dir must be repo-relative")
        _require("output" not in reports_path.parts, f"{job_id}: reports_dir cannot use output/")
        resolved_reports = (root / reports_path).resolve()
        _require(
            resolved_reports.is_relative_to(root), f"{job_id}: reports_dir escapes repository root"
        )

        files = _mapping(run.get("files"), f"{job_id}.files")
        _require(set(files) == EXPECTED_FILES, f"{job_id}: files must match required report set")
        for filename in sorted(EXPECTED_FILES):
            metadata = _mapping(files[filename], f"{job_id}.{filename}")
            expected_digest = str(metadata.get("sha256", ""))
            _require(SHA256_RE.fullmatch(expected_digest) is not None, "invalid SHA-256 digest")
            _require(
                expected_digest == expected_run["sha256"][filename],
                f"{job_id}/{filename}: recorded SHA-256 changed",
            )

            path = resolved_reports / filename
            if not path.is_file():
                observations.append(
                    {
                        "job_id": job_id,
                        "file": filename,
                        "path": str(reports_path / filename),
                        "status": "missing",
                        "expected_sha256": expected_digest,
                    }
                )
                continue

            observed_digest = _sha256(path)
            observations.append(
                {
                    "job_id": job_id,
                    "file": filename,
                    "path": str(reports_path / filename),
                    "status": "ok" if observed_digest == expected_digest else "checksum_mismatch",
                    "expected_sha256": expected_digest,
                    "observed_sha256": observed_digest,
                }
            )

    _require(seen_jobs == set(EXPECTED_RUNS), "required job set mismatch")
    missing = sum(item["status"] == "missing" for item in observations)
    mismatched = sum(item["status"] == "checksum_mismatch" for item in observations)
    ready = missing == 0 and mismatched == 0
    return {
        "status": "ready" if ready else "blocked",
        "issue": 5164,
        "schema_version": SCHEMA_VERSION,
        "required_file_count": len(observations),
        "verified_file_count": len(observations) - missing - mismatched,
        "missing_file_count": missing,
        "checksum_mismatch_count": mismatched,
        "downstream_export_allowed": ready,
        "files": observations,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Manifest path; defaults to the tracked contract relative to the repository root.",
    )
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable report.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Check the source-report contract; return nonzero until all six files verify."""
    args = _parser().parse_args(argv)
    manifest_path = args.manifest or _repo_root() / DEFAULT_MANIFEST
    try:
        report = validate_source_reports(_load_manifest(manifest_path))
    except ContractError as exc:
        report = {"status": "malformed", "issue": 5164, "error": str(exc)}
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(f"issue #5164 source-report contract malformed: {exc}")
        return 2

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            "issue #5164 h600 source reports: "
            f"{report['status']} "
            f"({report['verified_file_count']}/{report['required_file_count']} verified, "
            f"{report['missing_file_count']} missing, "
            f"{report['checksum_mismatch_count']} mismatched)"
        )
    return 0 if report["status"] == "ready" else 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
