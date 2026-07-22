#!/usr/bin/env python3
"""Independent cold-start reproduction of benchmark release 0.0.2.

Performs the full reproduction workflow:
1. Clone at release tag
2. Build environment from lockfile
3. Verify bundle checksums against manifest
4. Run pre-registered benchmark subset
5. Generate structured reproduction report

Designed to be run on a clean non-development machine.

Usage:
    # Full reproduction (clone + build + verify + benchmark)
    python scripts/repro/cold_start_reproduction_report.py --tag 0.0.2

    # Verify checksums only (skip build + benchmark)
    python scripts/repro/cold_start_reproduction_report.py --tag 0.0.2 --checksums-only

    # Use local checkout for a diagnostic only (the report is partial, not
    # independent-reproduction evidence, because the clone step is skipped)
    python scripts/repro/cold_start_reproduction_report.py --tag 0.0.2 --local-repo /path/to/repo

    # Custom output directory
    python scripts/repro/cold_start_reproduction_report.py --tag 0.0.2 --output-dir output/repro/my_run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from scripts.repro.release_0_0_2_subset_comparison import (
    compare_subset_results,
    extract_subset_run_metrics,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit_for_path(path: Path) -> str:
    """Return the commit that last changed a tracked provenance file."""
    try:
        return subprocess.check_output(
            ["git", "log", "-1", "--format=%H", "--", str(path)],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _env_info() -> dict[str, str]:
    info: dict[str, str] = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "os_name": os.name,
    }
    try:
        info["git_version"] = subprocess.check_output(
            ["git", "--version"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        info["git_version"] = "unknown"
    try:
        info["uv_version"] = subprocess.check_output(
            ["uv", "--version"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        info["uv_version"] = "unknown"
    return info


def _load_manifest(tag: str) -> dict[str, Any]:
    tag_slug = tag.replace(".", "_")
    manifest_path = (
        REPO_ROOT / "configs" / "releases" / f"release_{tag_slug}_checksum_manifest.yaml"
    )
    if not manifest_path.exists():
        raise FileNotFoundError(f"Checksum manifest not found: {manifest_path}")
    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)
    if not isinstance(manifest, dict):
        raise ValueError("Checksum manifest root must be a mapping.")
    release_tag = manifest.get("release_tag")
    if not isinstance(release_tag, str) or not release_tag:
        raise ValueError("Checksum manifest must define a non-empty release_tag.")
    return manifest


def _step_clone(tag: str, work_dir: Path) -> dict[str, Any]:
    """Clone the repository at the release tag."""
    clone_dir = work_dir / "clone"
    result: dict[str, Any] = {"step": "clone", "tag": tag}

    try:
        subprocess.check_call(
            [
                "git",
                "clone",
                "--branch",
                tag,
                "--depth",
                "1",
                "https://github.com/ll7/robot_sf_ll7.git",
                str(clone_dir),
            ],
            timeout=300,
        )
        result["status"] = "pass"
        result["clone_dir"] = str(clone_dir)
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=clone_dir,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            result["commit"] = commit
        except (subprocess.CalledProcessError, FileNotFoundError):
            result["commit"] = "unknown"
    except subprocess.CalledProcessError as exc:
        result["status"] = "fail"
        result["error"] = str(exc)
    except subprocess.TimeoutExpired:
        result["status"] = "fail"
        result["error"] = "Clone timed out after 300s"
    return result


def _step_build(clone_dir: Path) -> dict[str, Any]:
    """Build the environment from the lockfile."""
    result: dict[str, Any] = {"step": "build"}
    start = time.monotonic()
    try:
        subprocess.check_call(
            ["uv", "sync", "--all-extras"],
            cwd=clone_dir,
            timeout=600,
        )
        elapsed = time.monotonic() - start
        result["status"] = "pass"
        result["wall_time_sec"] = round(elapsed, 2)
    except subprocess.CalledProcessError as exc:
        result["status"] = "fail"
        result["error"] = str(exc)
    except subprocess.TimeoutExpired:
        result["status"] = "fail"
        result["error"] = "Build timed out after 600s"
    return result


def _step_verify_checksums(
    clone_dir: Path,
    manifest: dict[str, Any],
    output_dir: Path,
    bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Download (unless a bundle path is supplied) and verify release bundle checksums."""
    result: dict[str, Any] = {"step": "verify_checksums"}
    bundle_info = manifest["artifact_set"]["bundle_archive"]
    bundle_name = bundle_info["name"]
    expected_sha = bundle_info["sha256"]

    bundle_dir = output_dir / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    if bundle_path is None:
        bundle_path = bundle_dir / bundle_name
        try:
            subprocess.check_call(
                [
                    "gh",
                    "release",
                    "download",
                    manifest["release_tag"],
                    "--pattern",
                    bundle_name,
                    "--dir",
                    str(bundle_dir),
                    "--clobber",
                ],
                cwd=clone_dir,
                timeout=120,
            )
        except subprocess.CalledProcessError as exc:
            result["status"] = "fail"
            result["error"] = f"Bundle download failed: {exc}"
            return result
        except subprocess.TimeoutExpired:
            result["status"] = "fail"
            result["error"] = "Bundle download timed out after 120s"
            return result
    else:
        bundle_path = Path(bundle_path)
        if not bundle_path.is_file():
            result["status"] = "fail"
            result["error"] = f"Provided bundle path is not a file: {bundle_path}"
            return result

    actual_sha = _sha256_file(bundle_path)
    result["bundle_path"] = str(bundle_path)
    result["expected_sha256"] = expected_sha
    result["actual_sha256"] = actual_sha
    result["bundle_checksum_match"] = actual_sha == expected_sha
    result["bundle_size_bytes"] = bundle_path.stat().st_size

    embedded = manifest.get("embedded_artifacts", {})
    embedded_results: list[dict[str, Any]] = []
    if embedded:
        try:
            with tarfile.open(bundle_path, "r:gz") as tar:
                for name, info in embedded.items():
                    archive_path = info["path_in_archive"]
                    expected = info["sha256"]
                    art_result: dict[str, Any] = {
                        "name": name,
                        "archive_path": archive_path,
                        "expected_sha256": expected,
                    }
                    try:
                        member = tar.getmember(archive_path)
                        f = tar.extractfile(member)
                        if f is not None:
                            actual = hashlib.sha256(f.read()).hexdigest()
                            art_result["actual_sha256"] = actual
                            art_result["match"] = actual == expected
                        else:
                            art_result["match"] = False
                            art_result["error"] = "Could not read from archive"
                    except KeyError:
                        art_result["match"] = False
                        art_result["error"] = "Not found in archive"
                    embedded_results.append(art_result)
        except tarfile.TarError as exc:
            result["embedded_verification_error"] = str(exc)

    result["embedded_artifacts"] = embedded_results
    all_match = result["bundle_checksum_match"] and all(
        r.get("match", False) for r in embedded_results
    )
    result["status"] = "pass" if all_match else "fail"
    return result


def _step_run_subset(clone_dir: Path, manifest: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Run the pre-registered benchmark subset in run mode."""
    result: dict[str, Any] = {"step": "run_subset"}
    campaign = manifest.get("campaign", {})
    planners = manifest.get("planners", [])
    seed_policy = manifest.get("seed_policy", {})

    result["planners"] = planners
    result["seed_policy"] = seed_policy
    result["campaign_id"] = campaign.get("campaign_id")

    smoke_config = Path(
        "configs/benchmarks/releases/paper_experiment_matrix_v1_release_smoke_v0_1.yaml"
    )
    if not (clone_dir / smoke_config).exists():
        result["status"] = "fail"
        result["error"] = f"Smoke manifest not found: {smoke_config}"
        return result

    subset_run_dir = output_dir / "subset_run"
    subset_run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "uv",
        "run",
        "python",
        "scripts/tools/run_benchmark_release.py",
        "--manifest",
        str(smoke_config),
        "--mode",
        "run",
        "--output-root",
        str(subset_run_dir),
    ]
    result["command"] = shlex.join(cmd)
    start = time.monotonic()
    try:
        proc_out = subprocess.check_output(
            cmd,
            cwd=clone_dir,
            text=True,
            timeout=600,
        )
        elapsed = time.monotonic() - start
        result["wall_time_sec"] = round(elapsed, 2)
        try:
            run_payload = json.loads(proc_out)
            result["run_payload"] = run_payload
            result["campaign_root"] = run_payload.get("campaign_root")
            if run_payload.get("mode") != "run":
                result["status"] = "fail"
                result["error"] = (
                    f"Subset replay executed in preflight mode instead of run mode "
                    f"(mode={run_payload.get('mode')})"
                )
                return result
            if run_payload.get("campaign_execution_status") == "failed":
                result["status"] = "fail"
                result["error"] = (
                    f"Subset replay execution failed: {run_payload.get('status_reason')}"
                )
                return result
        except json.JSONDecodeError:
            pass

        result["status"] = "pass"
    except subprocess.CalledProcessError as exc:
        result["status"] = "fail"
        result["error"] = f"Subset run command failed: {exc}"
    except subprocess.TimeoutExpired:
        result["status"] = "fail"
        result["error"] = "Subset run timed out after 600s"
    return result


def _step_compare_subset(
    clone_dir: Path,
    manifest: dict[str, Any],
    output_dir: Path,
    run_subset_result: dict[str, Any],
) -> dict[str, Any]:
    """Compare numeric replay outcome against frozen release contract."""
    result: dict[str, Any] = {"step": "compare_subset"}
    if run_subset_result.get("status") != "pass":
        result["status"] = "fail"
        result["error"] = (
            f"Cannot compare subset: run_subset step status is {run_subset_result.get('status')}"
        )
        return result

    campaign_root_str = run_subset_result.get("campaign_root")
    if campaign_root_str and Path(campaign_root_str).is_dir():
        campaign_root = Path(campaign_root_str)
    else:
        subset_run_dir = output_dir / "subset_run"
        candidates = [d for d in subset_run_dir.glob("*") if d.is_dir()]
        if candidates:
            campaign_root = candidates[0]
        else:
            campaign_root = subset_run_dir

    extracted = extract_subset_run_metrics(campaign_root)
    if extracted.get("status") != "pass":
        result["status"] = "fail"
        result["error"] = extracted.get("error", "Failed to extract replay metrics")
        return result

    comp_res = compare_subset_results(extracted, manifest)
    result["status"] = comp_res["status"]
    result["overall_match"] = comp_res["overall_match"]
    result["comparison_rows"] = comp_res["comparison_rows"]
    result["global_deviations"] = comp_res["global_deviations"]
    return result


def generate_reproduction_report(
    tag: str,
    output_dir: Path,
    local_repo: Path | None = None,
    checksums_only: bool = False,
    bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Generate the full cold-start reproduction report.

    Args:
        bundle_path: Optional pre-downloaded bundle archive. When supplied, the
            verify_checksums step reuses it instead of downloading again, which
            makes the flow resilient to a previously downloaded artifact in the
            output directory and removes a redundant network call.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.monotonic()

    manifest = _load_manifest(tag)
    manifest_path = (
        REPO_ROOT
        / "configs"
        / "releases"
        / f"release_{tag.replace('.', '_')}_checksum_manifest.yaml"
    )
    env = _env_info()

    report: dict[str, Any] = {
        "schema": "cold-start-reproduction-report.v1",
        "review_marker": "AI-GENERATED NEEDS-REVIEW",
        "created_at_utc": _utc_now_iso(),
        "release_tag": tag,
        "release_id": manifest.get("release_id"),
        "target_commit": manifest.get("target_commit"),
        "tooling_commit": _git_commit_for_path(REPO_ROOT),
        "config_path": str(manifest_path),
        "config_sha256": _sha256_file(manifest_path),
        "config_commit": _git_commit_for_path(manifest_path),
        "lockfile_sha256": (
            _sha256_file(REPO_ROOT / "uv.lock") if (REPO_ROOT / "uv.lock").is_file() else "unknown"
        ),
        "environment": env,
        "steps": {},
        "instruction_gaps": [],
        "deviations": [],
        "limitations": [
            "This report was generated by automated tooling and may require "
            "manual verification on the target machine.",
            "Near-miss metrics are nondeterministic due to upstream pedestrian "
            "dynamics (see manifest reproducibility_contract).",
        ],
    }

    if local_repo:
        clone_dir = local_repo
        report["steps"]["clone"] = {
            "step": "clone",
            "status": "skip",
            "reason": f"Using local repo: {local_repo}",
        }
        reason = (
            "Independent reproduction is incomplete: the clean release-tag clone was skipped "
            "because --local-repo was used."
        )
        report["deviations"].append(reason)
        report["instruction_gaps"].append(
            "Run the workflow from a clean non-development machine/person without --local-repo."
        )
    else:
        clone_step = _step_clone(tag, output_dir / "workspace")
        report["steps"]["clone"] = clone_step
        if clone_step["status"] != "pass":
            report["overall_verdict"] = "fail"
            report["total_wall_time_sec"] = round(time.monotonic() - start_time, 2)
            return report
        clone_dir = Path(clone_step["clone_dir"])

    report["steps"]["verify_checksums"] = _step_verify_checksums(
        clone_dir,
        manifest,
        output_dir,
        bundle_path=bundle_path,
    )

    if not checksums_only:
        report["steps"]["build"] = _step_build(clone_dir)
        run_step = _step_run_subset(clone_dir, manifest, output_dir)
        report["steps"]["run_subset"] = run_step
        report["steps"]["compare_subset"] = _step_compare_subset(
            clone_dir, manifest, output_dir, run_step
        )

    statuses = [s.get("status", "skip") for s in report["steps"].values()]
    if any(s == "fail" for s in statuses):
        report["overall_verdict"] = "fail"
    elif all(s == "pass" for s in statuses):
        report["overall_verdict"] = "pass"
    else:
        report["overall_verdict"] = "partial"

    report["total_wall_time_sec"] = round(time.monotonic() - start_time, 2)
    return report


def main() -> None:
    """Run cold-start reproduction report generation from CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="0.0.2", help="Release tag")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/cold_start_reproduction"),
    )
    parser.add_argument("--local-repo", type=Path, help="Use local repo instead of cloning")
    parser.add_argument(
        "--checksums-only",
        action="store_true",
        help="Only verify checksums (skip build and benchmark)",
    )
    parser.add_argument(
        "--bundle-path",
        type=Path,
        help="Use a pre-downloaded bundle archive instead of downloading again",
    )
    args = parser.parse_args()

    report = generate_reproduction_report(
        tag=args.tag,
        output_dir=args.output_dir,
        local_repo=args.local_repo,
        checksums_only=args.checksums_only,
        bundle_path=args.bundle_path,
    )

    report_path = args.output_dir / "reproduction_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nVERDICT: {report['overall_verdict'].upper()}", file=sys.stderr)
    print(f"Report: {report_path}", file=sys.stderr)

    if report["overall_verdict"] == "fail":
        sys.exit(1)


if __name__ == "__main__":
    main()
