#!/usr/bin/env python3
"""Hydrate release 0.0.3.post1, adapt episode rows, and execute issue #5351 analysis.

This script implements the end-to-end pipeline for GitHub issue #5351:
1. Hydrates and verifies the release 0.0.3.post1 publication bundle.
2. Verifies outer archive SHA-256 digest, traversal structure, and exact 20,160 row count.
3. Reshapes nested episode records into EpisodeEventLedger.v2 rows without changing metric semantics.
4. Registers the checksum-pinned successor rows under docs/context/evidence/issue_5351_hierarchical_paired_release_analysis/successor_rows.jsonl.
5. Updates configs/benchmarks/releases/hierarchical_paired_release_analysis_issue_5351.yaml.
6. Executes the frozen hierarchical paired release analysis and registers deterministic reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.hierarchical_paired_release_analysis import (
    CLAIM_GATE_BLOCKED_REVIEW_PENDING,
    AnalysisPolicy,
    run_hierarchical_paired_release_analysis,
)
from robot_sf.benchmark.hierarchical_paired_release_inputs import (
    load_hierarchical_paired_release_input_manifest,
)
from robot_sf.errors import RobotSfError

EXPECTED_BUNDLE_SHA256 = "9bf6ea35a17ce812f0a9c841c3681bc072dcf7ba8c121cbcf05113b8514f4de1"
EXPECTED_RELEASE_TAG = "0.0.3.post1"
EXPECTED_PUBLICATION_COMMIT = "ded9027d2928512c14bc241397e0ab1d8f586654"
EXPECTED_TOTAL_EPISODES = 20160
EXPECTED_ARMS_COUNT = 14
EXPECTED_ROWS_PER_ARM = 1440
DEFAULT_HORIZON = 600.0

DEFAULT_MANIFEST_PATH = (
    "configs/benchmarks/releases/hierarchical_paired_release_analysis_issue_5351.yaml"
)
DEFAULT_EVIDENCE_DIR = "docs/context/evidence/issue_5351_hierarchical_paired_release_analysis"


class ReleaseAnalysisPipelineError(RobotSfError, ValueError):
    """Raised when release hydration, adaptation, or execution fails closed."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def find_or_download_bundle(bundle_path: Path | None, *, repo_root: Path) -> Path:
    """Locate or download release 0.0.3.post1 bundle and verify its digest.

    Returns:
        Path to the verified release tarball.
    """
    candidates = []
    if bundle_path is not None:
        candidates.append(bundle_path)
    candidates.extend(
        [
            Path(
                "/tmp/paper_experiment_matrix_v2_h600_s30_extended_release_v0_0_3_post1_corrected_publication_bundle.tar.gz"
            ),
            repo_root
            / "paper_experiment_matrix_v2_h600_s30_extended_release_v0_0_3_post1_corrected_publication_bundle.tar.gz",
        ]
    )

    existing = next((c for c in candidates if c.is_file()), None)
    if existing is None:
        target = Path(
            "/tmp/paper_experiment_matrix_v2_h600_s30_extended_release_v0_0_3_post1_corrected_publication_bundle.tar.gz"
        )
        print(f"Downloading release {EXPECTED_RELEASE_TAG} bundle via gh...")
        cmd = [
            "gh",
            "release",
            "download",
            EXPECTED_RELEASE_TAG,
            "--repo",
            "ll7/robot_sf_ll7",
            "-p",
            "paper_experiment_matrix_v2_h600_s30_extended_release_v0_0_3_post1_corrected_publication_bundle.tar.gz",
            "--dir",
            str(target.parent),
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(res.stdout)
        except (subprocess.SubprocessError, OSError) as exc:
            raise ReleaseAnalysisPipelineError(
                f"Could not download release {EXPECTED_RELEASE_TAG} bundle: {exc}"
            ) from exc
        existing = target

    actual_sha256 = sha256_file(existing)
    if actual_sha256 != EXPECTED_BUNDLE_SHA256:
        raise ReleaseAnalysisPipelineError(
            f"Release archive SHA-256 digest mismatch: expected {EXPECTED_BUNDLE_SHA256!r}, "
            f"got {actual_sha256!r} for {existing}"
        )
    return existing


def adapt_record_to_typed_ledger(
    record: dict[str, Any], *, planner_name: str
) -> tuple[dict[str, Any], str]:
    """Adapt one raw episode record into an EpisodeEventLedger.v2 row.

    Returns:
        (adapted_ledger_row, scenario_archetype)
    """
    scenario_id = str(record.get("scenario_id") or "unknown")
    seed = int(record.get("seed", 0))

    event_ledger = record.get("event_ledger") or {}
    exact_in = event_ledger.get("exact_events") or {}
    outcome_in = record.get("outcome") or {}

    collision = bool(exact_in.get("collision") or outcome_in.get("collision_event"))
    timeout = bool(exact_in.get("timeout") or outcome_in.get("timeout_event"))
    goal_reached = bool(exact_in.get("goal_reached") or outcome_in.get("route_complete"))
    invalid_run = bool(exact_in.get("invalid_run", False))

    metrics = record.get("metrics") or {}
    surrogate_in = event_ledger.get("surrogate_events") or {}
    near_miss = bool(surrogate_in.get("near_miss") or (metrics.get("near_misses", 0) > 0))

    steps = int(record.get("steps", 0))
    run_dt = float(record.get("scenario_params", {}).get("run_dt", 0.1))
    completion_time = max(steps * run_dt, 0.1)

    path_len = float(
        metrics.get("socnavbench_path_length", 0.0) or metrics.get("path_length", 0.0) or 0.0
    )
    distance_exposure = max(path_len, 0.001)

    archetype = str(
        record.get("scenario_params", {}).get("metadata", {}).get("archetype") or scenario_id
    )

    ledger_row = {
        "schema_version": "EpisodeEventLedger.v2",
        "scenario_id": scenario_id,
        "seed": seed,
        "planner": planner_name,
        "exact_events": {
            "collision": collision,
            "goal_reached": goal_reached,
            "timeout": timeout,
            "invalid_run": invalid_run,
        },
        "surrogate_events": {
            "near_miss": near_miss,
        },
        "provenance": {
            "completion_time": completion_time,
            "exposure": {
                "time": completion_time,
                "distance": distance_exposure,
                "opportunity": 1.0,
            },
        },
    }
    return ledger_row, archetype


def _parse_member_episodes(
    f: Any,
    *,
    member_name: str,
    planner_name: str,
    seen_cells: set[tuple[str, int, str]],
    rows: list[dict[str, Any]],
    family_of: dict[str, str],
) -> int:
    """Parse one arm's episodes stream and populate rows and family_of.

    Returns:
        The number of episode rows parsed.
    """
    arm_count = 0
    for line_no, line in enumerate(f, start=1):
        raw_str = line.decode("utf-8").strip()
        if not raw_str:
            continue
        try:
            record = json.loads(raw_str)
        except json.JSONDecodeError as exc:
            raise ReleaseAnalysisPipelineError(
                f"Corrupt JSON line {line_no} in {member_name}: {exc}"
            ) from exc

        row, archetype = adapt_record_to_typed_ledger(record, planner_name=planner_name)
        cell_key = (row["scenario_id"], row["seed"], row["planner"])
        if cell_key in seen_cells:
            raise ReleaseAnalysisPipelineError(f"Duplicate cell found in archive for {cell_key}")
        seen_cells.add(cell_key)
        family_of[row["scenario_id"]] = archetype
        rows.append(row)
        arm_count += 1
    return arm_count


def hydrate_and_adapt_release_bundle(
    tar_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Traverse the verified archive, validate rows, and return adapted ledger rows.

    Returns:
        (adapted_rows, family_of_mapping)
    """
    rows: list[dict[str, Any]] = []
    family_of: dict[str, str] = {}
    per_arm_counts: dict[str, int] = {}
    seen_cells: set[tuple[str, int, str]] = set()

    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar.getmembers():
                if not (
                    "episodes.jsonl" in member.name and not member.name.endswith(".provenance.json")
                ):
                    continue
                arm_dir_name = member.name.split("/")[-2]
                planner_name = arm_dir_name.removesuffix("__differential_drive")
                f = tar.extractfile(member)
                if f is None:
                    raise ReleaseAnalysisPipelineError(
                        f"Could not extract file {member.name} from archive"
                    )
                arm_count = _parse_member_episodes(
                    f,
                    member_name=member.name,
                    planner_name=planner_name,
                    seen_cells=seen_cells,
                    rows=rows,
                    family_of=family_of,
                )
                per_arm_counts[planner_name] = arm_count
    except (tarfile.TarError, OSError) as exc:
        raise ReleaseAnalysisPipelineError(
            f"Failed traversing release archive {tar_path}: {exc}"
        ) from exc

    if len(per_arm_counts) != EXPECTED_ARMS_COUNT:
        raise ReleaseAnalysisPipelineError(
            f"Expected {EXPECTED_ARMS_COUNT} arms in archive, found {len(per_arm_counts)}: "
            f"{sorted(per_arm_counts)}"
        )
    for planner_name, count in per_arm_counts.items():
        if count != EXPECTED_ROWS_PER_ARM:
            raise ReleaseAnalysisPipelineError(
                f"Arm {planner_name!r} has {count} rows, expected {EXPECTED_ROWS_PER_ARM}"
            )
    if len(rows) != EXPECTED_TOTAL_EPISODES:
        raise ReleaseAnalysisPipelineError(
            f"Total row count mismatch: expected {EXPECTED_TOTAL_EPISODES}, got {len(rows)}"
        )

    return rows, family_of


def write_successor_rows_and_update_manifest(
    rows: list[dict[str, Any]],
    *,
    evidence_dir: Path,
    manifest_path: Path,
    repo_root: Path,
) -> tuple[Path, str]:
    """Write deterministic successor rows JSONL and update the manifest.

    Returns:
        (rows_path, rows_sha256)
    """
    evidence_dir.mkdir(parents=True, exist_ok=True)
    rows_path = evidence_dir / "successor_rows.jsonl"

    sorted_rows = sorted(rows, key=lambda r: (r["planner"], r["scenario_id"], r["seed"]))
    serialized = "\n".join(json.dumps(r, sort_keys=True) for r in sorted_rows) + "\n"
    rows_path.write_text(serialized, encoding="utf-8")
    rows_sha256 = sha256_file(rows_path)

    relative_rows_path = rows_path.relative_to(repo_root).as_posix()

    manifest_content = manifest_path.read_text(encoding="utf-8")
    manifest = yaml.safe_load(manifest_content)

    manifest["successor_release"] = {
        "release_tag": EXPECTED_RELEASE_TAG,
        "commit": EXPECTED_PUBLICATION_COMMIT,
        "typed_ledger_rows": relative_rows_path,
        "typed_ledger_rows_sha256": rows_sha256,
    }

    updated_yaml = "# Fail-closed prerequisite manifest for issue #5351.\n" + yaml.safe_dump(
        manifest, sort_keys=False
    )
    manifest_path.write_text(updated_yaml, encoding="utf-8")

    return rows_path, rows_sha256


def render_evidence_readme(
    report: dict[str, Any],
    *,
    evidence_dir: Path,
    rows_sha256: str,
    relative_rows_path: str,
) -> Path:
    """Render a markdown summary of the statistical analysis and protocol conformance.

    Returns:
        Path to README.md in evidence_dir.
    """
    readme_path = evidence_dir / "README.md"

    conformance_rows = report.get("protocol_conformance", [])
    conformance_table = [
        "| Protocol Element | Declared Delivery | Status |",
        "| --- | --- | --- |",
    ]
    for row in conformance_rows:
        conformance_table.append(
            f"| `{row['id']}` | {row['declared_delivery']} | `{row['status']}` |"
        )

    paired_effects = report.get("paired_effects", [])
    total_comparisons = len(paired_effects)
    separable_count = sum(
        1
        for p in paired_effects
        if p.get("practical_effect", {}).get("verdict") == "practically_separable"
    )

    readme_content = f"""<!-- AI-GENERATED (robot_sf#5351, {time.strftime("%Y-%m-%d")}) - NEEDS-REVIEW -->
# Issue #5351 Hierarchical Paired Release Analysis Report

This directory registers the deterministic, checksum-pinned statistical analysis artifacts for issue #5351 over benchmark release `{EXPECTED_RELEASE_TAG}`.

> [!NOTE]
> Claim boundary: this is statistical analysis ON TOP of frozen release metrics. Output remains `blocked_review_pending` and promotes no benchmark, paper, or dissertation claim automatically.

## Successor Release Inputs

- Release Tag: `{EXPECTED_RELEASE_TAG}`
- Publication Commit: `{EXPECTED_PUBLICATION_COMMIT}`
- Typed Ledger Rows: [`{relative_rows_path}`]({relative_rows_path})
- Rows SHA-256: `{rows_sha256}`
- Total Episode Rows: `{report.get("paired_effects", [{}])[0].get("n_cells", 1440) * 14 if report.get("paired_effects") else 20160}` (14 arms × 1,440 episodes)

## Protocol Conformance

{chr(10).join(conformance_table)}

## Summary of Analysis

- Total Paired Comparisons Evaluated: {total_comparisons}
- Multiplicity Method: {report.get("multiplicity", {}).get("method", "holm_step_down")}
- Practically Separable Effects (Clear min_risk_difference >= 0.02): {separable_count} / {total_comparisons}
- Claim Gate Status: `{report.get("claim_gate", {}).get("status")}`
- Claim Gate Reason: {report.get("claim_gate", {}).get("reason")}

## Reproducibility

To re-run and verify determinism against release `{EXPECTED_RELEASE_TAG}`:

```bash
uv run python scripts/analysis/run_hierarchical_paired_release_analysis_issue_5351.py \\
  --repo-root .
```
"""
    readme_path.write_text(readme_content, encoding="utf-8")
    return readme_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-tar",
        type=Path,
        default=None,
        help="Path to release 0.0.3.post1 publication bundle tar.gz.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=f"Path to issue #5351 manifest YAML (default: {DEFAULT_MANIFEST_PATH}).",
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=None,
        help=f"Directory for registered evidence outputs (default: {DEFAULT_EVIDENCE_DIR}).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root directory.",
    )
    parser.add_argument(
        "--all-pairs",
        action="store_true",
        help="Evaluate all 91 pairwise planner combinations instead of baseline-only pairs.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Execute the full #5351 hydration, adaptation, and analysis pipeline."""
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    manifest_path = (args.manifest or (repo_root / DEFAULT_MANIFEST_PATH)).resolve()
    evidence_dir = (args.evidence_dir or (repo_root / DEFAULT_EVIDENCE_DIR)).resolve()

    print("--- Issue #5351 Hierarchical Paired Release Analysis Pipeline ---")
    t0 = time.time()

    bundle_tar = find_or_download_bundle(args.bundle_tar, repo_root=repo_root)
    print(f"Verified release bundle archive: {bundle_tar}")

    rows, family_of = hydrate_and_adapt_release_bundle(bundle_tar)
    print(f"Adapted {len(rows)} episode rows across {len({r['planner'] for r in rows})} planners.")

    rows_path, rows_sha256 = write_successor_rows_and_update_manifest(
        rows,
        evidence_dir=evidence_dir,
        manifest_path=manifest_path,
        repo_root=repo_root,
    )
    print(
        f"Registered successor rows at {rows_path.relative_to(repo_root)} (SHA-256: {rows_sha256[:16]}...)"
    )

    manifest = load_hierarchical_paired_release_input_manifest(manifest_path)

    planners = sorted({r["planner"] for r in rows})
    if args.all_pairs:
        planner_pairs = list(combinations(planners, 2))
    else:
        planner_pairs = [(p, "orca") for p in planners if p != "orca"]

    print(f"Executing hierarchical analysis over {len(planner_pairs)} planner pairs...")
    report = run_hierarchical_paired_release_analysis(
        manifest,
        repo_root=repo_root,
        successor_rows=rows,
        planner_pairs=planner_pairs,
        family_of=family_of,
        horizon=DEFAULT_HORIZON,
        policy=AnalysisPolicy(),
    )

    report_path = evidence_dir / "hierarchical_paired_release_analysis_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    relative_rows_path = rows_path.relative_to(repo_root).as_posix()
    readme_path = render_evidence_readme(
        report,
        evidence_dir=evidence_dir,
        rows_sha256=rows_sha256,
        relative_rows_path=relative_rows_path,
    )

    t1 = time.time()
    print(f"Pipeline finished in {t1 - t0:.2f}s.")
    print(f"Wrote report: {report_path.relative_to(repo_root)}")
    print(f"Wrote summary: {readme_path.relative_to(repo_root)}")
    print(f"Claim Gate Status: {report['claim_gate']['status']}")

    return 0 if report["claim_gate"]["status"] == CLAIM_GATE_BLOCKED_REVIEW_PENDING else 1


if __name__ == "__main__":
    sys.exit(main())
