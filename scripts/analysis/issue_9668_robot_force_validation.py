"""Build an arm-aware robot-force validation report from an accepted release matrix.

The predecessor metric-equivalence gate must pass first. Source episode records
are read one at a time; only scalar analysis inputs and bounded trace summaries
remain in memory. The report is descriptive simulator evidence, not a measure of
human discomfort or a substitute for the original episode JSONL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.analysis.issue_9666_robot_force_validation import (
    COMPARATORS,
    FORCES,
    analyze,
    metric_value,
    posthoc_discomfort,
    trace_evidence,
)

BASELINE_ARCHIVE_SHA256 = "684da7c557c426756f22ddbf5cb3270141ee8ae385669a39d36f324852a6fb2f"
BASELINE_SOURCE_SHA = "07f7e8d43084de748915e1b1eb8b2a1603357c6e"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_equivalence(path: Path, expected_source: str, expected_rows: int) -> str:
    """Require the complete old-metric and force-shape gates before release reporting."""
    digest = _sha256(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(report, dict)
        or report.get("status") != "pass"
        or report.get("candidate_source_sha") != expected_source
        or report.get("baseline_archive_sha256") != BASELINE_ARCHIVE_SHA256
        or report.get("baseline_source_sha") != BASELINE_SOURCE_SHA
        or report.get("expected_rows") != expected_rows
        or report.get("candidate_rows") != expected_rows
        or report.get("baseline_rows") != expected_rows
        or report.get("mismatch_episodes") != 0
        or report.get("scientific_manifest_differences") != []
        or not isinstance(report.get("robot_force_metrics"), dict)
        or report["robot_force_metrics"].get("status") != "pass"
        or report["robot_force_metrics"].get("checked_rows") != expected_rows
    ):
        raise ValueError("robot-force release report requires a complete passing equivalence gate")
    return digest


def _require_source(row: dict[str, Any], expected_source: str) -> None:
    provenance = row.get("result_provenance")
    source = provenance.get("repo_commit") if isinstance(provenance, dict) else None
    git_hash = row.get("git_hash")
    if source is not None and git_hash is not None and source != git_hash:
        raise ValueError("episode has conflicting source commits")
    if (source or git_hash) != expected_source:
        raise ValueError("episode source commit differs from equivalence report")


def _compact_episode(row: dict[str, Any], arm: str) -> tuple[dict[str, Any], bool]:
    """Keep only scalar inputs and true arm identity for the cohort analysis."""
    discomfort = metric_value(row, "human_discomfort_exposure_m_s")
    derived = False
    if discomfort is None:
        discomfort = posthoc_discomfort(row)
        derived = discomfort is not None
    source_metrics = row.get("metrics")
    if not isinstance(source_metrics, dict):
        raise ValueError("episode has no metrics object")
    metric_keys = set(FORCES) | set(COMPARATORS) | {"robot_force_exposed_ped_count"}
    metrics = {key: metric_value(row, key) for key in metric_keys}
    metrics["human_discomfort_exposure_m_s"] = discomfort
    compact = {
        "scenario_id": row["scenario_id"],
        "seed": row["seed"],
        "algo": arm,
        "steps": row["steps"],
        "metrics": metrics,
        "scenario_params": {
            "metadata": {
                "archetype": row.get("scenario_params", {}).get("metadata", {}).get("archetype")
            }
        },
    }
    return compact, derived


def _read_arm(
    path: Path,
    expected_source: str,
    seen: set[tuple[str, str, int]],
    locations: dict[tuple[str, str, int], tuple[Path, int]],
    artifact_location: str | None,
) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
    """Read and checksum one arm while projecting its large episode records."""
    arm = path.parent.name.removesuffix("__differential_drive")
    rows: list[dict[str, Any]] = []
    algorithms: set[str] = set()
    posthoc_rows = 0
    file_digest = hashlib.sha256()
    with path.open("rb") as stream:
        line_number = 0
        while raw := stream.readline():
            line_number += 1
            offset = stream.tell() - len(raw)
            file_digest.update(raw)
            if not raw.strip():
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: episode is not an object")
            _require_source(row, expected_source)
            scenario, seed = row.get("scenario_id"), row.get("seed")
            if not isinstance(scenario, str) or not scenario or type(seed) is not int:
                raise ValueError(f"{path}:{line_number}: invalid episode identity")
            key = arm, scenario, seed
            if key in seen:
                raise ValueError(f"{path}:{line_number}: duplicate arm/scenario/seed")
            seen.add(key)
            compact, derived = _compact_episode(row, arm)
            rows.append(compact)
            locations[key] = path, offset
            posthoc_rows += derived
            algorithm = row.get("algo")
            if not isinstance(algorithm, str) or not algorithm:
                raise ValueError(f"{path}:{line_number}: missing algorithm identity")
            algorithms.add(algorithm)
    stable_sha = _sha256(path)
    if stable_sha != file_digest.hexdigest():
        raise ValueError(f"episode artifact changed while it was analyzed: {path}")
    artifact_path = f"runs/{path.parent.name}/episodes.jsonl"
    source = {
        "artifact_path": artifact_path,
        "location": (
            f"{artifact_location.rstrip('/')}/{artifact_path}"
            if artifact_location
            else artifact_path
        ),
        "sha256": stable_sha,
        "rows": len(rows),
        "algorithms": sorted(algorithms),
    }
    return rows, posthoc_rows, source


def _selected_trace_evidence(
    location: tuple[Path, int], key: tuple[str, str, int], expected_source: str
) -> dict[str, Any]:
    """Read only a selected disagreement's original recorded force samples."""
    path, offset = location
    with path.open("rb") as stream:
        stream.seek(offset)
        row = json.loads(stream.readline())
    if not isinstance(row, dict):
        raise ValueError(f"selected episode at {path}:{offset} is not an object")
    _require_source(row, expected_source)
    if (
        path.parent.name.removesuffix("__differential_drive"),
        row.get("scenario_id"),
        row.get("seed"),
    ) != key:
        raise ValueError(f"selected episode identity changed at {path}:{offset}")
    return trace_evidence(row)


def build_report(
    campaign_root: Path,
    *,
    expected_source: str,
    expected_rows: int,
    equivalence_report: Path,
    artifact_location: str | None = None,
) -> dict[str, Any]:
    """Analyze every accepted release row without conflating shared algorithm names."""
    equivalence_sha = _require_equivalence(equivalence_report, expected_source, expected_rows)
    paths = sorted((campaign_root / "runs").glob("*/episodes.jsonl"))
    if not paths:
        raise ValueError("release campaign has no arm episode files")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    locations: dict[tuple[str, str, int], tuple[Path, int]] = {}
    sources = []
    posthoc_rows = 0
    for path in paths:
        arm_rows, derived_count, source = _read_arm(
            path, expected_source, seen, locations, artifact_location
        )
        rows.extend(arm_rows)
        posthoc_rows += derived_count
        sources.append(source)
    if len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} release rows, found {len(rows)}")
    result = analyze(rows)
    result["classification"] = "release_robot_force_validation"
    result["posthoc_discomfort"]["rows"] = posthoc_rows
    result["sources"] = sources
    result["equivalence_report_sha256"] = equivalence_sha
    result["source_commit"] = expected_source
    result["arm_identity"] = "runs/<arm>/episodes.jsonl; distinct arms may share algo"
    result["claim_boundary"] = (
        "Descriptive robot-attributable simulator force, parameter-dependent and not measured "
        "human discomfort; correlations and rank disagreements do not establish causality."
    )
    for disagreement in result["largest_rank_disagreements"]:
        key = disagreement["algo"], disagreement["scenario_id"], disagreement["seed"]
        disagreement["arm"] = disagreement.pop("algo")
        disagreement["trace_evidence"] = _selected_trace_evidence(
            locations[key], key, expected_source
        )
    return result


def _render_markdown(report: dict[str, Any]) -> str:
    """Summarize the complete machine-readable validation without a causal claim."""
    lines = [
        "# Robot-attributable force validation for benchmark data 0.0.8",
        "",
        report["claim_boundary"],
        "",
        f"- Episodes: {report['episodes']}",
        f"- Source commit: `{report['source_commit']}`",
        f"- Passing predecessor equivalence report SHA-256: `{report['equivalence_report_sha256']}`",
        f"- Post-hoc discomfort values recovered from recorded traces: "
        f"{report['posthoc_discomfort']['rows']}",
        "",
        "The companion JSON records all cohorts, nullable correlations, twenty largest "
        "rank disagreements, source-file checksums, and trace-evidence availability.",
        "Minimum-distance correlations use the negative of recorded distance, so a positive "
        "rho means greater force aligns with closer approach.",
        "",
        "| Force metric | Comparator | Paired rows | Spearman rho |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in report["correlations"]:
        if row["cohort"] != "overall":
            continue
        rho = row["spearman_rho"]
        lines.append(
            f"| `{row['force']}` | `{row['comparator']}` | {row['n']} | "
            f"{'unavailable' if rho is None else f'{rho:.6f}'} |"
        )
    lines.extend(["", "## Largest rank disagreements", ""])
    lines.extend(
        [
            "| Arm | Scenario | Seed | Force rank | Distance rank | Trace |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in report["largest_rank_disagreements"]:
        lines.append(
            f"| `{row['arm']}` | `{row['scenario_id']}` | {row['seed']} | "
            f"{row['force_rank']:.1f} | {row['distance_rank']:.1f} | "
            f"{row['trace_evidence']['status']} |"
        )
    return "\n".join([*lines, ""])


def main() -> int:
    """Write a checksum-bound release report after the old-metric gate passes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument("--expected-episodes", type=int, default=20160)
    parser.add_argument("--equivalence-report", type=Path, required=True)
    parser.add_argument("--artifact-location")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        args.campaign_root,
        expected_source=args.expected_source_sha,
        expected_rows=args.expected_episodes,
        equivalence_report=args.equivalence_report,
        artifact_location=args.artifact_location,
    )
    if args.output.suffix != ".json":
        raise ValueError("robot-force validation output must be a JSON path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    args.output.with_suffix(".md").write_text(_render_markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
