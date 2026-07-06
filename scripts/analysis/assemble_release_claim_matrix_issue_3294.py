#!/usr/bin/env python3
"""Assemble the issue #3294 v0.1 release claim matrix from tracked evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.benchmark_row_claim import validate_leaderboard_claims

SCHEMA_VERSION = "release_claim_matrix_issue_3294.v1"
DEFAULT_OUTPUT_DIR = Path("docs/context/evidence/issue_3294_release_claim_matrix")
DEFAULT_RELEASE_SNAPSHOT = Path(
    "docs/context/evidence/issue_3205_release_evidence_gate/snapshot_0_0_2.json"
)
DEFAULT_ARTIFACT_MANIFEST = Path(
    "docs/context/evidence/issue_2686_release_0_0_2_table_bundle/artifact_manifest.json"
)
DEFAULT_RELEASE_CONFIG = Path(
    "configs/benchmarks/releases/paper_experiment_matrix_v1_release_v0_1.yaml"
)
DEFAULT_ODD_COVERAGE = Path(
    "docs/context/evidence/issue_2911_odd_hazard_coverage_2026-06-17/coverage_matrix.json"
)
DEFAULT_SCENARIO_CERTIFICATION_SUMMARY = Path(
    "docs/context/evidence/issue_2910_release_scenario_certification/summary.json"
)
DEFAULT_LEADERBOARD_GLOB = "docs/leaderboards/*.rows.json"

CLAIM_BOUNDARY = (
    "Issue #3294 assembles a review matrix from existing tracked metadata and release artifacts. "
    "It does not run a new benchmark campaign, create scenario certification, or establish "
    "deployment-safety, real-world AMV, fallback, degraded, or unavailable rows as successful "
    "benchmark evidence."
)


@dataclass(frozen=True)
class SourcePaths:
    """Inputs used to assemble the release claim matrix."""

    release_snapshot: Path
    artifact_manifest: Path
    release_config: Path
    odd_coverage: Path
    scenario_certification_summary: Path
    leaderboard_sidecars: tuple[Path, ...]


def _repo_relative(path: Path) -> str:
    """Return a POSIX repository-relative path string."""
    return path.as_posix()


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object, failing clearly if the source is not object-shaped."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML object, failing clearly if the source is not object-shaped."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML object")
    return payload


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest for a tracked source file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _classify_row_claim(row: dict[str, Any]) -> str:
    """Map benchmark_row_claim.v1 fields into issue #3294 classification terms."""
    row_status = str(row.get("row_status", ""))
    planner_mode = str(row.get("planner_mode", ""))
    evidence_tier = str(row.get("evidence_tier", ""))
    if planner_mode == "fallback" or row_status == "fallback":
        return "fallback-excluded"
    if planner_mode == "degraded" or row_status == "degraded":
        return "degraded"
    if planner_mode == "not_available" or row_status in {
        "accepted_unavailable",
        "not_yet_populated",
    }:
        return "unavailable"
    if row_status in {"blocked", "excluded", "unexpected_failure", "revise"}:
        return "non-claim"
    if evidence_tier in {"benchmark", "paper_facing"} and row_status in {
        "successful_evidence",
        "pass",
    }:
        return "benchmark evidence"
    if evidence_tier in {"diagnostic", "smoke"}:
        return "diagnostic evidence"
    return "non-claim"


def _classify_odd_row(row: dict[str, Any]) -> str:
    """Map ODD coverage rows into issue #3294 classification terms."""
    status = str(row.get("status", ""))
    evidence_tier = str(row.get("evidence_tier", ""))
    if status in {"blocked", "absent"}:
        return "unavailable"
    if evidence_tier == "diagnostic":
        return "diagnostic evidence"
    return "non-claim"


def _scenario_certification_status(summary: dict[str, Any]) -> str:
    """Return publication-gate scenario-certification status from summary.

    Fail-closed: an explicit-``null`` status is treated as absent (not the
    string ``"None"``), and a missing or incomplete ``benchmark_eligibility_counts``
    block never certifies as accepted — accepted requires both ``excluded`` and
    ``stress_only`` to be present and zero.
    """
    raw_status = summary.get("publication_gate_status")
    status = str(raw_status).strip() if raw_status is not None else ""
    if status:
        return status
    eligibility_counts = summary.get("benchmark_eligibility_counts")
    if (
        isinstance(eligibility_counts, dict)
        and "excluded" in eligibility_counts
        and "stress_only" in eligibility_counts
        and int(eligibility_counts["excluded"]) == 0
        and int(eligibility_counts["stress_only"]) == 0
    ):
        return "scenario_cert.v1:accepted"
    return "scenario_cert.v1:blocked"


def _scenario_certification_prerequisites(summary: dict[str, Any]) -> list[str]:
    """Return compact missing-prerequisite text for non-accepted certification.

    Fail-closed: an explicit-``null`` blocker is treated as absent, and missing
    eligibility counts are reported as unavailable rather than a misleading
    ``(0 excluded, 0 stress-only)``.
    """
    status = _scenario_certification_status(summary)
    if status in {"scenario_cert.v1:accepted", "scenario_cert.v1:accepted_reviewed"}:
        return []
    raw_blocker = summary.get("publication_blocker")
    blocker = str(raw_blocker).strip() if raw_blocker is not None else ""
    if blocker:
        return [blocker]
    eligibility_counts = summary.get("benchmark_eligibility_counts")
    if (
        isinstance(eligibility_counts, dict)
        and "excluded" in eligibility_counts
        and "stress_only" in eligibility_counts
    ):
        excluded = int(eligibility_counts["excluded"])
        stress_only = int(eligibility_counts["stress_only"])
        return [
            (
                "scenario_cert.v1 summary is not publication-accepted "
                f"({excluded} excluded, {stress_only} stress-only)"
            )
        ]
    return [
        "scenario_cert.v1 summary is not publication-accepted "
        "(benchmark_eligibility_counts missing or incomplete)"
    ]


def _release_artifact_rows(
    *,
    snapshot: dict[str, Any],
    artifact_manifest: dict[str, Any],
    release_config: dict[str, Any],
    scenario_certification_summary: dict[str, Any],
    source_paths: SourcePaths,
) -> list[dict[str, Any]]:
    """Build matrix rows from the release artifact manifest and release evidence gate snapshot."""
    snapshot_artifacts = {
        str(item.get("artifact_id", "")): item
        for item in snapshot.get("artifacts", [])
        if isinstance(item, dict)
    }
    planners = list(release_config.get("planners", {}).get("keys", []))
    seed_policy = release_config.get("seed_policy", {})
    seed_schedule = {
        "mode": seed_policy.get("mode", "unknown"),
        "seed_set": seed_policy.get("seed_set", "unknown"),
        "seed_sets_path": seed_policy.get("seed_sets_path", "unknown"),
        "seeds": seed_policy.get("seeds", []),
    }
    scenario_certification_status = _scenario_certification_status(scenario_certification_summary)
    scenario_certification_prerequisites = _scenario_certification_prerequisites(
        scenario_certification_summary
    )
    rows: list[dict[str, Any]] = []
    for artifact in artifact_manifest.get("artifacts", []):
        if not isinstance(artifact, dict):
            continue
        artifact_id = str(artifact.get("artifact_id", ""))
        snapshot_row = snapshot_artifacts.get(artifact_id, {})
        sha_match = bool(snapshot_row.get("match"))
        classification = (
            "benchmark evidence"
            if snapshot.get("status") == "PASS" and sha_match
            else "unavailable"
        )
        rows.append(
            {
                "row_id": f"release_artifact:{artifact_id}",
                "section": "release_artifact",
                "classification": classification,
                "suite_or_surface": str(release_config.get("release_id", "unknown")),
                "scenario_family": "release_0_0_2_table_bundle",
                "planner_rows": planners,
                "planner_id": None,
                "planner_mode": "mixed",
                "odd_condition": "low_speed_public_space_v1",
                "hazard_class": "mixed_release_table",
                "seed_schedule": seed_schedule,
                "scenario_contract_ref": str(
                    release_config.get("scenario", {}).get("matrix_path", "unknown")
                ),
                "scenario_certification": scenario_certification_status,
                "scenario_certification_summary": {
                    "summary_ref": _repo_relative(source_paths.scenario_certification_summary),
                    "scenario_count": scenario_certification_summary.get("scenario_count"),
                    "classification_counts": scenario_certification_summary.get(
                        "classification_counts", {}
                    ),
                    "benchmark_eligibility_counts": scenario_certification_summary.get(
                        "benchmark_eligibility_counts", {}
                    ),
                },
                "artifact_uri": _repo_relative(source_paths.artifact_manifest),
                "artifact_sha256": artifact.get("sha256") or snapshot_row.get("tracked_sha256"),
                "artifact_match": sha_match,
                "claim_boundary": artifact.get("claim_boundary", CLAIM_BOUNDARY),
                "row_status": "release_snapshot_pass" if sha_match else "missing_or_mismatch",
                "availability_status": "available" if sha_match else "not_available",
                "benchmark_success": None,
                "exclusions": [
                    artifact.get("fallback_degraded_summary", "preserve release caveats")
                ],
                "source_refs": [
                    _repo_relative(source_paths.release_snapshot),
                    _repo_relative(source_paths.artifact_manifest),
                    _repo_relative(source_paths.release_config),
                    _repo_relative(source_paths.scenario_certification_summary),
                ],
                "missing_prerequisites": [
                    *scenario_certification_prerequisites,
                    *(
                        []
                        if sha_match
                        else ["release evidence snapshot did not verify this artifact"]
                    ),
                ],
            }
        )
    return rows


def _leaderboard_rows(source_paths: SourcePaths) -> list[dict[str, Any]]:
    """Build matrix rows from benchmark_row_claim.v1 leaderboard sidecars."""
    rows: list[dict[str, Any]] = []
    for sidecar in source_paths.leaderboard_sidecars:
        validation = validate_leaderboard_claims(sidecar)
        payload = _load_json(sidecar)
        for index, row in enumerate(payload.get("rows", [])):
            if not isinstance(row, dict):
                continue
            classification = _classify_row_claim(row)
            rows.append(
                {
                    "row_id": f"leaderboard:{sidecar.stem}:{index:03d}",
                    "section": "leaderboard_row_claim",
                    "classification": classification,
                    "suite_or_surface": row.get("suite_id"),
                    "scenario_family": row.get("suite_id"),
                    "planner_rows": [row.get("planner_id")],
                    "planner_id": row.get("planner_id"),
                    "planner_mode": row.get("planner_mode"),
                    "odd_condition": "not_recorded",
                    "hazard_class": "not_recorded",
                    "seed_schedule": {"seeds": row.get("seeds", [])},
                    "scenario_contract_ref": "not_recorded",
                    "scenario_certification": "not_recorded",
                    "artifact_uri": row.get("artifact_uri"),
                    "artifact_sha256": _sha256(Path(str(row.get("artifact_uri"))))
                    if Path(str(row.get("artifact_uri", ""))).exists()
                    else None,
                    "artifact_match": True,
                    "claim_boundary": row.get("claim_boundary"),
                    "claim_wording": row.get("claim_wording"),
                    "row_status": row.get("row_status"),
                    "availability_status": "available"
                    if classification in {"benchmark evidence", "diagnostic evidence"}
                    else classification,
                    "benchmark_success": classification == "benchmark evidence",
                    "metrics": row.get("metrics", {}),
                    "exclusions": row.get("exclusions", []),
                    "source_refs": [_repo_relative(sidecar)],
                    "validation": {
                        "sidecar_status": validation.get("status", "ok"),
                        "row_count": validation.get("row_count"),
                    },
                    "missing_prerequisites": []
                    if classification == "benchmark evidence"
                    else ["not benchmark-strength evidence"],
                }
            )
    return rows


def _odd_rows(*, odd_coverage: dict[str, Any], source_paths: SourcePaths) -> list[dict[str, Any]]:
    """Build matrix rows from the ODD hazard coverage matrix."""
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(odd_coverage.get("coverage_rows", [])):
        if not isinstance(row, dict):
            continue
        classification = _classify_odd_row(row)
        rows.append(
            {
                "row_id": f"odd_hazard:{index:03d}",
                "section": "odd_hazard_coverage",
                "classification": classification,
                "suite_or_surface": "odd_hazard_coverage.v1",
                "scenario_family": row.get("scenario_family"),
                "planner_rows": row.get("included_planners", []),
                "planner_id": None,
                "planner_mode": "not_recorded",
                "odd_condition": row.get("odd_condition"),
                "hazard_class": row.get("hazard_class"),
                "seed_schedule": {"seeds": "not_recorded"},
                "scenario_contract_ref": row.get("source_configs", []),
                "scenario_certification": "not_available",
                "artifact_uri": _repo_relative(source_paths.odd_coverage),
                "artifact_sha256": _sha256(source_paths.odd_coverage),
                "artifact_match": True,
                "claim_boundary": odd_coverage.get("claim_boundary", CLAIM_BOUNDARY),
                "row_status": row.get("status"),
                "availability_status": "not_available"
                if classification == "unavailable"
                else "available",
                "benchmark_success": False,
                "metrics": row.get("metrics", []),
                "exclusions": [row.get("gap_reason", "missing benchmark evidence")],
                "source_refs": [
                    _repo_relative(source_paths.odd_coverage),
                    *row.get("source_configs", []),
                ],
                "missing_prerequisites": [row.get("gap_reason", "not benchmark evidence")],
            }
        )
    return rows


def build_matrix(source_paths: SourcePaths) -> dict[str, Any]:
    """Build the complete issue #3294 release claim matrix payload."""
    release_snapshot = _load_json(source_paths.release_snapshot)
    artifact_manifest = _load_json(source_paths.artifact_manifest)
    release_config = _load_yaml(source_paths.release_config)
    odd_coverage = _load_json(source_paths.odd_coverage)
    scenario_certification_summary = _load_json(source_paths.scenario_certification_summary)
    rows = [
        *_release_artifact_rows(
            snapshot=release_snapshot,
            artifact_manifest=artifact_manifest,
            release_config=release_config,
            scenario_certification_summary=scenario_certification_summary,
            source_paths=source_paths,
        ),
        *_leaderboard_rows(source_paths),
        *_odd_rows(odd_coverage=odd_coverage, source_paths=source_paths),
    ]
    classification_counts: dict[str, int] = {}
    for row in rows:
        classification = str(row["classification"])
        classification_counts[classification] = classification_counts.get(classification, 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "issue": 3294,
        "claim_boundary": CLAIM_BOUNDARY,
        "sources": {
            "release_snapshot": _repo_relative(source_paths.release_snapshot),
            "artifact_manifest": _repo_relative(source_paths.artifact_manifest),
            "release_config": _repo_relative(source_paths.release_config),
            "odd_coverage": _repo_relative(source_paths.odd_coverage),
            "scenario_certification_summary": _repo_relative(
                source_paths.scenario_certification_summary
            ),
            "leaderboard_sidecars": [
                _repo_relative(path) for path in source_paths.leaderboard_sidecars
            ],
        },
        "summary": {
            "row_count": len(rows),
            "classification_counts": dict(sorted(classification_counts.items())),
            "release_snapshot_status": release_snapshot.get("status"),
            "release_tag": release_snapshot.get("release_tag"),
            "doi": release_snapshot.get("doi"),
        },
        "rows": rows,
    }


def _markdown_cell(value: Any) -> str:
    """Render a compact Markdown table cell."""

    def clean(cell_value: Any) -> str:
        return str(cell_value).replace("\n", " ").strip()

    if value is None:
        return "NA"
    if isinstance(value, list):
        text = ", ".join(clean(item) for item in value[:4])
        if len(value) > 4:
            text += f", +{len(value) - 4} more"
        return text
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return clean(value)


def render_markdown(matrix: dict[str, Any]) -> str:
    """Render a compact Markdown matrix for review."""
    lines = [
        "# Issue 3294 Release Claim Matrix",
        "",
        f"Schema: `{matrix['schema_version']}`",
        "",
        f"Claim boundary: {matrix['claim_boundary']}",
        "",
        "## Summary",
        "",
    ]
    summary = matrix["summary"]
    lines.extend(
        [
            f"- Rows: {summary['row_count']}",
            f"- Release snapshot status: `{summary.get('release_snapshot_status')}`",
            f"- Release tag: `{summary.get('release_tag')}`",
            f"- DOI: `{summary.get('doi')}`",
            f"- Classification counts: `{json.dumps(summary['classification_counts'], sort_keys=True)}`",
            "",
            "## Matrix",
            "",
            "| Row | Section | Classification | Surface | Scenario/Hazard | Planner(s) | Artifact | Caveat |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in matrix["rows"]:
        scenario_hazard = f"{_markdown_cell(row.get('scenario_family'))} / {_markdown_cell(row.get('hazard_class'))}"
        caveats = [
            *(row.get("exclusions") or []),
            *(row.get("missing_prerequisites") or []),
        ]
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_cell(row.get("row_id")),
                    _markdown_cell(row.get("section")),
                    _markdown_cell(row.get("classification")),
                    _markdown_cell(row.get("suite_or_surface")),
                    _markdown_cell(scenario_hazard),
                    _markdown_cell(row.get("planner_rows")),
                    _markdown_cell(row.get("artifact_uri")),
                    _markdown_cell(caveats),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Source Files",
            "",
        ]
    )
    for key, value in matrix["sources"].items():
        lines.append(f"- `{key}`: `{_markdown_cell(value)}`")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--release-snapshot", type=Path, default=DEFAULT_RELEASE_SNAPSHOT)
    parser.add_argument("--artifact-manifest", type=Path, default=DEFAULT_ARTIFACT_MANIFEST)
    parser.add_argument("--release-config", type=Path, default=DEFAULT_RELEASE_CONFIG)
    parser.add_argument("--odd-coverage", type=Path, default=DEFAULT_ODD_COVERAGE)
    parser.add_argument(
        "--scenario-certification-summary",
        type=Path,
        default=DEFAULT_SCENARIO_CERTIFICATION_SUMMARY,
    )
    parser.add_argument("--leaderboard-glob", default=DEFAULT_LEADERBOARD_GLOB)
    return parser.parse_args()


def main() -> int:
    """Run the matrix assembly CLI."""
    args = parse_args()
    source_paths = SourcePaths(
        release_snapshot=args.release_snapshot,
        artifact_manifest=args.artifact_manifest,
        release_config=args.release_config,
        odd_coverage=args.odd_coverage,
        scenario_certification_summary=args.scenario_certification_summary,
        leaderboard_sidecars=tuple(sorted(Path().glob(args.leaderboard_glob))),
    )
    matrix = build_matrix(source_paths)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = output_dir / "release_claim_matrix.json"
    markdown_path = output_dir / "release_claim_matrix.md"
    matrix_path.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(matrix), encoding="utf-8")
    print(
        json.dumps(
            {
                "schema_version": matrix["schema_version"],
                "row_count": matrix["summary"]["row_count"],
                "classification_counts": matrix["summary"]["classification_counts"],
                "matrix_path": _repo_relative(matrix_path),
                "markdown_path": _repo_relative(markdown_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
