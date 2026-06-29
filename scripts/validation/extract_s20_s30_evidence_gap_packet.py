#!/usr/bin/env python3
"""Extract a diagnostic-only evidence-gap packet for issue #1554 S20/S30 artifacts.

This helper reads already-retrieved benchmark artifacts. It never submits Slurm,
runs a benchmark, promotes raw artifacts, or edits paper/dissertation claims.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "s20-s30-evidence-gap-packet.v1"
DEFAULT_JOB_ID = "13175"
DEFAULT_ARTIFACT_ROOT = Path("output/issue1554-s20-h500-l40s-mem180/13175")
DEFAULT_PACKET_FIXTURE = Path(
    "docs/context/evidence/issue_3798_post_13175_s20_s30_evidence_gap_packet.json"
)
CLAIM_BOUNDARY = (
    "diagnostic-only evidence-gap packet; no full benchmark campaign run, no "
    "Slurm/GPU submission, no paper/dissertation claim edits, and no S20/S30 "
    "claim promotion"
)
PROMOTABLE_REVIEW_FILES = (
    "campaign_manifest.json",
    "run_meta.json",
    "reports/campaign_table.csv",
    "reports/campaign_table_core.md",
    "reports/campaign_table_experimental.md",
    "reports/snqi_diagnostics.md",
    "reports/statistical_sufficiency.json",
)
REQUIRED_METADATA_FILES = (
    "campaign_manifest.json",
    "run_meta.json",
    "reports/campaign_table.csv",
    "reports/seed_episode_rows.csv",
    "reports/statistical_sufficiency.json",
)


def build_packet(artifact_root: Path, *, job_id: str = DEFAULT_JOB_ID) -> dict[str, Any]:
    """Build a compact packet from retrieved artifact metadata."""

    root = artifact_root.resolve()
    missing_files = [path for path in REQUIRED_METADATA_FILES if not (root / path).is_file()]
    manifest = _load_json(root / "campaign_manifest.json") if not missing_files else {}
    run_meta = _load_json(root / "run_meta.json") if not missing_files else {}
    campaign_rows = _read_campaign_table(root / "reports/campaign_table.csv")
    seed_episode_rows = _read_seed_episode_rows(root / "reports/seed_episode_rows.csv")

    packet = {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked_missing_retrieved_metadata" if missing_files else "diagnostic_only",
        "job_id": str(job_id),
        "artifact_root": str(root),
        "claim_boundary": CLAIM_BOUNDARY,
        "campaign": _campaign_metadata(manifest, run_meta),
        "retrieved_artifacts": {
            "required_metadata_files": list(REQUIRED_METADATA_FILES),
            "missing_required_metadata_files": missing_files,
            "file_count": _count_files(root),
            "total_bytes": _sum_bytes(root),
        },
        "promotable_review_files": _promotable_review_files(root),
        "non_promotable_artifacts": [
            {
                "path_glob": "runs/*/episodes.jsonl",
                "reason": "Large raw episode logs remain ignored worktree output; promote only through an explicit durable artifact pointer if needed.",
            },
            {
                "path_glob": "reports/seed_episode_rows.csv",
                "reason": "Useful for validation, but too detailed for the public packet unless a later claim needs a compact derived table.",
            },
            {
                "path_glob": "reports/seed_variability_by_scenario.*",
                "reason": "Large scenario-level diagnostics; summarize before any public claim surface.",
            },
        ],
        "coverage_snapshot": {
            "planner_rows": campaign_rows,
            "planner_count": len(campaign_rows),
            "episode_rows": seed_episode_rows["rows"],
            "seed_count": len(seed_episode_rows["seeds"]),
            "seeds": seed_episode_rows["seeds"],
            "planners": seed_episode_rows["planners"],
        },
        "claim_blockers": _claim_blockers(campaign_rows, seed_episode_rows, missing_files),
        "next_slurm_go_no_go": _next_slurm_go_no_go(
            missing_files=missing_files,
            seed_count=len(seed_episode_rows["seeds"]),
        ),
        "validation_commands": [
            {
                "command": (
                    "uv run python scripts/validation/extract_s20_s30_evidence_gap_packet.py "
                    f"--artifact-root {root} --job-id {job_id} --markdown"
                ),
                "purpose": "Print this diagnostic packet from retrieved job metadata without submitting Slurm or editing claims.",
            },
            {
                "command": "uv run pytest tests/validation/test_extract_s20_s30_evidence_gap_packet.py",
                "purpose": "Fixture proof for present and missing retrieved-artifact metadata.",
            },
            {
                "command": "uv run python scripts/validation/check_s20_s30_archive_readiness.py --json",
                "purpose": "Fail-closed archive-readiness check; remains the claim gate for a canonical result store.",
            },
        ],
    }
    return packet


def load_packet_fixture(packet_fixture: Path) -> dict[str, Any]:
    """Load a compact tracked packet fixture for hosts without raw output artifacts."""

    packet = _load_json(packet_fixture)
    if packet.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{packet_fixture} schema_version must be {SCHEMA_VERSION!r}")
    if packet.get("job_id") != DEFAULT_JOB_ID:
        raise ValueError(f"{packet_fixture} job_id must be {DEFAULT_JOB_ID!r}")
    if packet.get("claim_boundary") != CLAIM_BOUNDARY:
        raise ValueError(f"{packet_fixture} claim boundary drifted")
    go_no_go = packet.get("next_slurm_go_no_go", {})
    if go_no_go.get("claim_promotion") != "no_go":
        raise ValueError(f"{packet_fixture} claim_promotion must remain no_go")
    if go_no_go.get("s30_submission_from_issue_3798") != "not_authorized_here":
        raise ValueError(f"{packet_fixture} must not authorize S30 submission from issue #3798")
    return packet


def render_markdown(packet: dict[str, Any]) -> str:
    """Render a public, compact Markdown packet."""

    lines = [
        f"# Job {packet['job_id']} S20/S30 Evidence-Gap Packet",
        "",
        (
            "This packet summarizes retrieved S20 (20-seed) job artifacts for issue #1554, "
            "keeps S30 (30-seed) as unexecuted escalation scope, and records why issue #3798 "
            "does not promote paper or dissertation claims."
        ),
        "",
        "## Claim Boundary",
        "",
        f"- Status: `{packet['status']}`",
        f"- Boundary: {packet['claim_boundary']}",
        "",
        "## Next Slurm Go/No-Go",
        "",
    ]
    for key, value in packet["next_slurm_go_no_go"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Retrieved Artifacts",
            "",
            f"- Artifact root: `{packet['artifact_root']}`",
            f"- File count: {packet['retrieved_artifacts']['file_count']}",
            f"- Total bytes: {packet['retrieved_artifacts']['total_bytes']}",
            (
                "- Missing required metadata files: "
                f"{packet['retrieved_artifacts']['missing_required_metadata_files'] or 'none'}"
            ),
            "",
            "## Campaign Metadata",
            "",
        ]
    )

    for key, value in packet["campaign"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Coverage Snapshot",
            "",
            f"- Planner rows: {packet['coverage_snapshot']['planner_count']}",
            f"- Episode rows in compact seed table: {packet['coverage_snapshot']['episode_rows']}",
            f"- Seed count: {packet['coverage_snapshot']['seed_count']}",
            f"- Seeds: `{packet['coverage_snapshot']['seeds']}`",
            f"- Planners: `{packet['coverage_snapshot']['planners']}`",
            "",
            "## Promotable Review Files",
            "",
        ]
    )
    for item in packet["promotable_review_files"]:
        lines.append(
            f"- `{item['path']}` ({item['bytes']} bytes, sha256 `{item['sha256']}`): "
            f"{item['promotion_scope']}"
        )
    lines.extend(["", "## Claim Blockers", ""])
    for blocker in packet["claim_blockers"]:
        lines.append(f"- {blocker}")
    lines.extend(["", "## Validation Commands", ""])
    for item in packet["validation_commands"]:
        lines.append(f"- `{item['command']}`")
        lines.append(f"  - {item['purpose']}")
    return "\n".join(lines) + "\n"


def _campaign_metadata(manifest: dict[str, Any], run_meta: dict[str, Any]) -> dict[str, Any]:
    git = manifest.get("git", {}) if isinstance(manifest.get("git"), dict) else {}
    seed_policy = (
        manifest.get("seed_policy", {}) if isinstance(manifest.get("seed_policy"), dict) else {}
    )
    repo = run_meta.get("repo", {}) if isinstance(run_meta.get("repo"), dict) else {}
    return {
        "campaign_id": manifest.get("campaign_id") or run_meta.get("campaign_id"),
        "config_name": manifest.get("name"),
        "started_at_utc": manifest.get("started_at_utc") or run_meta.get("started_at_utc"),
        "finished_at_utc": manifest.get("finished_at_utc") or run_meta.get("finished_at_utc"),
        "git_commit": git.get("commit") or repo.get("commit") or manifest.get("git_hash"),
        "git_branch": git.get("branch") or repo.get("branch"),
        "seed_set": seed_policy.get("seed_set"),
        "resolved_seed_count": len(seed_policy.get("resolved_seeds", [])),
        "scenario_matrix": manifest.get("scenario_matrix"),
        "scenario_matrix_hash": manifest.get("scenario_matrix_hash")
        or run_meta.get("scenario_matrix_hash"),
    }


def _promotable_review_files(root: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for rel_path in PROMOTABLE_REVIEW_FILES:
        path = root / rel_path
        if path.is_file():
            files.append(
                {
                    "path": rel_path,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                    "promotion_scope": "small metadata/report surface only; diagnostic evidence, not a claim upgrade",
                }
            )
    return files


def _claim_blockers(
    campaign_rows: list[dict[str, str]], seed_episode_rows: dict[str, Any], missing_files: list[str]
) -> list[str]:
    blockers = [
        "The retrieved job is S20 only; S30 remains an escalation path, not executed evidence.",
        "Artifacts live under ignored worktree output and are not yet a canonical campaign result store.",
        "Archive-readiness still depends on the fail-closed result-store checker before any claim promotion.",
        "No dissertation, paper, ranking, safety, or significance claim is edited by this packet.",
    ]
    if missing_files:
        blockers.append(f"Required retrieved metadata is missing: {missing_files}.")
    if seed_episode_rows["rows"] == 0:
        blockers.append("No compact seed episode rows were readable from the retrieved artifacts.")
    if campaign_rows and any(row.get("status") != "ok" for row in campaign_rows):
        blockers.append(
            "At least one campaign table row is not `ok`; row status needs fail-closed review."
        )
    return blockers


def _next_slurm_go_no_go(*, missing_files: list[str], seed_count: int) -> dict[str, str]:
    """Return the diagnostic-only next-action decision for the packet."""

    if missing_files:
        archive_readiness = "blocked_missing_retrieved_metadata"
        s30_status = "not_ready_missing_s20_metadata"
        next_step = "recover required retrieved metadata before considering any escalation"
    elif seed_count < 20:
        archive_readiness = "blocked_incomplete_s20_seed_coverage"
        s30_status = "not_ready_incomplete_s20"
        next_step = "reconcile S20 seed coverage before considering any escalation"
    else:
        archive_readiness = "run_fail_closed_checker_before_claim"
        s30_status = "defer_until_claim_owner_authorizes_escalation"
        next_step = (
            "run archive-readiness checker and decide S30 only in a separately authorized lane"
        )

    return {
        "claim_promotion": "no_go",
        "s20_archive_readiness": archive_readiness,
        "s30_submission_from_issue_3798": "not_authorized_here",
        "s30_escalation_status": s30_status,
        "compute_submission": "not_authorized",
        "next_valid_step": next_step,
    }


def _read_campaign_table(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "planner_key": row.get("planner_key", ""),
                    "planner_group": row.get("planner_group", ""),
                    "readiness_tier": row.get("readiness_tier", ""),
                    "status": row.get("status", ""),
                    "episodes": row.get("episodes", ""),
                    "success_mean": row.get("success_mean", ""),
                    "collisions_mean": row.get("collisions_mean", ""),
                    "snqi_mean": row.get("snqi_mean", "").lstrip("'"),
                }
            )
    return rows


def _read_seed_episode_rows(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"rows": 0, "seeds": [], "planners": []}
    rows = 0
    seeds: set[int] = set()
    planners: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows += 1
            if row.get("planner_key"):
                planners.add(str(row["planner_key"]))
            try:
                seeds.add(int(str(row.get("seed", ""))))
            except ValueError:
                continue
    return {"rows": rows, "seeds": sorted(seeds), "planners": sorted(planners)}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _count_files(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for path in root.rglob("*") if path.is_file())


def _sum_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument(
        "--packet-fixture",
        type=Path,
        help="Tracked compact packet JSON used when ignored raw artifacts are unavailable.",
    )
    parser.add_argument("--job-id", default=DEFAULT_JOB_ID)
    parser.add_argument("--markdown", action="store_true", help="Print a Markdown packet.")
    parser.add_argument("--json", action="store_true", help="Print the packet JSON.")
    parser.add_argument("--output", type=Path, help="Optional output path for the rendered packet.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the diagnostic packet extractor CLI."""

    args = _parse_args(argv)
    packet = (
        load_packet_fixture(args.packet_fixture)
        if args.packet_fixture is not None
        else build_packet(args.artifact_root, job_id=args.job_id)
    )
    rendered = (
        json.dumps(packet, indent=2, sort_keys=True) if args.json else render_markdown(packet)
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 1 if packet["status"] == "blocked_missing_retrieved_metadata" else 0


if __name__ == "__main__":
    raise SystemExit(main())
