#!/usr/bin/env python3
"""Promote issue #3637 analyzer outputs into a durable evidence bundle.

This command does not run the benchmark campaign and does not make a paper-facing
claim. It copies the compact post-run analyzer artifacts into docs/context/evidence
after checking the issue, replay limitation, and expected file contract.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from robot_sf.benchmark.identity.hash_utils import sha256_file

ISSUE = 3637
ANALYSIS_SCHEMA = "reactivity-replay-rank-study-analysis.v1"
PROMOTION_SCHEMA = "issue-3637-reactivity-replay-rank-promotion.v1"
DEFAULT_ANALYSIS_DIR = Path("output/issue_3637_reactivity_rank_study/analysis")
DEFAULT_EVIDENCE_DIR = Path("docs/context/evidence/issue_3637_reactivity_replay_rank_study")
REQUIRED_ARTIFACTS = (
    "README.md",
    "analysis.json",
    "frozen_gate_input.json",
    "seed_gate_decision.json",
    "rank_bootstrap_summary.json",
    "per_planner_condition_metrics.csv",
)


class PromotionError(RuntimeError):
    """Raised when analyzer outputs are not ready for durable promotion."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PromotionError(f"missing required artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise PromotionError(f"{path} must contain a JSON object")
    return payload


def _validate_analysis(analysis_dir: Path) -> dict[str, Any]:
    missing = [name for name in REQUIRED_ARTIFACTS if not (analysis_dir / name).exists()]
    if missing:
        raise PromotionError(
            "analysis directory is incomplete; missing " + ", ".join(sorted(missing))
        )

    analysis = _load_json(analysis_dir / "analysis.json")
    if analysis.get("schema_version") != ANALYSIS_SCHEMA:
        raise PromotionError(
            f"analysis.json schema_version must be {ANALYSIS_SCHEMA!r}, "
            f"got {analysis.get('schema_version')!r}"
        )
    if analysis.get("issue") != ISSUE:
        raise PromotionError(f"analysis.json issue must be {ISSUE}")
    replay = analysis.get("replay_limitation")
    if not isinstance(replay, dict) or replay.get("is_trajectory_playback") is not False:
        raise PromotionError("analysis replay_limitation must state is_trajectory_playback=false")
    if analysis.get("claim_decision") != "post_run_gate_input_ready":
        raise PromotionError("analysis claim_decision must be post_run_gate_input_ready")
    if analysis.get("episode_count") != analysis.get("expected_episode_count"):
        raise PromotionError("analysis episode_count must match expected_episode_count")
    seed_gate_decision = _load_json(analysis_dir / "seed_gate_decision.json")
    decision = seed_gate_decision.get("decision")
    if not isinstance(decision, str) or not decision:
        raise PromotionError("seed_gate_decision.json must include decision")
    return analysis


def _write_manifest(evidence_dir: Path, copied: list[str]) -> None:
    lines = []
    for name in sorted(copied):
        lines.append(f"{sha256_file(evidence_dir / name)}  {name}")
    (evidence_dir / "manifest.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary(evidence_dir: Path, analysis: dict[str, Any], analysis_dir: Path) -> None:
    summary = {
        "schema_version": PROMOTION_SCHEMA,
        "issue": ISSUE,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_analysis_dir": str(analysis_dir),
        "source_campaign_issue": analysis.get("source_campaign_issue"),
        "planners": analysis.get("planners"),
        "seeds": analysis.get("seeds"),
        "scenario_set": analysis.get("scenario_set"),
        "episode_count": analysis.get("episode_count"),
        "claim_decision": analysis.get("claim_decision"),
        "claim_boundary": analysis.get("claim_boundary"),
        "replay_limitation": analysis.get("replay_limitation"),
        "rank_effect": analysis.get("rank_effect"),
        "paired_seed_bootstrap": analysis.get("paired_seed_bootstrap"),
        "seed_sufficiency_gate_decision": _load_json(analysis_dir / "seed_gate_decision.json"),
        "artifact_policy": (
            "Compact analyzer outputs are tracked here. Raw campaign JSONL/video/output files "
            "remain outside git and need durable external storage pointers before paper-facing use."
        ),
        "out_of_scope": [
            "no benchmark campaign execution",
            "no Slurm/GPU submission",
            "no paper/dissertation claim edit",
        ],
    }
    (evidence_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_readme(evidence_dir: Path, analysis: dict[str, Any]) -> None:
    planners = ", ".join(str(planner) for planner in analysis.get("planners", []))
    seeds = analysis.get("seeds", [])
    seed_range = f"{seeds[0]}..{seeds[-1]}" if seeds else "unknown"
    replay_note = analysis.get("replay_limitation", {}).get("note", "not recorded")
    lines = [
        "# Issue #3637 Reactivity Replay Rank Study Evidence",
        "",
        "Plain-language summary: this bundle preserves compact analyzer outputs for the "
        "predeclared issue #3637 reactivity-vs-replay study. It is not a paper-facing "
        "claim by itself.",
        "",
        f"- Evidence status: `{analysis.get('claim_decision')}`",
        f"- Planners: {planners}",
        f"- Seeds: {seed_range}",
        f"- Episode rows checked: `{analysis.get('episode_count')}`",
        f"- Replay limitation: {replay_note}",
        f"- Claim boundary: {analysis.get('claim_boundary')}",
        "",
        "## Files",
        "",
        "- `analysis.json`: full analyzer summary.",
        "- `frozen_gate_input.json`: seed-sufficiency gate input frozen from the analyzer.",
        "- `rank_bootstrap_summary.json`: paired-seed bootstrap stability summary.",
        "- `per_planner_condition_metrics.csv`: compact planner/arm metrics table.",
        "- `summary.json`: promotion metadata and artifact policy.",
        "- `manifest.sha256`: checksums for promoted compact artifacts.",
        "",
        "Raw campaign outputs are not stored in this bundle.",
        "",
    ]
    (evidence_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def promote(analysis_dir: Path, evidence_dir: Path, *, overwrite: bool = False) -> dict[str, Any]:
    """Copy validated analyzer outputs into a durable evidence directory."""

    analysis = _validate_analysis(analysis_dir)
    if evidence_dir.exists():
        if not overwrite:
            raise PromotionError(f"evidence directory already exists: {evidence_dir}")
        if not evidence_dir.is_dir():
            raise PromotionError(f"evidence path exists and is not a directory: {evidence_dir}")
    evidence_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    for name in REQUIRED_ARTIFACTS:
        if name == "README.md":
            continue
        shutil.copyfile(analysis_dir / name, evidence_dir / name)
        copied.append(name)

    _write_summary(evidence_dir, analysis, analysis_dir)
    copied.append("summary.json")
    _write_readme(evidence_dir, analysis)
    copied.append("README.md")
    _write_manifest(evidence_dir, copied)

    return {
        "schema_version": PROMOTION_SCHEMA,
        "status": "promoted",
        "issue": ISSUE,
        "evidence_dir": str(evidence_dir),
        "copied_files": sorted(copied + ["manifest.sha256"]),
        "claim_boundary": analysis.get("claim_boundary"),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace files in an existing evidence directory",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the evidence promotion command."""

    args = _parse_args(argv)
    try:
        result = promote(args.analysis_dir, args.evidence_dir, overwrite=args.overwrite)
    except PromotionError as exc:
        print(json.dumps({"status": "blocked", "issue": ISSUE, "reason": str(exc)}))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
