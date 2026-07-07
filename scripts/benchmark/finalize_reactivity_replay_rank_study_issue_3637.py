#!/usr/bin/env python3
"""Finalize issue #3637 post-run analysis and compact evidence promotion.

This command does not run the benchmark campaign. It is the post-run bridge for
the predeclared campaign output: validate the completed campaign with the
issue-specific analyzer, promote the compact reviewable artifacts, and emit one
machine-readable handoff report.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_ANALYZER_PATH = (
    _REPO_ROOT / "scripts" / "benchmark" / ("analyze_reactivity_replay_rank_study_issue_3637.py")
)
_PROMOTER_PATH = (
    _REPO_ROOT / "scripts" / "benchmark" / ("promote_reactivity_replay_rank_study_issue_3637.py")
)


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ANALYZER = _load_module(_ANALYZER_PATH, "_issue_3637_analyzer")
_PROMOTER = _load_module(_PROMOTER_PATH, "_issue_3637_promoter")

ISSUE = 3637
SCHEMA_VERSION = "issue-3637-reactivity-replay-rank-finalization.v1"

DEFAULT_PACKET = _ANALYZER.DEFAULT_PACKET
DEFAULT_CAMPAIGN_DIR = _ANALYZER.DEFAULT_CAMPAIGN_DIR
DEFAULT_CAMPAIGN_REPORT = _ANALYZER.DEFAULT_CAMPAIGN_REPORT
DEFAULT_ANALYSIS_DIR = _ANALYZER.DEFAULT_OUTPUT_DIR
DEFAULT_EVIDENCE_DIR = _PROMOTER.DEFAULT_EVIDENCE_DIR


class FinalizationError(RuntimeError):
    """Raised when issue #3637 post-run finalization cannot complete."""


def finalize(
    *,
    packet_path: Path,
    campaign_dir: Path,
    campaign_report: Path,
    analysis_dir: Path,
    evidence_dir: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run analyzer and promoter as one post-campaign finalization step."""
    try:
        analysis = _ANALYZER.analyze(
            packet_path=packet_path,
            campaign_dir=campaign_dir,
            campaign_report=campaign_report,
            output_dir=analysis_dir,
        )
        promotion = _PROMOTER.promote(
            analysis_dir,
            evidence_dir,
            overwrite=overwrite,
        )
    except (_ANALYZER.AnalysisInputError, _PROMOTER.PromotionError) as exc:
        raise FinalizationError(str(exc)) from exc

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "finalized",
        "issue": ISSUE,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "packet": str(packet_path),
        "campaign_dir": str(campaign_dir),
        "campaign_report": str(campaign_report),
        "analysis_dir": str(analysis_dir),
        "evidence_dir": str(evidence_dir),
        "analysis": {
            "schema_version": analysis.get("schema_version"),
            "episode_count": analysis.get("episode_count"),
            "expected_episode_count": analysis.get("expected_episode_count"),
            "claim_decision": analysis.get("claim_decision"),
            "claim_boundary": analysis.get("claim_boundary"),
            "seed_sufficiency_gate_decision": analysis.get("seed_sufficiency_gate_decision"),
            "replay_limitation": analysis.get("replay_limitation"),
        },
        "promotion": promotion,
        "closure_decision": "review_promoted_evidence_before_closure",
        "remaining_acceptance_criteria": [
            "Maintainer/reviewer must inspect promoted evidence and decide whether "
            "the seed-sufficiency result supports closing issue #3637.",
            "No paper-facing claim is established by this command alone.",
        ],
        "forbidden_actions_confirmed": {
            "benchmark_campaign_execution": False,
            "slurm_gpu_submission": False,
            "paper_dissertation_claim_edit": False,
        },
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--campaign-dir", type=Path, default=DEFAULT_CAMPAIGN_DIR)
    parser.add_argument("--campaign-report", type=Path, default=DEFAULT_CAMPAIGN_REPORT)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace files in an existing evidence directory",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the issue #3637 post-run finalizer CLI."""
    args = _parse_args(argv)
    try:
        result = finalize(
            packet_path=args.packet,
            campaign_dir=args.campaign_dir,
            campaign_report=args.campaign_report,
            analysis_dir=args.analysis_dir,
            evidence_dir=args.evidence_dir,
            overwrite=args.overwrite,
        )
    except FinalizationError as exc:
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "blocked",
                    "issue": ISSUE,
                    "reason": str(exc),
                    "next_empirical_action": (
                        "Run the predeclared issue #3637 campaign, then rerun this finalizer."
                    ),
                },
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
