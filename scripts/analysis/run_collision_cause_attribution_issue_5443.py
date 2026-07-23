"""Drive the rule-based cause analyser over the #5443 injected-fault fixtures.

This runner wires the issue #5443 stack end to end on CPU:

    fixtures (kinematic fault injections)
      -> last-avoidable counterfactual replay (per fixture)
      -> deterministic rule-based cause analyser
      -> AttributionVerdict set
      -> frozen validation report (``collision_cause_attribution.py``)

It runs no simulation, trains no classifier, and makes no campaign or paper-grade
claim. It is *controlled-fixture injected-fault validation only* (criterion 3-5).

The analyser never reads the manifest's ground-truth ``cause_class``: the fixture
builders provide only observable fault evidence plus counterfactual repairs, and
the analyser attributes from computed replay/repair evidence. The frozen scoring
module then compares the analyser's verdicts against the manifest answer key.

Usage::

    # Emit analyser verdicts only (for inspection or external scoring):
    uv run python scripts/analysis/run_collision_cause_attribution_issue_5443.py \\
        --verdicts output/issue_5443_verdicts.json

    # Emit verdicts and score them against the frozen manifest, exiting non-zero
    # on a ``revise`` verdict (the issue stop rule):
    uv run python scripts/analysis/run_collision_cause_attribution_issue_5443.py \\
        --verdicts output/issue_5443_verdicts.json --score --out output/issue_5443_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from robot_sf.benchmark.collision_cause_analyser import analyse_suite
from robot_sf.benchmark.collision_cause_attribution import (
    REPORT_STATUS_SCORED,
    VERDICT_PASS,
    build_validation_report,
)
from robot_sf.benchmark.last_avoidable_fixtures import build_collision_cause_fixtures

_DEFAULT_MANIFEST = Path("tests/benchmark/fixtures/collision_cause_attribution_manifest_5443.json")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verdicts",
        type=Path,
        default=None,
        help="Optional path to write the analyser verdicts JSON.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_DEFAULT_MANIFEST,
        help="Path to the frozen ground-truth fixture manifest JSON (for scoring).",
    )
    parser.add_argument(
        "--score",
        action="store_true",
        help="Score the analyser verdicts against the frozen manifest and apply the stop rule.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write the scored validation report JSON (requires --score).",
    )
    return parser.parse_args(argv)


def build_verdicts_payload() -> dict[str, Any]:
    """Build analyser verdicts over all 14 fixtures with provenance.

    Returns:
        A JSON-safe mapping with the analyser schema version, the verdict list,
        and per-fixture computed evidence (for transparency).
    """
    fixtures = build_collision_cause_fixtures()
    result = analyse_suite(fixtures)
    evidence = [
        {
            "fixture_id": ev.fixture_id,
            "replay_verdict": ev.replay.verdict,
            "replay_t_uca": ev.replay.t_uca,
            "replay_t_inevitable": ev.replay.t_inevitable,
            "replay_t_contact": ev.replay.config.t_contact,
            "replay_deterministic": ev.replay.determinism.deterministic,
            "decisive_faults": [f.fault_type for f in ev.decisive_faults],
            "present_faults": [f.fault_type for f in ev.present_faults],
            "metric_quirk": ev.metric_quirk,
            "avoidable_pred": ev.avoidable_pred,
        }
        for ev in result.evidence
    ]
    return {
        "schema_version": "collision_cause_analyser_run.v1",
        "analyser": "rule_based_kinematic_replay",
        "issue": 5443,
        "claim_boundary": (
            "controlled-fixture injected-fault validation only; not a real-trace "
            "root-cause claim; assigns no legal or moral fault"
        ),
        "verdicts": result.verdict_mappings(),
        "evidence": evidence,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the analyser and optionally score the verdicts.

    Returns:
        Process exit code: 0 when no scoring is requested, or when a scored
        report returns ``pass``; 1 when a scored report returns ``revise``.
    """
    args = _parse_args(argv)

    payload = build_verdicts_payload()

    if args.verdicts is not None:
        args.verdicts.parent.mkdir(parents=True, exist_ok=True)
        args.verdicts.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    if not args.score:
        # Print just the verdicts for inspection.
        print(json.dumps(payload["verdicts"], indent=2, sort_keys=True))
        return 0

    fixtures = json.loads(args.manifest.read_text(encoding="utf-8"))["fixtures"]
    report = build_validation_report(fixtures, payload["verdicts"])
    text = json.dumps(report.to_dict(), indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)

    if report.status == REPORT_STATUS_SCORED and report.report is not None:
        return 0 if report.report.verdict == VERDICT_PASS else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
