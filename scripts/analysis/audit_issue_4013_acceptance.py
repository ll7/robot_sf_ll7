#!/usr/bin/env python3
"""Generate the issue #4013 acceptance and closure audit artifact.

The audit consolidates merged diagnostic evidence and the real-trajectory
readiness gate. It intentionally keeps #4013 open while real pedestrian
trajectory data, real-trajectory training, and representative evaluation are
not yet available.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_ISSUE = 4013
_DEFAULT_EVIDENCE_DIR = Path("docs/context/evidence/issue_4013_learned_model_based_planning")


def build_acceptance_audit(
    *,
    evidence_dir: str | Path = _DEFAULT_EVIDENCE_DIR,
) -> dict[str, Any]:
    """Build a conservative closure audit for issue #4013."""

    evidence_dir = Path(evidence_dir)
    comparison = _read_json_object(evidence_dir / "comparison_report.v1.json")
    readiness = _read_json_object(evidence_dir / "real_trajectory_readiness.v1.json")
    training_manifest = _read_json_object(evidence_dir / "training_manifest.v1.json")
    training_metrics = _read_json_object(evidence_dir / "training_metrics.v1.json")

    diagnostic_comparison_ready = (
        comparison.get("status") == "diagnostic_ready"
        and comparison.get("paired_seed_count", 0) >= 1
        and not comparison.get("blockers")
    )
    fallback_clean = (
        comparison.get("fallback_degraded_rows", {}).get("excluded") == 0
        and comparison.get("fallback_degraded_rows", {}).get("included_as_non_evidence") == 0
    )
    diagnostic_training_ready = training_manifest.get("evidence_tier") == "diagnostic-only"
    loss_improved = _loss_improved(training_metrics)
    real_ready = readiness.get("status") == "ready_for_real_trajectory_training"

    criteria = [
        {
            "criterion": "Short-horizon predictor contract and learned backend exist.",
            "status": "met",
            "evidence": [
                "Merged PRs #4474/#4629 established the learned-prediction MPC path.",
                "training_manifest.v1.json records a learned short-horizon predictor artifact.",
            ],
            "remaining_work": None,
        },
        {
            "criterion": "Short-horizon predictor trains or fails closed with dataset blocker.",
            "status": "met" if diagnostic_training_ready and loss_improved else "blocked",
            "evidence": [
                f"training evidence_tier={training_manifest.get('evidence_tier')!r}.",
                f"training loss improved={loss_improved!r}.",
                (
                    "real_trajectory_readiness.v1.json reports ready_for_real_trajectory_training."
                    if real_ready
                    else "real_trajectory_readiness.v1.json records the Phase 3 real-data blocker."
                ),
            ],
            "remaining_work": None
            if diagnostic_training_ready and loss_improved
            else "Regenerate diagnostic trainer evidence with decreasing loss or explicit blocker.",
        },
        {
            "criterion": "Model-based action selection runs on a smoke scenario.",
            "status": "met" if diagnostic_comparison_ready else "blocked",
            "evidence": [
                "PR #4644 loaded a checkpoint-backed planner without fallback.",
                f"comparison status={comparison.get('status')!r}.",
                f"comparison paired_seed_count={comparison.get('paired_seed_count')!r}.",
            ],
            "remaining_work": None
            if diagnostic_comparison_ready
            else "Run the #4013 model-based comparison smoke to diagnostic_ready.",
        },
        {
            "criterion": "Comparator smoke includes cv_prediction_mpc and a model-free baseline.",
            "status": "met" if _comparison_roles_met(comparison) else "blocked",
            "evidence": [
                f"comparison roles={sorted(comparison.get('roles', {}).keys())}.",
                "PR #4655 produced the 3-arm diagnostic comparison report.",
            ],
            "remaining_work": None
            if _comparison_roles_met(comparison)
            else "Regenerate comparison report with learned, cv_prediction_mpc, and model-free arms.",
        },
        {
            "criterion": "Fallback/degraded rows are excluded or marked non-evidence.",
            "status": "met" if fallback_clean else "blocked",
            "evidence": [f"fallback_degraded_rows={comparison.get('fallback_degraded_rows')!r}."],
            "remaining_work": None
            if fallback_clean
            else "Regenerate report with fallback/degraded rows excluded from evidence.",
        },
        {
            "criterion": "Claim boundary excludes large world-model and paper-grade claims.",
            "status": "met" if comparison.get("evidence_tier") == "diagnostic-only" else "blocked",
            "evidence": [
                f"comparison evidence_tier={comparison.get('evidence_tier')!r}.",
                f"comparison claim_boundary={comparison.get('claim_boundary')!r}.",
            ],
            "remaining_work": None
            if comparison.get("evidence_tier") == "diagnostic-only"
            else "Restore diagnostic-only claim boundary before treating this as issue evidence.",
        },
        {
            "criterion": "Real pedestrian trajectory dataset is reachable and checksum-pinned.",
            "status": "met" if real_ready else "blocked",
            "evidence": [
                f"readiness status={readiness.get('status')!r}.",
                f"readiness blockers={readiness.get('blockers')!r}.",
                f"manifest availability={readiness.get('availability')!r}.",
            ],
            "remaining_work": None
            if real_ready
            else (
                "Stage ETH/BIWI or approved real trajectories under "
                "ROBOT_SF_EXTERNAL_DATA_ROOT and satisfy the pinned checksum preflight."
            ),
        },
        {
            "criterion": "Real-trajectory predictor training has run on validated data.",
            "status": "blocked",
            "evidence": [
                "No checked-in evidence records a real-trajectory training run.",
                f"readiness next_action={readiness.get('next_action')!r}.",
            ],
            "remaining_work": (
                "After readiness reaches ready_for_real_trajectory_training, run the "
                "real-trajectory trainer and publish compact metrics/manifest evidence."
            ),
        },
        {
            "criterion": (
                "Representative evaluation compares learned predictor against cv_prediction_mpc "
                "and a model-free baseline."
            ),
            "status": "blocked",
            "evidence": [
                "Existing comparison_report.v1.json is diagnostic synthetic/checkpoint smoke evidence.",
                "Issue thread after PR #4712 still requires real-trajectory training plus representative evaluation.",
            ],
            "remaining_work": (
                "Run the representative real-trajectory evaluation against cv_prediction_mpc "
                "and one model-free baseline after real-data training."
            ),
        },
    ]

    blockers = [
        item["remaining_work"]
        for item in criteria
        if item["status"] != "met" and item.get("remaining_work")
    ]
    closure_status = "complete" if not blockers else "partial"

    return {
        "schema_version": "issue_4013.acceptance_audit.v1",
        "issue": _ISSUE,
        "generated_at": datetime.now(UTC).isoformat(),
        "closure_status": closure_status,
        "claim_boundary": (
            "Closure audit and integration report only. Diagnostic smoke evidence is not "
            "benchmark, navigation-quality, paper, or dissertation evidence."
        ),
        "merged_pr_evidence": [
            {"pr": 4474, "evidence": "learned-prediction MPC comparator preflight"},
            {"pr": 4587, "evidence": "fail-closed diagnostic comparison report contract"},
            {"pr": 4629, "evidence": "seeded synthetic short-horizon predictor training"},
            {"pr": 4644, "evidence": "checkpoint-backed planner plan() smoke"},
            {"pr": 4655, "evidence": "3-arm diagnostic comparison report"},
            {"pr": 4665, "evidence": "BYO real-trajectory manifest and readiness checker"},
            {"pr": 4679, "evidence": "official ETH CVL acquisition URL and instructions"},
            {"pr": 4700, "evidence": "real-trajectory trainer data path"},
            {"pr": 4704, "evidence": "validated manifest requires reachable staging directory"},
            {"pr": 4712, "evidence": "validated manifest requires pinned tree checksum"},
        ],
        "criteria": criteria,
        "blockers_remaining": blockers,
        "new_blockers": [],
        "intentional_non_actions": [
            "No raw external dataset staging or redistribution.",
            "No full benchmark campaign run.",
            "No Slurm/GPU submission.",
            "No paper/dissertation claim edits.",
        ],
        "next_empirical_action": (
            "Provide a reachable checksum-pinned ETH/BIWI staging root, run real-trajectory "
            "predictor training, then run representative evaluation versus cv_prediction_mpc "
            "and one model-free baseline."
        ),
        "forbidden_actions_confirmed": {
            "full_benchmark_campaign_run": False,
            "slurm_or_gpu_submission": False,
            "paper_or_dissertation_claim_edit": False,
        },
    }


def write_acceptance_audit(
    *,
    evidence_dir: str | Path = _DEFAULT_EVIDENCE_DIR,
    output_json: str | Path | None = None,
    output_markdown: str | Path | None = None,
) -> dict[str, Any]:
    """Build and write JSON plus Markdown audit artifacts."""

    evidence_dir = Path(evidence_dir)
    audit = build_acceptance_audit(evidence_dir=evidence_dir)
    output_json = (
        Path(output_json) if output_json is not None else evidence_dir / "acceptance_audit.v1.json"
    )
    output_markdown = (
        Path(output_markdown)
        if output_markdown is not None
        else evidence_dir / "acceptance_audit.v1.md"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_markdown.write_text(_render_markdown(audit), encoding="utf-8")
    return audit


def _comparison_roles_met(comparison: Mapping[str, Any]) -> bool:
    roles = comparison.get("roles", {})
    return isinstance(roles, Mapping) and {
        "learned_prediction_mpc",
        "cv_prediction_mpc",
        "model_free_baseline",
    }.issubset(roles)


def _loss_improved(training_metrics: Mapping[str, Any]) -> bool:
    initial = training_metrics.get("initial_train_loss", training_metrics.get("initial_loss"))
    final = training_metrics.get("final_train_loss", training_metrics.get("final_loss"))
    return isinstance(initial, int | float) and isinstance(final, int | float) and final < initial


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise OSError(f"could not read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"could not parse {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _escape_table_cell(text: str) -> str:
    """Escape pipe characters so JSON-derived content cannot corrupt the Markdown table."""

    return text.replace("|", "\\|")


def _render_markdown(audit: Mapping[str, Any]) -> str:
    lines = [
        "# Issue #4013 Acceptance Audit",
        "",
        "This audit maps issue #4013 acceptance criteria to merged PR evidence and the current "
        "real-trajectory readiness gate. It is conservative: diagnostic smoke evidence is not "
        "treated as benchmark, navigation-quality, paper, or dissertation evidence.",
        "",
        f"- Closure status: `{audit['closure_status']}`.",
        f"- Claim boundary: {audit['claim_boundary']}",
        f"- Next empirical action: {audit['next_empirical_action']}",
        "",
        "## Criteria",
        "",
        "| Criterion | Status | Evidence | Remaining work |",
        "| --- | --- | --- | --- |",
    ]
    for item in audit["criteria"]:
        evidence = "<br>".join(_escape_table_cell(str(part)) for part in item["evidence"])
        remaining = _escape_table_cell(item.get("remaining_work") or "None.")
        criterion = _escape_table_cell(item["criterion"])
        lines.append(f"| {criterion} | `{item['status']}` | {evidence} | {remaining} |")

    lines.extend(["", "## Merged PR Evidence", ""])
    for item in audit["merged_pr_evidence"]:
        lines.append(f"- PR #{item['pr']}: {item['evidence']}.")

    lines.extend(["", "## Blockers Remaining", ""])
    if audit["blockers_remaining"]:
        lines.extend(f"- {blocker}" for blocker in audit["blockers_remaining"])
    else:
        lines.append("- None.")

    lines.extend(["", "## Intentional Non-Actions", ""])
    lines.extend(f"- {item}" for item in audit["intentional_non_actions"])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=_DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-markdown", type=Path, default=None)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    if args.write:
        audit = write_acceptance_audit(
            evidence_dir=args.evidence_dir,
            output_json=args.output_json,
            output_markdown=args.output_markdown,
        )
    else:
        audit = build_acceptance_audit(evidence_dir=args.evidence_dir)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
