#!/usr/bin/env python3
"""Generate the issue #4016 acceptance-criteria audit artifact.

The audit is intentionally conservative: smoke diagnostics can satisfy smoke
criteria, but they do not close benchmark-strength measured comparison criteria.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_ISSUE = 4016
_REQUIRED_METRICS = (
    "success_rate",
    "collision_rate",
    "near_miss_rate",
    "mean_min_clearance",
    "mean_path_efficiency",
)


def build_acceptance_audit(
    *,
    evidence_dir: str | Path = "docs/context/evidence/issue_4016_distributional_rl_smoke",
) -> dict[str, Any]:
    """Build the conservative issue #4016 closure audit from checked-in evidence."""
    evidence_dir = Path(evidence_dir)
    summary = _read_json_object(evidence_dir / "summary.json")
    comparison = _read_json_object(evidence_dir / "distributional_rl_risk_comparison.json")
    mean_manifest = _read_json_object(evidence_dir / "qr_dqn_mean_manifest.json")
    cvar_manifest = _read_json_object(evidence_dir / "qr_dqn_cvar_manifest.json")

    criteria = _criteria(summary, comparison, mean_manifest, cvar_manifest)
    closure_status = "complete" if all(item["status"] == "met" for item in criteria) else "partial"
    blockers = [
        item["remaining_work"]
        for item in criteria
        if item["status"] != "met" and item.get("remaining_work")
    ]
    return {
        "schema_version": "issue_4016.acceptance_audit.v1",
        "issue": _ISSUE,
        "generated_at": datetime.now(UTC).isoformat(),
        "closure_status": closure_status,
        "claim_boundary": (
            "closure audit of merged implementation evidence; diagnostic-only smoke evidence is "
            "not benchmark-strength safety evidence"
        ),
        "merged_prs_reviewed": [
            {
                "pr": 4157,
                "evidence": "distributional RL primitives: lattice, quantile critic, risk objectives",
            },
            {
                "pr": 4215,
                "evidence": "QR-DQN smoke trainer and runtime adapter",
            },
            {
                "pr": 4283,
                "evidence": "trainer/adapter handoff proof",
            },
            {
                "pr": 4476,
                "evidence": "diagnostic mean-vs-risk comparison report contract",
            },
            {
                "pr": 4672,
                "evidence": "real smoke checkpoint materialized through mean and CVaR manifests",
            },
        ],
        "criteria": criteria,
        "blockers": blockers,
        "forbidden_actions_confirmed": {
            "full_benchmark_campaign_run": False,
            "slurm_or_gpu_submission": False,
            "paper_or_dissertation_claim_edit": False,
        },
    }


def write_acceptance_audit(
    *,
    evidence_dir: str | Path = "docs/context/evidence/issue_4016_distributional_rl_smoke",
    output_json: str | Path | None = None,
    output_markdown: str | Path | None = None,
) -> dict[str, Any]:
    """Build and write the JSON and Markdown audit artifacts."""
    evidence_dir = Path(evidence_dir)
    audit = build_acceptance_audit(evidence_dir=evidence_dir)
    output_json = (
        Path(output_json) if output_json is not None else evidence_dir / "acceptance_audit.json"
    )
    output_markdown = (
        Path(output_markdown)
        if output_markdown is not None
        else evidence_dir / "acceptance_audit.md"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_markdown.write_text(_render_markdown(audit), encoding="utf-8")
    return audit


def _criteria(
    summary: Mapping[str, Any],
    comparison: Mapping[str, Any],
    mean_manifest: Mapping[str, Any],
    cvar_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    same_checkpoint = mean_manifest.get("checkpoint_path") == cvar_manifest.get("checkpoint_path")
    same_seed = mean_manifest.get("seed") == cvar_manifest.get("seed")
    same_steps = mean_manifest.get("total_timesteps") == cvar_manifest.get("total_timesteps")
    fallback_clean = (
        summary.get("fallback_or_degraded") is False
        and comparison.get("fallback_degraded_rows", {}).get("excluded") == 0
        and comparison.get("fallback_degraded_rows", {}).get("included_as_non_evidence") == 0
        and mean_manifest.get("fallback_or_degraded") is False
        and cvar_manifest.get("fallback_or_degraded") is False
    )
    metric_keys_present = _has_metrics(mean_manifest) and _has_metrics(cvar_manifest)
    diagnostic_only = (
        summary.get("evidence_tier") == "diagnostic-only"
        and comparison.get("evidence_tier") == "diagnostic-only"
        and comparison.get("effect", {}).get("benchmark_safety_claim") is False
    )

    return [
        {
            "criterion": "QR-DQN-style distributional critic trains on a smoke scenario.",
            "status": "met",
            "evidence": [
                "PR #4215 added the QR-DQN smoke trainer path.",
                "PR #4672 records output/models/distributional_rl/issue_4016/training_manifest.json "
                "as the source for checked-in smoke manifests.",
                f"summary.json records fallback_or_degraded={summary.get('fallback_or_degraded')!r}.",
            ],
        },
        {
            "criterion": "Risk-sensitive selection runs in map_runner/runtime adapter.",
            "status": "met",
            "evidence": [
                "PR #4215 added robot_sf/baselines/distributional_rl.py and "
                "robot_sf/benchmark/map_runner_policies/distributional_rl.py.",
                "PR #4672 materialized cvar_lower runtime diagnostics from the merged adapter.",
            ],
        },
        {
            "criterion": "Same checkpoint can be evaluated in mean and cvar_lower selection modes.",
            "status": "met" if same_checkpoint and same_seed and same_steps else "blocked",
            "evidence": [
                f"mean checkpoint: {mean_manifest.get('checkpoint_path')}",
                f"cvar checkpoint: {cvar_manifest.get('checkpoint_path')}",
                f"matched seed={same_seed}, matched total_timesteps={same_steps}.",
            ],
            "remaining_work": None
            if same_checkpoint and same_seed and same_steps
            else "Regenerate paired manifests from one checkpoint with matched seed and timesteps.",
        },
        {
            "criterion": "Mean-value comparator on the same discrete action lattice is available.",
            "status": "met" if mean_manifest.get("risk_objective") == "mean" else "blocked",
            "evidence": [
                "PR #4672 added configs/baselines/distributional_rl_issue_4016_mean.yaml.",
                f"qr_dqn_mean_manifest.json risk_objective={mean_manifest.get('risk_objective')!r}.",
            ],
            "remaining_work": None
            if mean_manifest.get("risk_objective") == "mean"
            else "Materialize a mean-selection manifest on the same action lattice.",
        },
        {
            "criterion": "Matched-seed diagnostic comparison is recorded.",
            "status": "met"
            if comparison.get("effect", {}).get("comparison_status") == "valid_diagnostic"
            else "blocked",
            "evidence": [
                "PR #4476 added scripts/analysis/compare_distributional_rl_issue_4016.py.",
                "PR #4672 checked in distributional_rl_risk_comparison.json and .md.",
                f"comparison_status={comparison.get('effect', {}).get('comparison_status')!r}.",
            ],
            "remaining_work": None
            if comparison.get("effect", {}).get("comparison_status") == "valid_diagnostic"
            else "Run the diagnostic comparison generator on matched mean and cvar manifests.",
        },
        {
            "criterion": (
                "Reports include collision, near-miss, min-clearance, success/progress, and "
                "path-efficiency tradeoffs."
            ),
            "status": "partial",
            "evidence": [
                f"Required metric keys present in smoke manifests: {metric_keys_present}.",
                "PR #4672 gate review states these are synthetic-observation placeholders, not "
                "measured benchmark safety outcomes.",
            ],
            "remaining_work": (
                "Run or ingest a benchmark-runner measured comparison for mean and cvar_lower modes "
                "on safety-relevant scenarios; keep fallback/degraded rows excluded or marked "
                "non-evidence."
            ),
        },
        {
            "criterion": "Fallback/degraded rows are explicitly excluded or marked non-evidence.",
            "status": "met" if fallback_clean else "blocked",
            "evidence": [
                f"summary fallback_or_degraded={summary.get('fallback_or_degraded')!r}.",
                f"comparison fallback_degraded_rows={comparison.get('fallback_degraded_rows')!r}.",
            ],
            "remaining_work": None
            if fallback_clean
            else "Regenerate comparison with degraded/fallback rows excluded or marked non-evidence.",
        },
        {
            "criterion": "Claim boundary is explicit and does not promote paper-facing safety claims.",
            "status": "met" if diagnostic_only else "blocked",
            "evidence": [
                f"summary evidence_tier={summary.get('evidence_tier')!r}.",
                f"comparison benchmark_safety_claim="
                f"{comparison.get('effect', {}).get('benchmark_safety_claim')!r}.",
            ],
            "remaining_work": None
            if diagnostic_only
            else "Restore diagnostic-only claim boundary before using this evidence.",
        },
    ]


def _has_metrics(manifest: Mapping[str, Any]) -> bool:
    metrics = manifest.get("metrics")
    return isinstance(metrics, Mapping) and all(key in metrics for key in _REQUIRED_METRICS)


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


def _render_markdown(audit: Mapping[str, Any]) -> str:
    lines = [
        "# Issue #4016 Acceptance Audit",
        "",
        "This audit maps issue #4016 closure criteria to merged implementation evidence. "
        "It is conservative: diagnostic smoke evidence is not treated as benchmark-strength "
        "safety evidence.",
        "",
        f"- Closure status: `{audit['closure_status']}`.",
        f"- Claim boundary: {audit['claim_boundary']}.",
        "- No full benchmark campaign, Slurm/GPU submission, or paper/dissertation claim edit "
        "was performed.",
        "",
        "## Merged PR Evidence",
        "",
    ]
    for item in audit["merged_prs_reviewed"]:
        lines.append(f"- PR #{item['pr']}: {item['evidence']}.")
    lines.extend(["", "## Criteria", ""])
    lines.append("| Criterion | Status | Evidence | Remaining work |")
    lines.append("| --- | --- | --- | --- |")
    for item in audit["criteria"]:
        evidence = "<br>".join(str(value) for value in item["evidence"])
        remaining = item.get("remaining_work") or "None"
        lines.append(f"| {item['criterion']} | `{item['status']}` | {evidence} | {remaining} |")
    lines.extend(["", "## Closure Decision", ""])
    if audit["closure_status"] == "complete":
        lines.append("All listed criteria are met; #4016 can close when this audit is accepted.")
    else:
        lines.append(
            "#4016 should stay open. The remaining blocker is measured benchmark-runner evidence "
            "for the mean-vs-CVaR safety-relevant comparison; the current checked-in smoke metrics "
            "are diagnostic placeholders."
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line audit generator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-dir",
        default="docs/context/evidence/issue_4016_distributional_rl_smoke",
        help="Issue #4016 evidence directory.",
    )
    parser.add_argument("--output-json", help="Acceptance audit JSON output path.")
    parser.add_argument("--output-markdown", help="Acceptance audit Markdown output path.")
    args = parser.parse_args(argv)
    audit = write_acceptance_audit(
        evidence_dir=args.evidence_dir,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
    )
    print(json.dumps({"closure_status": audit["closure_status"], "issue": _ISSUE}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
