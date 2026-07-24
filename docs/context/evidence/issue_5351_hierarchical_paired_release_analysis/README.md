<!-- AI-GENERATED (robot_sf#5351) - NEEDS-REVIEW -->
# Issue #5351 Hierarchical Paired Release Analysis Report

This directory registers the deterministic, checksum-pinned statistical analysis artifacts for issue #5351 over benchmark release `0.0.3.post1`.

> [!NOTE]
> Claim boundary: this is statistical analysis ON TOP of frozen release metrics. Output remains `blocked_review_pending` and promotes no benchmark, paper, or dissertation claim automatically.

## Successor Release Inputs

- Release Tag: `0.0.3.post1`
- Publication Commit: `ded9027d2928512c14bc241397e0ab1d8f586654`
- Typed Ledger Rows: [`docs/context/evidence/issue_5351_hierarchical_paired_release_analysis/successor_rows.jsonl`](successor_rows.jsonl)
- Rows SHA-256: `c45c2ed8defdadaf47c001277e6bf9ca0c2238c101570d1d64be8015060febea`
- Total Episode Rows: `20160` (14 arms × 1,440 episodes)

## Protocol Conformance

| Protocol Element | Declared Delivery | Status |
| --- | --- | --- |
| `paired_effects` | paired risk-difference and odds-risk-ratio table on matched cells | `delivered_analysis_pending_human_review` |
| `hierarchical_intervals` | scenario-family cluster hierarchical-bootstrap interval table | `delivered_analysis_pending_human_review` |
| `sensitivity_analyses` | seed-level and family-level sensitivity table | `delivered_analysis_pending_human_review` |
| `multiplicity_control` | predeclared planner-pair and family-comparison multiplicity table | `delivered_analysis_pending_human_review` |
| `practical_effect_reporting` | practical-effect threshold comparison table | `delivered_analysis_pending_human_review` |
| `censored_completion_time` | censored timeout-aware completion-time summary | `delivered_analysis_pending_human_review` |
| `normalized_near_miss_exposure` | near-miss exposure normalization summary by time, distance, and opportunity | `delivered_analysis_pending_human_review` |
| `claim_gate_and_conformance` | machine-readable claim gate and protocol conformance table | `delivered_analysis_pending_human_review` |

## Summary of Analysis

- Total Paired Comparisons Evaluated: 39
- Multiplicity Method: holm_step_down
- Practically Separable Effects (Clear min_risk_difference >= 0.02): 13 / 39
- Exposure Handling: source near-miss event counts are retained; computed zero exposure remains zero, while non-derivable interaction opportunity is explicitly excluded without a synthetic denominator.
- Claim Gate Status: `blocked_review_pending`
- Claim Gate Reason: analysis executed over successor rows; claim promotion requires human review of the deterministic report

## Reproducibility

To re-run and verify determinism against release `0.0.3.post1`:

```bash
uv run python scripts/analysis/run_hierarchical_paired_release_analysis_issue_5351.py \
  --repo-root .
```
