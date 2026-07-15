<!-- AI-GENERATED: diagnostic-only integration report; NEEDS-REVIEW: maintainer/domain sign-off before reuse. -->
# Job 13274 Claim Decision and S30 Disposition

This integration report consolidates the completed job-13274 harvest, the preserved
rank-stability packet, and the repository's existing S30 ruling. It is a review surface, not a
new campaign result.

## Decision

- **Evidence classification:** `diagnostic-only`.
- **Rank profile:** constraints-first success ranking. SNQI ranks remain
  `blocked_invalid_metric` because the source contract failed with
  `rank_alignment_spearman=-0.2` below the `0.3` threshold under `enforcement=warn`.
- **S30 disposition:** `deferred_for_dissertation_draft`.
- **Escalation condition:** run the predeclared S30 schedule only if a specific strict ordering
  claim needs at least 30 seeds or a reviewer explicitly requires it. This report does not submit
  that run.

The disposition follows the maintainer S30 ruling recorded in
[`docs/context/issue_1554_s20_s30_seed_budget.md`](../../issue_1554_s20_s30_seed_budget.md):
S20 is the dissertation-draft evidence tier and S30 remains a reversible, predeclared future
escalation. The job-13274 packet does not override that ruling: 45 of 280 adjacent comparisons
are CI-separable, 235 are not distinguishable at this budget, only 12 of 35 scenario top-1
planners are stable, and 32 of 35 scenarios show rank movement under resampling.

## Claim card

### Supported for review

- The verified 8,640-episode S20 harvest contains 315 counted cells and no fallback/degraded
  cells.
- Constraints-first per-cell confidence intervals and rank-stability statistics are reproducible
  from the preserved compact artifacts.
- Adjacent comparisons must be reported with the CI boundary above; the result does not support a
  strict total planner ordering at this budget.

### Intentionally not supported

- No planner-superiority, benchmark, paper, or dissertation claim is promoted.
- No SNQI rank or SNQI-based adjacent-order claim is promoted.
- No S30 result is implied: no S30 campaign rows exist in this bundle.

## Blockers and next action

| State | Item | Handling |
| --- | --- | --- |
| Remaining | Maintainer/domain claim-card review | Review this card against the #3216 claim boundary before reuse. |
| Remaining | Headline propagation to #3216 | Post the compact headline and durable paths when issue-comment authorization is available. |
| Intentional | S30 is unexecuted | Keep the schedule predeclared; do not queue compute automatically. |
| Intentional | SNQI contract failure | Keep SNQI diagnostic-only until its contract is repaired and revalidated. |

The next empirical action is therefore conditional, not automatic: if review identifies a strict
adjacent ordering that matters to the draft or a reviewer requests it, authorize the existing
S30 protocol, preserve its artifacts, and rerun the same constraints-first analysis. Otherwise,
retain this S20 diagnostic result and report its uncertainty boundaries.

## Provenance and scope

- Source bundle: [`README.md`](README.md), [`result.json`](result.json), [`report.md`](report.md),
  and [`analysis_provenance.json`](analysis_provenance.json).
- Related headline context: [`docs/context/issue_3216_headline_ci_rank_stability.md`](../../issue_3216_headline_ci_rank_stability.md).
- This report changes no metric semantics, campaign rows, benchmark configuration, or paper/
  dissertation text.
- No full benchmark campaign, Slurm/GPU submission, raw-data promotion, or issue/PR comment is
  part of this slice.
