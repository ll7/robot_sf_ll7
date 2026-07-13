<!-- AI-GENERATED (robot_sf#4882, 2026-07-13) — NEEDS-REVIEW -->
# S20-prefix vs S30 Rank Stability

Diagnostic only. Compares the planner ranking at the **S20 prefix** (seeds 111–130, the first 20 of the
S30 schedule) against the full **S30** ranking (seeds 111–140), exploiting the pre-registered
`S10 ⊂ S20 ⊂ S30` prefix-nesting property (#4304). This isolates the seed-budget effect on ranking with
everything else — scenarios, code, config — held fixed. It differs from a comparison against a separate
external S20 *campaign* (a different run); the prefix method removes campaign-to-campaign confounds.

| Metric | Kendall τ (S20-prefix vs S30) | Top rank changed? | Stability |
| --- | --- | --- | --- |
| Success rate | **1.000** | no | stable |
| Collision-event rate | **1.000** | no | stable |
| Normalized time-to-goal | **0.333** | no | **boundary-sensitive** |

## Reading

- **Success and collision orderings are fully rank-stable** from S20 to S30 (τ = 1.0): the leading
  hybrid stays first and ORCA stays last on both. The hybrid-vs-ORCA ordering the S30 wave was
  commissioned to test is not an artifact of the seed budget.
- **Normalized-time ordering is boundary-sensitive** (τ = 0.333): the top rank is unchanged but the
  mid-order reshuffles — notably the leading hybrid and PPO trade places on normalized time between
  S20-prefix and S30. Normalized time should not be read as a stable ranking axis at this budget.

Full per-arm means and rank deltas are in [`s20_s30_rank_stability.json`](./s20_s30_rank_stability.json).
This comparison is diagnostic and rewrites no paper or dissertation claim.
