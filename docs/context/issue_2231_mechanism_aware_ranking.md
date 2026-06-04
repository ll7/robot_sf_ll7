# Issue #2231 Mechanism-Aware Ranking Comparison

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/2231>

Status: diagnostic synthesis from tracked issue #1353 evidence.

## Question

Do aggregate success/collision rankings hide planner differences that become visible when local
navigation mechanism diagnostics are included?

This note compares two ranking views on one existing tracked evidence surface:

- nominal aggregate ranking: success rate first, then collision rate, then SNQI as a tie-breaker;
- mechanism-aware ranking: SNQI planner ordering, SNQI contract status, component sensitivity,
  execution availability, projection/infeasible rates, and stress-surface stability.

The result is a diagnostic ranking comparison, not a benchmark leaderboard and not a universal
planner score.

## Evidence Surface

Primary tracked bundle:
[docs/context/evidence/issue_1353_broader_amv_2026-05-26/](evidence/issue_1353_broader_amv_2026-05-26/)

The bundle preserves compact nominal and stress campaign evidence:

- nominal broader-baseline surface:
  [nominal/campaign_analysis.md](evidence/issue_1353_broader_amv_2026-05-26/nominal/campaign_analysis.md),
  [nominal/campaign_table.md](evidence/issue_1353_broader_amv_2026-05-26/nominal/campaign_table.md),
  [nominal/snqi_diagnostics.md](evidence/issue_1353_broader_amv_2026-05-26/nominal/snqi_diagnostics.md);
- stress broader-baseline surface:
  [stress/campaign_analysis.md](evidence/issue_1353_broader_amv_2026-05-26/stress/campaign_analysis.md),
  [stress/campaign_table.md](evidence/issue_1353_broader_amv_2026-05-26/stress/campaign_table.md),
  [stress/snqi_diagnostics.md](evidence/issue_1353_broader_amv_2026-05-26/stress/snqi_diagnostics.md).

The evidence includes `goal`, `orca`, `ppo`, `prediction_planner`, `sacadrl`,
`social_force`, `socnav_sampling`, and a `socnav_bench` row that is `not_available`.
The comparison below excludes `socnav_bench` from ranking because it has zero episodes and an
explicit missing-assets availability reason.

## Nominal Surface

The nominal surface shows a clear divergence between success-first ordering and SNQI ordering.

| Planner | Success-first rank | Success | Collision | SNQI rank | SNQI mean | Diagnostic signal |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `ppo` | 1 | `0.3333` | `0.0833` | 4 | `-0.1007` | Best aggregate success tie, but lower SNQI and high jerk in the campaign table. |
| `prediction_planner` | 2 | `0.3333` | `0.2500` | 6 | `-0.1134` | Tied success, worse collision rate, and slowest runtime hotspot. |
| `orca` | 3 | `0.2500` | `0.0833` | 5 | `-0.1129` | Lower success than PPO, low collision rate, but adapter projection/infeasible rate is `0.6780`. |
| `sacadrl` | 4 | `0.2500` | `0.2500` | 2 | `-0.0891` | Weaker success/collision view, stronger SNQI view. |
| `socnav_sampling` | 5 | `0.2500` | `0.3333` | 1 | `-0.0814` | Best SNQI despite high collision rate and adapter projection/infeasible rate `0.9866`. |
| `goal` | 6 | `0.2500` | `0.3333` | 3 | `-0.0956` | Native core row, but collision-heavy in aggregate. |
| `social_force` | 7 | `0.0000` | `0.0000` | 7 | `-0.1291` | Agreement that it should not rank highly: no successes and worst SNQI ordering. |

Nominal SNQI contract status is `warn`, rank alignment Spearman is `0.4643`, and the dominant
component is `time_penalty`. The SNQI diagnostics also report that planner ordering changed under
six one-at-a-time weight ablations. That makes the SNQI view useful as a diagnostic lens, but not
stable enough to promote as a replacement leaderboard.

## Stress Surface

The stress surface sharpens the divergence. PPO is still best by success rate, but ORCA becomes the
SNQI-ranked leader and the SNQI contract fails.

| Planner | Success-first rank | Success | Collision | SNQI rank | SNQI mean | Diagnostic signal |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `ppo` | 1 | `0.2222` | `0.1667` | 3 | `-0.1538` | Best success, but not best SNQI; success-sensitive ranking is unstable under ablation. |
| `orca` | 2 | `0.1667` | `0.0764` | 1 | `-0.1509` | Best SNQI and lowest collision rate among successful planners. |
| `socnav_sampling` | 3 | `0.1528` | `0.5417` | 2 | `-0.1535` | Near-top SNQI despite the highest collision rate, a mechanism-aware caveat rather than a promotion signal. |
| `prediction_planner` | 4 | `0.0694` | `0.2153` | 6 | `-0.1947` | Lower success, middling collisions, and largest runtime hotspot. |
| `sacadrl` | 5 | `0.0208` | `0.3472` | 5 | `-0.1737` | Agreement that stress behavior is weak. |
| `goal` | 6 | `0.0000` | `0.2500` | 4 | `-0.1656` | SNQI view is less punitive than success-first ranking, but zero success blocks promotion. |
| `social_force` | 7 | `0.0000` | `0.2500` | 7 | `-0.2062` | Agreement that it remains weak under stress. |

Stress SNQI contract status is `fail`, rank alignment Spearman is `0.2857`, and the dominant
component is again `time_penalty`. Planner ordering changed under five one-at-a-time weight
ablations. The top sensitivity dimensions are success and collision weights, so mechanism-aware
ranking is not independent of ordinary outcomes; it reframes them with time, near-miss, comfort,
force, and jerk penalties.

## Interpretation

Observed agreement:

- `social_force` is weak in both views on both surfaces.
- `ppo` remains a high-priority candidate because it has the strongest success rate on both
  surfaces.
- `prediction_planner` should not be promoted from aggregate success alone because its nominal
  success tie comes with a worse collision rate, weaker SNQI rank, and the dominant runtime hotspot.

Observed divergence:

- Nominal aggregate ranking favors `ppo` and `prediction_planner`, while nominal SNQI favors
  `socnav_sampling`, `sacadrl`, and `goal`.
- Stress aggregate ranking favors `ppo`, while stress SNQI favors `orca`.
- `socnav_sampling` illustrates why mechanism-aware ranking cannot be read as simple promotion:
  it ranks well by SNQI while carrying high collision rates and adapter projection/infeasible rates.

Design decision:

- Treat `ppo` and `orca` as the most useful next comparators on this evidence surface: PPO is the
  success-first leader, while ORCA is the stress SNQI and low-collision leader.
- Treat `prediction_planner` as deprioritized for this lane unless runtime or collision behavior is
  the specific research target.
- Do not use SNQI alone to pick a winner. Use it to flag when success/collision ranks need a
  mechanism-level explanation, especially under stress.

## Evidence Gaps

The comparison is diagnostic-only for four reasons:

- The #1353 bundle provides SNQI components and availability fields, but not a full
  failure-mechanism taxonomy assignment per episode.
- SNQI contract status is `warn` on nominal and `fail` on stress, so SNQI should remain an
  operational aggregation diagnostic rather than paper-facing utility.
- Several rows run through adapter modes with projection/infeasible rates; mechanism interpretation
  must keep native-vs-adapter boundaries visible.
- The stress and nominal surfaces have different scenario breadth and episode counts, so this is a
  paired interpretation surface, not a controlled causal ablation.

Next smallest proof step:

1. For a future mechanism-aware ranking artifact, build a single paired table that includes
   nominal outcomes, SNQI, row status, execution mode, projection/infeasible rate, and failure
   mechanism labels from the current taxonomy.
2. Use the stress surface to select a focused trace review for PPO versus ORCA before adding any
   new universal ranking score.
3. Keep `socnav_bench` as `not_available` until the missing SocNavBench assets are hydrated; do not
   count it as a failed or successful planner.

## Validation

This docs-only synthesis was checked with:

```bash
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check origin/main...HEAD
```
