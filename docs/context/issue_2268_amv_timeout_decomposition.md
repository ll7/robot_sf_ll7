# Issue #2268 AMV Timeout Decomposition 2026-06-05

Issue: [#2268](https://github.com/ll7/robot_sf_ll7/issues/2268)
Parent issue: [#2259](https://github.com/ll7/robot_sf_ll7/issues/2259)
Related evidence: [#2224](https://github.com/ll7/robot_sf_ll7/issues/2224),
[#2249](https://github.com/ll7/robot_sf_ll7/pull/2249)
Status: diagnostic-only analysis; clipping improved, but the observed timeout driver remained low
route progress.

## Goal

Explain why `actuation_aware_hybrid_rule_v0` reduced synthetic AMV command clipping in the matched
Issue #2224 / PR #2249 smoke but did not improve task success.

This note analyzes the durable compact evidence already tracked for the one-episode
`amv_actuation_smoke` slice:

- Scenario: `classic_cross_trap_high`
- Baseline candidate: `hybrid_rule_v3_fast_progress`
- Actuation-aware candidate: `actuation_aware_hybrid_rule_v0`
- Evidence tier: `analysis_only`

It is not planner-ranking, benchmark-strength, calibrated AMV, hardware, safety, or paper-facing
evidence.

## Observed Comparison

| Metric | `hybrid_rule_v3_fast_progress` | `actuation_aware_hybrid_rule_v0` | Delta |
| --- | ---: | ---: | ---: |
| Success rate | 0.0000 | 0.0000 | 0.0000 |
| Collision rate | 0.0000 | 0.0000 | 0.0000 |
| Near-miss rate | 0.0000 | 0.0000 | 0.0000 |
| Mean command clip fraction | 0.2750 | 0.1875 | -0.0875 |
| Mean yaw-rate saturation fraction | 0.0000 | 0.0000 | 0.0000 |
| Mean signed braking peak | -2.5000 | -2.5000 | 0.0000 |
| Mean average speed | 1.6623 | 1.6670 | +0.0047 |
| Mean minimum distance | 2.1571 | 2.3627 | +0.2056 |
| Failure mode | `timeout_low_progress: 1` | `timeout_low_progress: 1` | unchanged |

## Classification

The durable evidence supports this mechanism boundary:

- **Command feasibility improved**: mean command clipping fell from `0.2750` to `0.1875`.
- **Speed-cap or yaw-saturation explanation is weak for this slice**: mean average speed changed by
  only `+0.0047`, yaw saturation stayed `0.0000`, and signed braking peak stayed `-2.5000`.
- **Task success stayed blocked by route progress**: both candidates ended with
  `timeout_low_progress: 1` and `0.0000` success.
- **Navigation success was orthogonal to the clipping improvement in this smoke**: the actuation
  scorer made commands less clipped without changing the termination class.

The best supported timeout classification for this durable slice is therefore
`route_progress_blocker_with_feasibility_improvement`.

## Missing Evidence

The tracked #2224/#2249 evidence does not preserve per-step route-progress traces, command
saturation over time, or deadlock/oscillation features. That prevents a finer split between:

- route planner geometry,
- command feasibility after clipping,
- conservative actuation scoring,
- local deadlock,
- and scenario-level progress bottlenecks.

Do not rerun broad AMV actuation variants from this result alone. The next useful proof is a
trace-level extractor or rerun that records route progress and command-feasibility time series for
the same matched smoke.

## Recommendation

Keep `actuation_aware_hybrid_rule_v0` as diagnostic-only feasibility instrumentation. It should not
be promoted as a navigation-success improvement from the #2224/#2249 smoke.

The next targeted follow-up should add or preserve per-step route-progress and command-feasibility
signals, then rerun the same matched smoke or a tiny predeclared matched slice. A broader AMV
actuation benchmark should wait until that trace-level blocker is understood.

## Evidence

- Prior ranking note:
  [issue_2224_amv_actuation_ranking.md](issue_2224_amv_actuation_ranking.md)
- Prior compact comparison:
  [evidence/issue_2224_amv_actuation_ranking_2026-06-04/comparison.json](evidence/issue_2224_amv_actuation_ranking_2026-06-04/comparison.json)
- Prior manifest:
  [evidence/issue_2224_amv_actuation_ranking_2026-06-04/manifest.json](evidence/issue_2224_amv_actuation_ranking_2026-06-04/manifest.json)
- This compact summary:
  [evidence/issue_2268_amv_timeout_decomposition_2026-06-05/summary.json](evidence/issue_2268_amv_timeout_decomposition_2026-06-05/summary.json)
- This timeout table:
  [evidence/issue_2268_amv_timeout_decomposition_2026-06-05/timeout_decomposition.csv](evidence/issue_2268_amv_timeout_decomposition_2026-06-05/timeout_decomposition.csv)
- This manifest:
  [evidence/issue_2268_amv_timeout_decomposition_2026-06-05/manifest.md](evidence/issue_2268_amv_timeout_decomposition_2026-06-05/manifest.md)

## Validation

This analysis used tracked compact evidence only. No new benchmark or broad rerun was performed.

Validation commands:

```bash
python -m json.tool docs/context/evidence/issue_2268_amv_timeout_decomposition_2026-06-05/summary.json
python - <<'PY'
import csv
from pathlib import Path
rows = list(csv.DictReader(Path("docs/context/evidence/issue_2268_amv_timeout_decomposition_2026-06-05/timeout_decomposition.csv").open()))
assert {row["candidate"] for row in rows} == {"hybrid_rule_v3_fast_progress", "actuation_aware_hybrid_rule_v0"}
assert all(row["failure_mode"] == "timeout_low_progress" for row in rows)
PY
bash scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check
```
