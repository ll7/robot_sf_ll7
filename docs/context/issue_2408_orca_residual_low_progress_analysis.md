# Issue #2408 ORCA-Residual Low-Progress Analysis

Date: 2026-06-06
Issue: <https://github.com/ll7/robot_sf_ll7/issues/2408>
Parents: <https://github.com/ll7/robot_sf_ll7/issues/1475>,
<https://github.com/ll7/robot_sf_ll7/issues/1358>
Related decisions: [issue_2311_orca_residual_lane_decision.md](issue_2311_orca_residual_lane_decision.md),
[issue_2410_hybrid_component_readiness_refresh.md](issue_2410_hybrid_component_readiness_refresh.md)

## Scope

This note classifies the existing ORCA-residual BC v0 low-progress smoke failure before another
rerun. It does not launch training, submit SLURM jobs, promote checkpoints, or claim
learned-residual success.

The compact machine-readable artifacts live under
`docs/context/evidence/issue_2408_orca_residual_low_progress_2026-06-06/`.

## Evidence Sources

- `docs/context/evidence/issue_1475_orca_residual_bc_smoke_12749_summary.json` for the tracked
  job 12749 smoke summary.
- `docs/context/policy_search/reports/2026-06-05_orca_residual_guarded_ppo_v0_smoke.md` for the
  v0 smoke report and baseline deltas.
- `docs/context/evidence/issue_1967_orca_residual_bc_smoke_adapter_summary.json` for the
  adapter-repair boundary and earlier repaired local smoke.
- `docs/context/issue_2272_orca_residual_launch_packet_status.md` and
  `docs/context/issue_2311_orca_residual_lane_decision.md` for the previous fail-closed decision.
- `configs/training/orca_residual/orca_residual_bc_issue_1428.yaml` and
  `configs/policy_search/candidates/orca_residual_guarded_ppo_progress_v1.yaml` for the current
  progress-probe launch packet.

## Observed Failure

The tracked job 12749 smoke row reached execution after the adapter and missing-JSONL blockers were
repaired. It produced one usable row with:

| Metric | Value |
| --- | ---: |
| Episodes | 1 |
| Success rate | 0.0000 |
| Collision rate | 0.0000 |
| Near-miss rate | 0.0000 |
| Failure mode | `timeout_low_progress` |
| Termination reason | `max_steps` |
| Mean average speed | 0.8038 |
| Shield decisions | 80 |
| Shield override rate | 0.0000 |
| Shield hard-constraint violation rate | 0.0000 |

The row is diagnostic-only. It is not benchmark-strength evidence and it does not show that
ORCA-residual methods are generally ineffective.

## Question Classification

| Question | Classification | Confidence | Evidence | Decision impact |
| --- | --- | ---: | --- | --- |
| Was the residual policy inactive? | unlikely / unresolved | 0.45 | The matrix records residual active rate 1.0 and no guard veto/fallback, but tracked artifacts do not include per-step raw residual magnitude. | Add residual magnitude and clipping instrumentation before interpreting a new row. |
| Was the BC dataset too small? | plausible contributing factor | 0.65 | The smoke packet used three seed episodes for BC; the discriminating failure appears after the BC-trained candidate path, not during adapter repair. | Do not spend another unchanged BC rerun without objective/dataset redesign evidence. |
| Was the smoke target inappropriate? | rejected | 0.90 | `planner_sanity_simple` is the first gate; the v0 row underperformed goal, ORCA, and PPO baselines and did not earn nominal escalation. | Keep the smoke gate; do not weaken the target as the primary fix. |
| Was the residual bounded too tightly? | unresolved secondary hypothesis | 0.35 | v0 used 0.25 linear/angular residual bounds; v1 raises them to 0.35, but tracked v0 evidence lacks residual clipping rate. | Treat v1 as a progress-probe, not proof that bound width caused v0 failure. |
| Did guard/fallback dominate? | not observed | 0.95 | Shield override, hard-constraint violation, guard veto, fallback, and degraded rates are all recorded as 0.0 in the compact evidence. | The hard guard is not the current bottleneck. |
| Was the learned component unable to produce route progress? | primary observed failure | 0.85 | The row timed out with low progress after 80 guarded steps, no collision, no near miss, and no guard intervention. | Revise the objective/diagnostics before another smoke; no nominal escalation. |

## Decision

Selected decision output: `rerun_with_instrumentation` on top of the already selected
`revise_residual_objective` path.

Recommendation for Issue 1475 and Issue 1358:

- Continue the ORCA-residual lane only through the revised progress-probe packet
  `orca_residual_guarded_ppo_progress_v1`.
- Do not rerun unchanged v0 BC smoke and do not submit `nominal_sanity` from the failed v0 row.
- Before or during the next bounded smoke, capture residual clipping rate, mean/max raw residual
  magnitude, bounded residual magnitude, guard veto/override rates, and fallback/degraded status in
  the durable compact summary.
- If the v1 bounded smoke also produces `success_rate=0.0` with `timeout_low_progress`, stop the
  current residual-BC lane shape or reopen only through a named objective/dataset redesign.

Current confidence: `0.82`.

## Claim Boundary

This is analysis-only routing evidence. It explains the v0 low-progress failure boundary and the
next evidence needed. It is not a new benchmark result, not a learned-component success claim, and
not a paper-facing hybrid-learning conclusion.

## Validation

```bash
python -m json.tool docs/context/evidence/issue_2408_orca_residual_low_progress_2026-06-06/summary.json
python - <<'PY'
import csv
from pathlib import Path
rows = list(csv.DictReader(Path(
    "docs/context/evidence/issue_2408_orca_residual_low_progress_2026-06-06/question_classification.csv"
).open(newline="")))
assert len(rows) == 6
assert {row["question_id"] for row in rows} == {
    "residual_inactive",
    "dataset_too_small",
    "smoke_target_inappropriate",
    "residual_bound_too_tight",
    "guard_or_fallback_dominated",
    "learned_component_no_route_progress",
}
PY
uv run python scripts/validation/check_docs_proof_consistency.py \
  --path docs/context/issue_2408_orca_residual_low_progress_analysis.md \
  --path docs/context/evidence/issue_2408_orca_residual_low_progress_2026-06-06/summary.json \
  --path docs/context/catalog.yaml
uv run python scripts/validation/validate_orca_residual_lineage_packet.py \
  --config configs/training/orca_residual/orca_residual_bc_issue_1428.yaml --json
git diff --check
```
