<!-- AI-GENERATED (robot_sf#5034, 2026-07-15) - NEEDS-REVIEW -->
# Issue #5034 → parent #4977 result classification (2026-07-15)

Plain-language summary: this record closes the last remaining Definition-of-Done item
of issue #5034 — *"the parent issue #4977 is updated with the result classification."*
It classifies the control-action-latency fidelity sweep, as executed so far, as
**`diagnostic-only`** evidence per the issue #5034 evidence tier set and
`docs/benchmark_governance.md`. It does **not** claim `nominal benchmark evidence`;
the authorized native 7,344-episode fixed-scope campaign (parent #4977's stated
nominal tier) has not run. No campaign, Slurm, or GPU job was launched for this
classification step.

## Classification

- `evidence_tier`: `targeted smoke`
- `result_classification`: `diagnostic-only`
- `parent_issue`: `4977`
- `status`: `evidence`
- Claim boundary: control-action-latency metric-evidence promotion only; reads raw
  fidelity-campaign episode rows, isolates the `control_action_latency` axis, and
  reports action-latency metadata plus success / collision / minimum-clearance
  metrics per native latency cell. It runs no episode and promotes no claim beyond
  the declared campaign evidence tier; it is not simulator-realism evidence, not
  sim-to-real evidence, and not paper-facing evidence.

## Why diagnostic-only (and not nominal benchmark evidence)

Issue #5034's own Definition of Done lists three valid result classifications:
*nominal benchmark evidence*, *diagnostic-only*, or *blocked / not benchmark
evidence*. The `nominal` tier requires the authorized native 7,344-episode
fixed-scope campaign over the native ORCA / hybrid planner set. That campaign is
**out of the cheap-lane scope** (it needs Slurm-class capacity) and has **not**
been executed. The `diagnostic-only` classification is therefore the honest,
fail-closed classification of the work completed to date, and it is explicitly
permitted by the issue contract.

## What has produced the diagnostic-only evidence

- PR #5026: configurable control-to-actuation latency queue in `RobotEnv.step()`
  (default 0, backward compatible) plus the `control_action_latency` campaign
  axis (0/1/3 steps = 0/100/300 ms-equivalent) in
  `configs/research/fidelity_sensitivity_v1.yaml`.
- PR #5061: fail-closed latency-sweep preflight guard.
- PR #5085: control-action-latency evidence promoter (`robot_sf/benchmark/
  control_action_latency_evidence.py`), classifying non-native / fallback /
  degraded rows as exclusions per issue #691.
- PR #5536: fixed-scope scenario resolution and launch-plan provenance.
- PR #5620: coverage / reconciliation gate (promoter v2 + `planner_group`
  provenance).
- PR #5629: durable fixed-scope launch plan +
  `docs/context/evidence/issue_5034_control_action_latency_sweep_plan_2026-07-14/`.
- PR #5648: cheap-lane CPU control-action-latency sweep — **36 native rows**
  (`baseline_social_force`, `goal_seek`; seeds `101/102/103`; latency steps
  `0/1/3`; 2 scenarios), classified `diagnostic-only` and promoted under
  `docs/context/evidence/issue_5034_control_action_latency_sweep/`.

## Diagnostic-only result observed (from #5648, 36 native rows)

| Planner | Latency steps | Latency ms | Cells | Success | Collision | Min clearance |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_social_force` | 0 | 0.0 | 6 | 0.0 | 0.0 | 3.707 |
| `baseline_social_force` | 1 | 100.0 | 6 | 0.0 | 0.0 | 3.786 |
| `baseline_social_force` | 3 | 300.0 | 6 | 0.0 | 0.0 | 4.084 |
| `goal_seek` | 0 | 0.0 | 6 | 0.0 | 1.0 | -0.045 |
| `goal_seek` | 1 | 100.0 | 6 | 0.0 | 1.0 | -0.040 |
| `goal_seek` | 3 | 300.0 | 6 | 0.0 | 1.0 | -0.049 |

Excluded rows: `0`. These are compact CPU-smoke cells, not the full fixed-scope
campaign; treat the numbers as exploratory signal, not a benchmark conclusion.

## Remaining work before parent #4977 can reach the nominal tier

- Execute the authorized native 7,344-episode fixed-scope campaign
  (ORCA / hybrid planner groups; run
  `scripts/benchmark/run_fidelity_sensitivity_campaign.py --fixed-scope-execute`),
  requiring Slurm-class capacity.
- Promote the real rows with the strict fixed-scope contract
  (`validate_fixed_scope_latency_coverage`) and promote a durable raw-JSONL
  external pointer.
- Re-classify parent #4977 from `diagnostic-only` to `nominal benchmark
  evidence` only after that campaign's rows pass the fixed-scope gate.

Until then, parent #4977's control-action-latency result stands at
**`diagnostic-only`**, recorded here and in the #5648 `summary.json`
(`parent_issue: 4977`, `result_classification: diagnostic-only`).

## Source artifacts

- `docs/context/evidence/issue_5034_control_action_latency_sweep/summary.json`
- `docs/context/evidence/issue_5034_control_action_latency_sweep/README.md`
- `docs/context/evidence/issue_5034_control_action_latency_sweep_plan_2026-07-14/`
