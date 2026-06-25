# Issue #2445 ORCA-Residual Progress-Probe Decision

Date: 2026-06-23
Issue: <https://github.com/ll7/robot_sf_ll7/issues/2445>
Parents: <https://github.com/ll7/robot_sf_ll7/issues/1475>,
<https://github.com/ll7/robot_sf_ll7/issues/1358>
Prior decision: [issue_2408_orca_residual_low_progress_analysis.md](issue_2408_orca_residual_low_progress_analysis.md)
(stop rule), PR #2420.

## Scope

This note classifies the ORCA-residual progress-probe target **after** the revised v1 bounded smoke
result (SLURM job 12913) and decides whether the ORCA-residual BC lane should **continue, revise, or
stop**. It is a decision over *existing* evidence: there is no training to run here. It does not
launch training, submit SLURM jobs, promote checkpoints, or claim learned-residual success.

The compact machine-readable decision artifact lives under
`docs/context/evidence/issue_2445_orca_residual_progress_probe_decision_2026-06-23/`.

## Evidence Source

- `docs/context/evidence/issue_1475_orca_residual_bc_smoke_12913_2026-06-17/summary.json` — the
  revised v1 bounded smoke result (SLURM job 12913, candidate
  `orca_residual_guarded_ppo_progress_v1`, source commit
  `dcb14927f08277d123d9666ad91b8d6abbc4fe9d`).

## The #2408 Stop Rule (input decision rule)

From [issue_2408_orca_residual_low_progress_analysis.md](issue_2408_orca_residual_low_progress_analysis.md)
and PR #2420:

> If the v1 bounded smoke also produces `success_rate=0.0` with `timeout_low_progress`, **stop** the
> current residual-BC lane shape or reopen only through a **named objective/dataset redesign**. Do
> not rerun unchanged v0 BC and do not submit `nominal_sanity` from the failed v0 row.

## Observed v1 Smoke (job 12913)

| Field | Value |
| --- | ---: |
| Episodes | 1 |
| Success rate | `0.0` |
| Failure mode | `timeout_low_progress` (count 1) |
| Termination reason | `max_steps` |
| Mean average speed | 0.8041 |
| Status | `failed_closed` |
| Classification | `missing_required_smoke_evidence` |
| `nominal_escalation_allowed` | `false` |
| `residual_clipping_rate` | `null` (missing) |
| `guard_veto_rate` | `null` (missing) |
| `fallback_degraded_status` | `null` (missing) |
| `artifact_pointer_status` | `null` (missing) |

Two independent triggers fire on this row:

1. **Reproduced v0 failure pattern**: `success_rate=0.0` **and** `timeout_low_progress`, exactly the
   pattern the #2408 stop rule names.
2. **Missing required smoke-evidence fields**: all four required fields
   (`residual_clipping_rate`, `guard_veto_rate`, `fallback_degraded_status`, `artifact_pointer_status`)
   are null, so the smoke is itself `failed_closed` / `missing_required_smoke_evidence`. This
   independently forbids nominal escalation (`nominal_escalation_allowed=false`).

## Decision

Selected decision output: **`stop`** the current ORCA-residual BC lane shape.

The deterministic classifier
(`scripts/analysis/classify_orca_residual_progress_probe_issue_2445.py`) reports the
`orca_residual_progress_probe_decision` block with `decision: stop`, grounded in the two triggers
above. The fail-closed branch reports the missing-required-fields trigger first; even if the fields
were present, the reproduced-v0-failure branch (`success_rate=0.0` + `timeout_low_progress`) would
still return `stop`.

Recommendation for Issue #1475 and Issue #1358 (handoff note — this lane does not edit GitHub):

- **Stop** the current residual-BC lane shape. Do not rerun the unchanged v0/v1 BC smoke and do not
  submit `nominal_sanity` from this failed-closed row.
- **Reopen only through a named objective/dataset redesign**: a new BC objective, dataset, scenario,
  or instrumentation lane that (a) emits and tests the four required smoke-evidence fields and
  (b) changes the residual objective/data so the progress probe is not a relabeled unchanged rerun.
- Update #1358 / #1475 with this decision (handoff only; do not edit GitHub from this lane).

## What a Valid Reopen Requires

A reopen is valid only if it is a *named redesign*, not a relabeled rerun:

1. The smoke runner/finalizer emits the four required fields (`residual_clipping_rate`,
   `guard_veto_rate`, `fallback_degraded_status`, `artifact_pointer_status`) and they are tested.
2. The BC objective, dataset, or scenario is materially changed (e.g. the larger BC dataset pending
   as a wandb artifact, a revised progress objective, or a different scenario split) — not the same
   v0/v1 packet rerun.
3. A single bounded smoke is rerun under the redesigned lane *before* any nominal escalation.

## Claim Boundary

This is analysis-only routing evidence over a **failed-closed** smoke. It explains why the current
residual-BC lane shape stops and what a valid reopen requires. It is **not** readiness evidence, not
a new benchmark result, not a learned-component success claim, and must **not** be used to justify
nominal or larger SLURM reruns.

## Validation

```bash
uv run python scripts/analysis/classify_orca_residual_progress_probe_issue_2445.py --help
uv run python scripts/analysis/classify_orca_residual_progress_probe_issue_2445.py   # real run on 12913 => stop
uv run python -m pytest tests/ -k "orca_residual_progress_probe or 2445" -q
```
