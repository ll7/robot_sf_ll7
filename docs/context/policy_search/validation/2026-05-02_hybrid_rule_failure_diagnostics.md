# Hybrid Rule Failure Diagnostics

Date: 2026-05-02

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/874>

## Goal

Diagnose the remaining timeout and static-route failure modes for the current
`hybrid_rule_v3_fast_progress_static_escape` policy-search candidate before adding more planner
mechanisms or tuning constants.

This note is diagnostic only. It does not promote a planner, exclude additional scenarios, or
change the benchmark contract.

## Evidence Sources

- Candidate report:
  `docs/context/policy_search/reports/2026-05-01_hybrid_rule_v3_fast_progress_static_escape_full_matrix.md`
- Failure report:
  `docs/context/policy_search/validation/fast_progress_static_recenter07_scenario_override_full_failure_report/failure_report.md`
- Prior audit:
  `docs/context/policy_search/validation/2026-05-01_seeded_horizon_policy_audit.md`
- Wrap-up note:
  `docs/context/policy_search/validation/2026-05-01_policy_search_wrapup.md`
- Fresh local step diagnostics under
  `output/ai/autoresearch/issue_874_diagnostics/`

The `output/` traces are worktree-local proof artifacts, not durable dependencies. The commands in
the validation section reproduce them.

No video artifacts were present in this checkout for the remaining failures, so this pass used the
tracked reports plus fresh step-level traces.

## Aggregate Context

The current full-matrix run for `hybrid_rule_v3_fast_progress_static_escape` reports:

- episodes: `141`
- success: `130/141` (`0.9220`)
- collision rate: `0.0213`
- near-miss rate: `0.4113`
- failure taxonomy: `3` static collisions, `6` timeout-low-progress failures, `2` intrusive
  near-miss failures

Five raw failures already have strong invalid/impossible evidence in the prior audit:

- `classic_cross_trap_high` seed `112`
- `francis2023_circular_crossing` seed `111`
- `francis2023_narrow_doorway` seeds `111`, `112`, `113`

This diagnostic pass did not reclassify those exclusions. It focuses on the six failures that are
not proven impossible:

- `classic_merging_low` seeds `111`, `113`
- `classic_merging_medium` seeds `111`, `112`, `113`
- `francis2023_leave_group` seed `113`

## Fresh Trace Outcomes

| Scenario | Seed | Outcome | Trace Summary |
|---|---:|---|---|
| `classic_merging_low` | 111 | obstacle collision at episode step `242` | no pedestrian collision, no near misses, final reward dominated by collision penalty |
| `classic_merging_low` | 113 | horizon reached without success/collision | final command `[0.0, 0.0]`, max zero-command streak `310` steps |
| `classic_merging_medium` | 111 | horizon reached without success/collision | final command `[0.0, 0.0]`, max zero-command streak `226` steps |
| `classic_merging_medium` | 112 | horizon reached without success/collision | final command `[0.0, 0.0]`, max zero-command streak `187` steps |
| `classic_merging_medium` | 113 | horizon reached without success/collision | final command `[0.0, 0.0]`, max zero-command streak `166` steps |
| `francis2023_leave_group` | 113 | max-steps termination at episode step `400` | final command `[0.0, 0.0]`, distance to goal `2.5317`, comfort exposure `0.3333` |

Observed evidence:

- The classic merging failures are not first-step invalid initializations and do not have a
  geometry proof comparable to `francis2023_narrow_doorway`.
- The timeout traces show long zero-command tails after earlier progress, which matches the prior
  static-clearance local-minimum diagnosis.
- `classic_merging_low` seed `111` is the unsafe variant of the same family: the candidate keeps
  making progress and eventually clips an obstacle rather than freezing.
- `francis2023_leave_group` seed `113` reaches the final group interaction without static or
  pedestrian collision, then freezes near the goal until the scenario time limit.

## Failure Classes

### Classic Merging Static-Corridor Deadlock

Affected seeds:

- `classic_merging_low` seed `113`
- `classic_merging_medium` seeds `111`, `112`, `113`

Observed evidence:

- Fresh traces end with repeated zero commands and no successful route completion.
- Prior diagnostics report that the route leads the robot into a narrow static-clearance band where
  many moving candidates are rejected by `static_clearance`.
- Lower speed/turn caps did not solve the pattern; several settings either timed out or introduced
  many near misses.
- Raising `static_recenter_weight` on `classic_merging_medium` seed `111` converted timeouts into
  obstacle collisions, so stronger recenter pressure is not a safe standalone fix.

Hypothesis:

- The retained candidate lacks a corridor-aware recovery mode that can choose a safe lateral or
  route-realigning motion when forward candidates are either rejected by static clearance or scored
  below stopping.
- This is planner work, not a scenario exclusion, because no current evidence proves the route is
  geometrically impossible for the robot.

### Classic Merging Static-Corridor Collision

Affected seed:

- `classic_merging_low` seed `111`

Observed evidence:

- Fresh trace reproduces obstacle collision at episode step `242`.
- The terminal metadata reports no pedestrian collision and no near misses.
- Prior scenario-scoped `static_hard_safety_margin: 0.0` avoided one obstacle collision but still
  failed tested merging seeds with all candidates rejected by static clearance.

Hypothesis:

- The collision is the unsafe side of the same corridor-recovery problem: the planner sometimes
  accepts progress through a narrow static band instead of stopping, but the available motion is
  not guaranteed to preserve the hard static-collision gate.

### Francis Leave-Group Dynamic Hard-Radius Deadlock

Affected seed:

- `francis2023_leave_group` seed `113`

Observed evidence:

- Fresh trace terminates at max steps with final zero command and no static or pedestrian
  collision.
- Prior targeted diagnostics show tuned ORCA succeeds on all three `francis2023_leave_group`
  seeds, including seed `113`, with no collisions.
- The candidate report states the retained hybrid candidate freezes because all candidates are
  rejected by dynamic collision when a pedestrian remains just inside the hard dynamic radius.

Hypothesis:

- This is not a static-route failure. It is a dynamic-agent interaction where the local hard-radius
  filter leaves no acceptable hybrid-rule command near the group.
- A scenario-adaptive algorithm switch to ORCA for `francis2023_leave_group` is currently the
  narrowest evidence-backed mechanism, but it still needs a full-matrix rerun before promotion.

## Rejected Knobs That Should Stay Default-Off

The recent search already tested several constant-only or broad relaxation paths. They should remain
default-off unless new evidence changes the safety tradeoff:

- global or scenario-scoped `static_hard_safety_margin: 0.0`
- higher `static_recenter_weight` for merging
- lower speed/turn caps for merging
- bounded static-clearance entry through the hard band
- longer route-guide lookahead
- broad Francis-family speed overrides

These either failed to recover the target seeds, introduced obstacle collisions, or regressed other
scenario families.

## Recommended Next Mechanisms

1. For classic merging, investigate a static-corridor recovery mechanism that preserves hard
   static-collision safety. The validation slice should include all five merging seeds listed above
   plus the nominal/stress gates used by the retained candidate.
2. For `francis2023_leave_group`, rerun the existing `scenario_adaptive_hybrid_orca_v1` candidate
   on the full matrix before changing the hybrid-rule planner. Targeted evidence already supports
   ORCA for that scenario, but aggregate proof is still missing.

Do not classify the six non-excluded failures as impossible from the current evidence.

## Validation Commands

Fresh diagnostics:

```bash
LOGURU_LEVEL=WARNING uv run python scripts/validation/run_policy_search_step_diagnostics.py \
  --candidate hybrid_rule_v3_fast_progress_static_escape \
  --stage full_matrix \
  --scenario-name classic_merging_low \
  --seed 111 \
  --horizon 500 \
  --output-dir output/ai/autoresearch/issue_874_diagnostics/classic_merging_low_111_h500

LOGURU_LEVEL=WARNING uv run python scripts/validation/run_policy_search_step_diagnostics.py \
  --candidate hybrid_rule_v3_fast_progress_static_escape \
  --stage full_matrix \
  --scenario-name classic_merging_low \
  --seed 113 \
  --horizon 500 \
  --output-dir output/ai/autoresearch/issue_874_diagnostics/classic_merging_low_113_h500

LOGURU_LEVEL=WARNING uv run python scripts/validation/run_policy_search_step_diagnostics.py \
  --candidate hybrid_rule_v3_fast_progress_static_escape \
  --stage full_matrix \
  --scenario-name classic_merging_medium \
  --seed 111 \
  --horizon 500 \
  --output-dir output/ai/autoresearch/issue_874_diagnostics/classic_merging_medium_111_h500

LOGURU_LEVEL=WARNING uv run python scripts/validation/run_policy_search_step_diagnostics.py \
  --candidate hybrid_rule_v3_fast_progress_static_escape \
  --stage full_matrix \
  --scenario-name classic_merging_medium \
  --seed 112 \
  --horizon 500 \
  --output-dir output/ai/autoresearch/issue_874_diagnostics/classic_merging_medium_112_h500

LOGURU_LEVEL=WARNING uv run python scripts/validation/run_policy_search_step_diagnostics.py \
  --candidate hybrid_rule_v3_fast_progress_static_escape \
  --stage full_matrix \
  --scenario-name classic_merging_medium \
  --seed 113 \
  --horizon 500 \
  --output-dir output/ai/autoresearch/issue_874_diagnostics/classic_merging_medium_113_h500

LOGURU_LEVEL=WARNING uv run python scripts/validation/run_policy_search_step_diagnostics.py \
  --candidate hybrid_rule_v3_fast_progress_static_escape \
  --stage full_matrix \
  --scenario-name francis2023_leave_group \
  --seed 113 \
  --horizon 500 \
  --output-dir output/ai/autoresearch/issue_874_diagnostics/francis2023_leave_group_113_h500
```

Reference report checks:

```bash
uv run python scripts/tools/build_policy_search_failure_report.py --help
test -f docs/context/policy_search/validation/fast_progress_static_recenter07_scenario_override_full_failure_report/failure_report.md
```
