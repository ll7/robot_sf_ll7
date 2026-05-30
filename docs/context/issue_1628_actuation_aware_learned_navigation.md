# Issue #1628 Actuation-Aware Learned Navigation For AMVs

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1628>

Date: 2026-05-30

## Goal

Assess whether learned local-navigation policies are a plausible Robot SF research path
specifically because autonomous micromobility vehicle actuation constraints matter, rather than
because another generic learned policy might improve average benchmark score.

This note is analysis-only. It does not train a policy, add a benchmark row, change vehicle
dynamics, promote a checkpoint, or claim learned-policy superiority over classical planners.

## Current Actuation Baseline

| Surface | Current state | Learned-policy implication |
|---|---|---|
| Issue #1546 design note | Defines a synthetic single-AMV, differential-drive actuation-envelope stress slice with acceleration, braking, yaw-rate, angular-acceleration, latency, and update-rate concepts. | Supplies the stress variables a learned policy should observe or be evaluated against, but not a calibrated hardware envelope. |
| Issue #1556 implementation note | Adds the checked-in synthetic actuation slice, `paper_facing: false`, scenario AMV overrides, profile provenance, and actuation diagnostics. | Provides the first executable synthetic envelope and diagnostic metrics for future learned-policy smoke work. |
| Issue #1569 compact smoke evidence | Shows the synthetic slice runs locally with three successful evidence rows, no fallback/degraded rows, but `success_mean=0.0000` for `goal`, `orca`, and `social_force`. | Establishes that the stress surface is hard and diagnostic, not that any learned policy is better. |
| Issue #1572 metadata cleanup | Closes the scenario-AMV and adapter command-space/projection metadata gaps from the smoke review. | Makes future actuation summaries safer to interpret without rereading raw campaign rows. |
| Issue #1559 calibrated-envelope follow-up | Remains blocked on a durable AMV calibration source before paper-facing use. | Blocks hardware-calibrated or real-world transfer language for learned policies. |
| Issue #1618 learned-policy adapter contract | Defines observation/action/checkpoint metadata, deterministic inference, diagnostics, and fail-closed status semantics. | Any learned AMV policy must pass this boundary before smoke or benchmark evidence counts. |
| Issue #1675 learned risk surface | Proves a deterministic local risk/potential surface can be produced and consumed without a learned checkpoint. | Useful as an interface fixture for risk-aware AMV policies, not performance evidence. |

## Research Hypothesis

The strongest AMV-specific learned-policy hypothesis is:

> A learned local policy may be useful when it can anticipate or reduce actuation-envelope stress,
> such as command clipping, braking spikes, yaw-rate saturation, stop-go jerk, or low-clearance
> recovery, while preserving fail-closed safety and explicit adapter diagnostics.

This is narrower than "learned policies beat classical planners." The hypothesis is about whether
learning can improve behavior under constrained execution where hand-designed planners may ask for
commands that an AMV envelope clips, delays, or smooths away.

## Candidate Formulations

| Formulation | What it learns | Required Robot SF contract | Why it is plausible | Main risk |
|---|---|---|---|---|
| Actuation-penalized PPO | Direct velocity/yaw-rate commands with reward terms for clipping, yaw saturation, signed braking peaks, jerk, and progress. | Issue #1618 metadata, deterministic inference, no hidden fallback, profile provenance in smoke output. | Reuses existing PPO adapter concepts and the synthetic actuation metrics already emitted by Issue #1556. | Reward shaping may reduce clipping while also reducing progress or creating brittle behavior. |
| ORCA residual under AMV envelope | A bounded residual over an ORCA or classical command, with hard guards and actuation diagnostics. | Residual action bounds, raw ORCA command, residual command, post-guard command, fallback/degraded status. | Keeps classical collision-avoidance structure while learning where envelope-limited behavior needs correction. | Residual may simply mask ORCA projection artifacts or depend on unavailable durable training traces. |
| Learned risk / potential surface | A local risk field or cost supplement consumed by an existing local planner. | Risk-surface provenance, ego-frame conventions, deterministic producer, unavailable-on-malformed-surface behavior. | Builds on Issue #1675 and avoids directly replacing the controller. | Risk may not encode actuation limits unless trained or parameterized with command-history context. |
| Planner arbitration | Selects among classical, predictive, guarded, or learned modes using only inference-available features. | Switch-rate diagnostics, no hindsight labels at inference, per-step selected planner and guard/fallback reason. | May be safer than an end-to-end replacement policy if the selector learns when actuation stress makes one planner family unsuitable. | Hindsight leakage and unsafe oscillation are easy to introduce without a strict trace contract. |
| Offline or sequence policy | Uses durable traces to learn delayed or history-aware command choices under envelope stress. | Durable dataset manifest, observation/action/return schema, held-out split, and non-claim training boundary. | Could represent latency/history effects better than a memoryless PPO policy. | Currently blocked until durable trace provenance and schema readiness are proven. |

## Scenario Hypotheses

Use the existing Issue #1546/#1556 scenario set first because it already has synthetic actuation
profile provenance and compact smoke evidence.

| Scenario / family | Actuation question | Expected differentiator |
|---|---|---|
| `classic_cross_trap_high` | Can the policy avoid yaw-rate or curvature deadlock without oscillating? | yaw-rate saturation, command clipping bursts, time-to-goal, stall/failure-to-progress. |
| `classic_bottleneck_high` | Can the policy handle stop-go restart pressure smoothly? | signed braking peaks, jerk, near misses, low-clearance recovery. |
| `classic_overtaking_medium` | Can the policy plan longitudinal speed changes without repeated clipping? | acceleration/braking demand, route progress, collision/near-miss tradeoff. |
| `francis2023_blind_corner` | Does actuation context help avoid late turn-in under occlusion-like geometry? | clearance, time-to-collision, yaw-rate demand. |
| `francis2023_intersection_wait` | Can the policy relaunch after yielding without jerk or saturation spikes? | jerk, stalled time, progress, signed braking peak. |

## Required Inputs

Minimum observation fields for an actuation-aware learned policy:

- ego pose, heading, current linear/angular velocity, and goal vector;
- local pedestrian/obstacle context through structured, LiDAR, or occupancy-derived observations;
- configured synthetic actuation profile ID and bounds when the policy is explicitly profile-aware;
- previous command, previous clipped/adapted command, or a bounded command-history feature if the
  formulation claims to reduce saturation or jerk;
- no future collision labels, outcome labels, or post-hoc planner success fields at inference time.

Minimum action contract:

- bounded continuous command compatible with the active kinematics, preferably `unicycle_vw` for the
  first differential-drive AMV slice;
- explicit raw model action, adapted command, post-guard command, projection/clipping policy, and
  fallback reason in diagnostics;
- fail-closed `not_available` or `failed` status when checkpoint, observation, kinematics, or
  profile metadata is missing.

Minimum metrics:

- success, collision, near miss, time-to-goal, progress, and clearance metrics;
- `command_clip_fraction`, `yaw_rate_saturation_fraction`, and `signed_braking_peak_m_s2` from the
  synthetic actuation report;
- jerk, curvature, energy, stalled time, and failure-to-progress where available;
- execution mode, readiness status, availability status, guard/fallback status, and benchmark-track
  metadata for any future track-aware run.

Minimum data/provenance:

- a compact synthetic Issue #1556-style diagnostic surface for first smoke work;
- paired traces with and without the synthetic envelope only after the trace writer can preserve
  command-history and profile metadata durably;
- checkpoint, normalizer, training config, evaluation config, and data split provenance before any
  training issue starts;
- calibrated AMV profile provenance from Issue #1559 before paper-facing or hardware-aligned claims.

## Recommendation

Recommended first follow-up: **create a bounded actuation-aware learned-policy smoke preflight only
after one concrete candidate has a checkpoint or deterministic fixture contract.**

Follow-up issue: <https://github.com/ll7/robot_sf_ll7/issues/1740>

The cleanest first experiment is not a broad training campaign. It is a local smoke/preflight that:

1. uses the Issue #1556 synthetic actuation slice or a one-scenario subset of it;
2. runs one candidate policy family, most likely an actuation-penalized PPO fixture/config or an
   ORCA-residual fixture, against the three classical rows already represented in the smoke surface;
3. validates the Issue #1618 learned-policy adapter metadata before any episodes run;
4. records synthetic profile provenance plus raw/adapted/post-guard actions and actuation
   diagnostics;
5. fails closed for missing checkpoints, missing command-history fields, fallback, degraded
   execution, or unsupported kinematics;
6. publishes only compact diagnostic evidence and explicitly leaves the AMV claim map unchanged
   unless a later calibrated, sufficiently powered campaign justifies a claim update.

## Current Blockers

- Issue #1559 blocks calibrated or paper-facing actuation claims.
- No durable learned-policy dataset or checkpoint currently proves actuation-aware improvement.
- Existing synthetic evidence has zero success for the three classical rows, so the first learned
  smoke must distinguish "the path runs" from "the policy solves the slice."
- Any use of command-history or clipped-history observations must avoid leaking future outcomes or
  post-hoc benchmark labels.
- Planner arbitration and residual approaches need a trace contract that prevents hindsight labels
  from entering inference.

## Validation

This analysis was built from:

```bash
gh issue view 1628 1556 1569 1572 1559 1618 1624 1625
rg -n "actuation|AMV|micromobility|kinematic|latency|learned|PPO|ORCA residual|risk|arbitration" \
  docs/context docs/benchmark* configs scripts tests robot_sf
```

Validation for this docs-only change:

```bash
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check origin/main...HEAD
```
