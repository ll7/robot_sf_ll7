# Issue #2261 Static-Recenter Slice-Local Explanation

Date: 2026-06-05
Issue: <https://github.com/ll7/robot_sf_ll7/issues/2261>
Parent evidence: Issues #2180, #2221, and #2266.

## Scope

This note explains why static recentering should stay classified as a slice-local diagnostic
mechanism after the Issue #2221 held-out smoke. It does not tune static recentering, run a broader
benchmark, produce paper-facing panels, or claim definitive per-step activation behavior.

Compact evidence summary:
`docs/context/evidence/issue_2261_static_recenter_slice_local_2026-06-05/summary.json`.

## Observed Evidence

Issue #2180 created the positive local signal: on the h500 one-factor manifest,
`issue_2170_static_recenter_only` improved success by `+0.056` over the baseline with no collision
or near-miss delta. That was directional component evidence over an 18-row diagnostic slice, not a
planner-promotion claim.

Issue #2221 then ran the held-out scenario-family smoke for the same mechanism row against
`hybrid_rule_v3_fast_progress` with seed `111`, horizon `500`, and two held-out scenarios. Terminal
outcomes were exactly preserved:

| Scenario | Baseline | Static recenter | Terminal delta |
| --- | --- | --- | --- |
| `classic_station_platform_medium` | `max_steps`, 60 near misses, 0 collisions | `max_steps`, 60 near misses, 0 collisions | none |
| `francis2023_intersection_wait` | success, 0 near misses, 0 collisions | success, 0 near misses, 0 collisions | none |

Issue #2266 checked the durable compact evidence and found that activation count, first activation
step, selected command source, and progress/trajectory delta were not preserved. Therefore the
strong observed conclusion is terminal parity plus missing activation evidence, not proof that
static recentering never activated.

## Mechanism Fit

The static-recenter term is gated in `robot_sf/planner/hybrid_rule_local_planner.py`: it only scores
when the planner is stalled, the robot is farther than `goal_far_distance`, the nearest pedestrian
is at least `slow_distance_human` away, and the candidate is a rotate-in-place command whose probed
forward heading clears static obstacles. In the base config used by
`issue_2170_static_recenter_only`, those key thresholds are:

| Gate | Value |
| --- | ---: |
| `slow_distance_human` | `1.0` |
| `freezing_speed_threshold` | `0.05` |
| `goal_far_distance` | `0.8` |
| `deadlock_rotation_threshold` | `0.15` |
| `static_recenter_probe_speed` | `0.3` |

That makes static recentering a narrow mechanism for static-obstacle heading recovery away from
nearby pedestrians. The two held-out rows do not strongly match that mechanism:

- `classic_station_platform_medium` is a pedestrian-flow station scene with `ped_density: 0.05`,
  explicit waiting pedestrians, 60 near misses, and no success improvement. The failure looks
  pedestrian-intrusive and progress-limited, which is structurally different from a clean
  static-obstacle recentering failure mode. The nearest-pedestrian gate may also suppress useful
  recentering during the crowded portions, but the durable artifacts do not prove that per step.
- `francis2023_intersection_wait` already succeeds with the baseline, has `ped_density: 0.0`, and
  records no near misses or collisions in the held-out smoke. It does not provide a failed static
  local-minimum case for recentering to rescue.

## Explanation

The most conservative explanation is:

1. Static recentering had a directional local signal in the Issue #2180 diagnostic slice.
2. The Issue #2221 held-out rows either did not contain the static-recenter failure mode, or did
   not expose it in a way that changed the selected command source, progress, trajectory, or
   terminal outcome.
3. Because activation and command-source traces are missing, the repository cannot distinguish
   "never activated" from "activated but irrelevant" or "activated but lost arbitration."

Classification: `slice_local_boundary`.

Confidence: `0.78` that static recentering should not be promoted as a transferable mechanism from
the current evidence. Confidence is lower, about `0.55`, for the exact causal explanation because
the held-out activation trace is missing.

## Stop / Revise / Narrow Recommendation

Recommendation: `narrow`.

Keep static recentering as a narrow static-obstacle recovery probe for diagnostic slices where the
robot is stalled away from pedestrians. Do not broaden the mechanism claim to held-out
scenario-family transfer unless an instrumented rerun records activation count, first activation
step, selected command source, and progress/trajectory delta.

The smallest useful follow-up, if maintainers want definitive attribution, is an instrumented rerun
of the Issue #2221 held-out smoke, starting with `classic_station_platform_medium` seed `111`.
That rerun should record per-step `selected_terms["static_recenter"]`, `selected_source`,
nearest-pedestrian distance, progress windows, and route-progress delta. Until that exists, the
result should remain diagnostic analysis-only evidence.

## Claim Boundary

This note is not benchmark evidence and not a paper-facing mechanism claim. It is a conservative
research-direction update: static recentering remains useful as local diagnostic evidence, but the
held-out smoke establishes a transfer boundary rather than a transferable improvement.

## Validation

```bash
python -m json.tool docs/context/evidence/issue_2261_static_recenter_slice_local_2026-06-05/summary.json
bash scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check
```
