# Issue #2180 One-Factor Hybrid Component h500 Run

Status: current, complete local h500 diagnostic evidence.

Issue #2180 executes the Issue #2170 one-factor manifest at h500 after the h80 staged pilots
resolved executability and the selector ORCA-extra dependency. All rows ran in native local mode
with the `orca` extra synced and `rvo2` import proven.

## Evidence

- Compact evidence bundle:
  `docs/context/evidence/issue_2180_one_factor_h500_2026-06-03/`
- Manifest:
  `configs/policy_search/ablation_manifests/issue_2170_one_factor_hybrid_component_manifest.yaml`
- Predecessor notes:
  [issue_2174_one_factor_ablation_pilot.md](issue_2174_one_factor_ablation_pilot.md),
  [issue_2176_remaining_one_factor_h80.md](issue_2176_remaining_one_factor_h80.md), and
  [issue_2178_selector_orca_extra_h80.md](issue_2178_selector_orca_extra_h80.md)

## Result

All eight candidates wrote 18/18 jobs with zero failed jobs at h500.

| Comparison | Success delta | Collision delta | Near-miss delta | Avg-speed delta | Runtime delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| static_escape_only_minus_base | 0.000 | 0.000 | 0.000 | 0.000 | +27.690s |
| static_recenter_only_minus_base | +0.056 | 0.000 | 0.000 | +0.075 | +17.418s |
| escape_recenter_pair_minus_static_escape_only | +0.111 | 0.000 | 0.000 | +0.116 | -12.234s |
| grouped_transit_minus_escape_recenter_pair | 0.000 | 0.000 | 0.000 | -0.000 | -0.128s |
| continuous_checks_minus_grouped_static | -0.111 | 0.000 | -0.222 | -0.071 | +11.542s |
| selector_only_minus_grouped_static | -0.056 | 0.000 | 0.000 | -0.057 | -10.289s |
| speed_progress_2p4_minus_base | -0.056 | 0.000 | +0.111 | -0.004 | +5.369s |

## Interpretation

Confidence is about 0.80 that recentering, not static escape alone, is the clearest positive
component in this manifest slice. Static escape alone is identical to the base row on success,
collision, near-miss, and average speed, but costs runtime. Recenter-only improves one of 18 rows
over base; adding recentering after static escape improves two of 18 rows over static escape alone.

Confidence is about 0.75 that corridor-transit terms are neutral in this slice: the grouped static
row and escape-plus-recenter row are effectively identical. Continuous static checks look like a
safety/progress trade-off: fewer near misses but two fewer successes versus grouped static.
Selector-only and speed/progress-2.4 are not supported by this h500 slice as independent gains.

## Claim Boundary

- Local h500 diagnostic evidence, not a planner-promotion claim.
- Native local execution with zero failed jobs; no fallback/degraded/unavailable rows.
- The sample is still only the manifest's 18-row scenario/seed slice, so treat one-row and two-row
  success deltas as directional component evidence, not final paper-facing causality.
