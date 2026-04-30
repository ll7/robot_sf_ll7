# Best Non-Learning Local Navigation Policy Report

## Executive Summary

The best current-code non-learning policy from this iteration is
`hybrid_rule_v3_static_margin0_waypoint2`, a deterministic configuration of the
`hybrid_rule_v3_teb_like_rollout` planner.

It is not promotion-ready. On the 18-episode `nominal_sanity` slice it restores the best observed
success count, keeps zero collision terminations, and improves over v0. On the 24-episode
`stress_slice`, it improves over the previous margin-0 candidate on success and near misses while
preserving zero collisions. It still times out in most dynamic-agent scenarios.

A follow-up route-lookahead branch found one nominal-success gain at 8 cells, but every attempt to
make that branch safety-acceptable either retained static collisions or regressed success/stress
behavior. The selected policy therefore remains the simpler waypoint2 configuration.

Additional static-escape and mild-speed experiments also failed the safety gate. Static escape
reduced nominal near misses but introduced one static collision; 2.2 m/s speed looked safe on
nominal sanity but regressed on stress with two static collisions.

A route-guide commitment bonus for stalled progress was also rejected: it increased route-guide
selection, but over-committed in doorway cases and introduced two nominal static collisions.

## Final Selected Policy

`hybrid_rule_v3_static_margin0_waypoint2`:

- deterministic DWA-style candidate sampling,
- full-rollout static footprint clearance filtering,
- separate `static_hard_safety_margin: 0.0 m` so tight static passages use the reported robot
  radius without an extra static buffer,
- `waypoint_switch_distance: 2.0 m`, which reduced stress-slice route stalls compared with the
  0.9 m setting,
- dynamic-agent collision prediction with SocNav structured pedestrian velocities converted from
  robot ego frame to world frame,
- route-guide candidate from the occupancy-grid route planner,
- explicit selected-candidate, top-k, rejection-reason, and rejected-example diagnostics.

Rejected static-clearance escape and route-commitment knobs remain available only as default-off
ablation controls so the candidate reports can be reproduced. They are not part of the selected
policy.

## Quantitative Results

| Candidate | Stage | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| `hybrid_rule_v0_minimal` | nominal_sanity | 0.1667 | 0.4444 | 0.2222 | 3.9469 | 1.7281 | reject |
| `hybrid_rule_v3_teb_like_rollout` corrected artifact | nominal_sanity | 0.2778 | 0.0000 | 0.1667 | 3.8643 | 1.6867 | reference |
| `hybrid_rule_v3_teb_like_rollout` full static margin | nominal_sanity | 0.2222 | 0.0000 | 0.1667 | 3.8366 | 1.6616 | reject |
| `hybrid_rule_v3_static_margin0` | nominal_sanity | 0.2778 | 0.0000 | 0.2222 | 3.8323 | 1.6868 | reference |
| `hybrid_rule_v3_static_margin0_waypoint2` | nominal_sanity | 0.2778 | 0.0000 | 0.2222 | 3.8495 | 1.7052 | best current |
| `hybrid_rule_v3_waypoint2_static_escape` | nominal_sanity | 0.2778 | 0.0556 | 0.0556 | 4.0204 | 1.7688 | reject |
| `hybrid_rule_v3_waypoint2_speed2p2` | nominal_sanity | 0.2778 | 0.0000 | 0.1667 | 3.8313 | 1.6764 | stress-tested |
| `hybrid_rule_v3_waypoint2_route_commit` | nominal_sanity | 0.1667 | 0.1111 | 0.1667 | 3.7788 | 1.6813 | reject |
| `hybrid_rule_v3_static_margin0_waypoint3` | nominal_sanity | 0.2778 | 0.0556 | 0.1111 | 3.6750 | 1.7100 | reject |
| `hybrid_rule_v3_static_margin0_comfort` | nominal_sanity | 0.2222 | 0.0556 | 0.1111 | 4.0558 | 1.5927 | reject |
| `hybrid_rule_v3_waypoint2_mild_comfort` | nominal_sanity | 0.2778 | 0.0556 | 0.0556 | 3.9394 | 1.7285 | reject |
| `hybrid_rule_v3_waypoint2_progress` | nominal_sanity | 0.2778 | 0.0556 | 0.1111 | 3.8532 | 1.7492 | reject |
| `hybrid_rule_v3_waypoint2_dynamic_clearance` | nominal_sanity | 0.1667 | 0.1111 | 0.2222 | 3.8351 | 1.6665 | reject |
| `hybrid_rule_v3_waypoint2_route_lookahead8` | nominal_sanity | 0.3333 | 0.0556 | 0.1667 | 3.9712 | 1.7855 | reject |
| `hybrid_rule_v3_waypoint2_route_lookahead6` | nominal_sanity | 0.2222 | 0.0000 | 0.1667 | 3.7507 | 1.6964 | reject |
| `hybrid_rule_v3_waypoint2_route_lookahead8_inflation4` | nominal_sanity | 0.2778 | 0.1111 | 0.1111 | 3.9232 | 1.7714 | reject |
| `hybrid_rule_v3_waypoint2_route_lookahead8_static05` | nominal_sanity | 0.2778 | 0.0000 | 0.1667 | 3.8066 | 1.6924 | stress-tested |
| `hybrid_rule_v3_waypoint2_route_lookahead8_static02` | nominal_sanity | 0.2222 | 0.0000 | 0.1667 | 3.8299 | 1.6479 | reject |
| `hybrid_rule_v3_waypoint2_route_lookahead8_clearance1` | nominal_sanity | 0.1667 | 0.1111 | 0.1111 | 3.9191 | 1.6786 | reject |
| `hybrid_rule_v4_recovery_aware` | nominal_sanity | 0.2222 | 0.0000 | 0.1667 | 3.8880 | 1.6510 | reject |
| `hybrid_rule_v3_fast_progress` | nominal_sanity | 0.1667 | 0.0000 | 0.2222 | 3.7599 | 1.5624 | reject |
| `hybrid_rule_v3_dynamic_relaxed` | nominal_sanity | 0.2222 | 0.0000 | 0.1667 | 3.7283 | 1.6644 | reject |
| `hybrid_rule_v3_progress_2p4` | nominal_sanity | 0.2222 | 0.0000 | 0.1667 | 3.7714 | 1.6308 | reject |
| `mpc_clearance_sampler_v1` | nominal_sanity | 0.1667 | 0.2778 | 0.2222 | 4.0883 | 1.5560 | reject |

Primary artifacts:

- `configs/algos/hybrid_rule_v3_static_margin0_waypoint2.yaml`
- `output/policy_search/hybrid_rule_v3_static_margin0_waypoint2_nominal/summary.json`
- `output/policy_search/hybrid_rule_v3_static_margin0_waypoint2_nominal/nominal_sanity__hybrid_rule_v3_static_margin0_waypoint2.jsonl`
- `docs/context/policy_search/reports/2026-04-30_hybrid_rule_v3_static_margin0_waypoint2_nominal_sanity.md`
- `output/policy_search/hybrid_rule_v3_static_margin0_waypoint2_stress/summary.json`
- `docs/context/policy_search/reports/2026-04-30_hybrid_rule_v3_static_margin0_waypoint2_stress_slice.md`

Stress-slice result for the selected policy:

| Candidate | Stage | Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---|---|---:|---:|---:|---:|---:|---:|
| `hybrid_rule_v3_static_margin0` | stress_slice | 24 | 0.2917 | 0.0000 | 0.2500 | 4.7441 | 1.6522 |
| `hybrid_rule_v3_static_margin0_waypoint2` | stress_slice | 24 | 0.3333 | 0.0000 | 0.2083 | 4.7580 | 1.6936 |
| `hybrid_rule_v3_waypoint2_route_lookahead8_static05` | stress_slice | 24 | 0.2917 | 0.0000 | 0.2917 | 4.7495 | 1.6285 |
| `hybrid_rule_v3_waypoint2_speed2p2` | stress_slice | 24 | 0.2500 | 0.0833 | 0.2917 | 4.7483 | 1.5932 |

## Safety Analysis

The selected waypoint2 policy had zero collision terminations on nominal sanity and stress slice.
The full-rollout static clearance check fixed a discovered safety gap where low-speed creep could
pass the hard static-clearance check after the shorter hard horizon. Temporary comfort/progress
retunes and an over-large 3.0 m waypoint handoff were rejected because each introduced a static
collision.

## Comfort Analysis

The selected policy has a 0.2222 nominal near-miss rate and 0.2083 stress-slice near-miss rate.
Comfort retunes reduced nominal near misses, but introduced a static collision, so they were
rejected under the safety-first acceptance rule.

## Efficiency Analysis

Timeouts remain the blocker. The selected policy succeeds on all open sanity seeds and two classic
doorway seeds, improves stress success to 0.3333, but still times out on many head-on, crossing,
overtaking, and Francis-style scenarios.

## What Failed

- Full static clearance with a 5 cm extra margin: collision-free, but too conservative in doorway
  cases and lost one success.
- Stop-margin relaxation: rejected because it introduced a static-collision termination.
- `hybrid_rule_v3_static_margin0_waypoint3`: larger waypoint handoff was too aggressive and
  introduced a static collision.
- `hybrid_rule_v3_static_margin0_comfort`, `hybrid_rule_v3_waypoint2_mild_comfort`, and
  `hybrid_rule_v3_waypoint2_progress`: improved comfort/speed secondary metrics but introduced a
  static collision, so rejected.
- `hybrid_rule_v3_waypoint2_static_escape`: allowing slow, non-worsening motion inside the
  conservative static-clearance band reduced nominal near misses but introduced a static collision.
- `hybrid_rule_v3_waypoint2_speed2p2`: tied waypoint2 on nominal success and reduced nominal near
  misses, but stress-slice success fell to 0.2500 and collision rose to 0.0833.
- `hybrid_rule_v3_waypoint2_route_commit`: boosting route-guide candidates after weak 3 s progress
  over-selected route commands, lost doorway successes, and introduced two static collisions.
- `hybrid_rule_v3_waypoint2_dynamic_clearance`: stronger dynamic-clearance scoring worsened nominal
  success and introduced two static collisions.
- `hybrid_rule_v3_waypoint2_route_lookahead8`: improved nominal success to 0.3333, but introduced a
  static collision, so it cannot be selected under the safety-first rule.
- `hybrid_rule_v3_waypoint2_route_lookahead6`, `hybrid_rule_v3_waypoint2_route_lookahead8_static02`,
  and `hybrid_rule_v3_waypoint2_route_lookahead8_clearance1`: attempted to keep the lookahead
  progress signal, but either lost success or retained static collisions.
- `hybrid_rule_v3_waypoint2_route_lookahead8_static05`: made the lookahead8 branch collision-free
  on nominal sanity, but stress-slice success and near misses regressed versus waypoint2.
- `hybrid_rule_v3_waypoint2_route_lookahead8_inflation4`: route inflation did not address the
  unsafe branch and increased nominal static collisions.
- `hybrid_rule_v4_recovery_aware`: recovery rotations and stall-rotation scoring lost doorway
  successes.
- `hybrid_rule_v3_fast_progress`: higher speed envelope increased intrusive near misses and reduced
  success.
- `hybrid_rule_v3_dynamic_relaxed`: shorter hard dynamic horizon changed the doorway seed mix but
  reduced aggregate success.
- `mpc_clearance_sampler_v1`: unsafe on nominal sanity due static collisions.

## Recommended Paper Interpretation

Supported: a deterministic hybrid-rule planner can remove v0 static collisions and beat v0 success
on the nominal sanity slice without training.

Not supported: claiming a benchmark-ready replacement for existing paper-facing baselines. The
best current hybrid-rule variant remains experimental, timeout-limited, and comfort-limited.

## Validation

- `uv run pytest tests/planner/test_hybrid_rule_local_planner.py -q`
- `uv run ruff check robot_sf/planner/hybrid_rule_local_planner.py robot_sf/benchmark/map_runner.py tests/planner/test_hybrid_rule_local_planner.py`
- `uv run pytest tests/planner/test_hybrid_rule_local_planner.py tests/benchmark/test_algorithm_metadata_contract.py tests/benchmark/test_map_runner_preflight_profiles.py -q`
- `uv run pytest tests/validation/test_policy_search_common.py tests/validation/test_run_policy_search_candidate.py -q`
- `LOGURU_LEVEL=WARNING uv run python scripts/validation/run_policy_search_candidate.py --candidate hybrid_rule_v3_static_margin0_waypoint2 --stage smoke --workers 1 --output-dir output/policy_search/hybrid_rule_v3_static_margin0_waypoint2_smoke`
- `LOGURU_LEVEL=WARNING uv run python scripts/validation/run_policy_search_candidate.py --candidate hybrid_rule_v3_static_margin0_waypoint2 --stage nominal_sanity --workers 1 --output-dir output/policy_search/hybrid_rule_v3_static_margin0_waypoint2_nominal`
- `LOGURU_LEVEL=WARNING uv run python scripts/validation/run_policy_search_candidate.py --candidate hybrid_rule_v3_static_margin0_waypoint2 --stage stress_slice --workers 2 --output-dir output/policy_search/hybrid_rule_v3_static_margin0_waypoint2_stress`
- `LOGURU_LEVEL=WARNING uv run python scripts/validation/run_policy_search_candidate.py --candidate hybrid_rule_v3_waypoint2_static_escape --stage smoke --workers 1 --output-dir output/policy_search/hybrid_rule_v3_waypoint2_static_escape_smoke`
- `LOGURU_LEVEL=WARNING uv run python scripts/validation/run_policy_search_candidate.py --candidate hybrid_rule_v3_waypoint2_static_escape --stage nominal_sanity --workers 1 --output-dir output/policy_search/hybrid_rule_v3_waypoint2_static_escape_nominal`
- `LOGURU_LEVEL=WARNING uv run python scripts/validation/run_policy_search_candidate.py --candidate hybrid_rule_v3_waypoint2_speed2p2 --stage smoke --workers 1 --output-dir output/policy_search/hybrid_rule_v3_waypoint2_speed2p2_smoke`
- `LOGURU_LEVEL=WARNING uv run python scripts/validation/run_policy_search_candidate.py --candidate hybrid_rule_v3_waypoint2_speed2p2 --stage nominal_sanity --workers 1 --output-dir output/policy_search/hybrid_rule_v3_waypoint2_speed2p2_nominal`
- `LOGURU_LEVEL=WARNING uv run python scripts/validation/run_policy_search_candidate.py --candidate hybrid_rule_v3_waypoint2_speed2p2 --stage stress_slice --workers 2 --output-dir output/policy_search/hybrid_rule_v3_waypoint2_speed2p2_stress`
- `LOGURU_LEVEL=WARNING uv run python scripts/validation/run_policy_search_candidate.py --candidate hybrid_rule_v3_waypoint2_route_commit --stage smoke --workers 1 --output-dir output/policy_search/hybrid_rule_v3_waypoint2_route_commit_smoke`
- `LOGURU_LEVEL=WARNING uv run python scripts/validation/run_policy_search_candidate.py --candidate hybrid_rule_v3_waypoint2_route_commit --stage nominal_sanity --workers 1 --output-dir output/policy_search/hybrid_rule_v3_waypoint2_route_commit_nominal`
