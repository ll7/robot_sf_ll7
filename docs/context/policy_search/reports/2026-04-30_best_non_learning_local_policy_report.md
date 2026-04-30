# Best Non-Learning Local Navigation Policy Report

## Executive Summary

The best current-code non-learning policy from this iteration is
`hybrid_rule_v3_static_margin0`, a deterministic configuration of the
`hybrid_rule_v3_teb_like_rollout` planner.

It is not promotion-ready. On the 18-episode `nominal_sanity` slice it restores the best observed
success count, keeps zero collision terminations, and improves over v0, but it still times out in
most dynamic-agent scenarios and has more intrusive near misses than the stricter static-margin
variant.

## Final Selected Policy

`hybrid_rule_v3_static_margin0`:

- deterministic DWA-style candidate sampling,
- full-rollout static footprint clearance filtering,
- separate `static_hard_safety_margin: 0.0 m` so tight static passages use the reported robot
  radius without an extra static buffer,
- dynamic-agent collision prediction with SocNav structured pedestrian velocities converted from
  robot ego frame to world frame,
- route-guide candidate from the occupancy-grid route planner,
- explicit selected-candidate, top-k, rejection-reason, and rejected-example diagnostics.

## Quantitative Results

| Candidate | Stage | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| `hybrid_rule_v0_minimal` | nominal_sanity | 0.1667 | 0.4444 | 0.2222 | 3.9469 | 1.7281 | reject |
| `hybrid_rule_v3_teb_like_rollout` corrected artifact | nominal_sanity | 0.2778 | 0.0000 | 0.1667 | 3.8643 | 1.6867 | reference |
| `hybrid_rule_v3_teb_like_rollout` full static margin | nominal_sanity | 0.2222 | 0.0000 | 0.1667 | 3.8366 | 1.6616 | reject |
| `hybrid_rule_v3_static_margin0` | nominal_sanity | 0.2778 | 0.0000 | 0.2222 | 3.8323 | 1.6868 | best current |
| `hybrid_rule_v4_recovery_aware` | nominal_sanity | 0.2222 | 0.0000 | 0.1667 | 3.8880 | 1.6510 | reject |
| `hybrid_rule_v3_fast_progress` | nominal_sanity | 0.1667 | 0.0000 | 0.2222 | 3.7599 | 1.5624 | reject |
| `hybrid_rule_v3_dynamic_relaxed` | nominal_sanity | 0.2222 | 0.0000 | 0.1667 | 3.7283 | 1.6644 | reject |
| `hybrid_rule_v3_progress_2p4` | nominal_sanity | 0.2222 | 0.0000 | 0.1667 | 3.7714 | 1.6308 | reject |
| `mpc_clearance_sampler_v1` | nominal_sanity | 0.1667 | 0.2778 | 0.2222 | 4.0883 | 1.5560 | reject |

Primary artifacts:

- `output/policy_search/hybrid_rule_v3_static_margin0_nominal/summary.json`
- `output/policy_search/hybrid_rule_v3_static_margin0_nominal/nominal_sanity__hybrid_rule_v3_static_margin0.jsonl`
- `docs/context/policy_search/reports/2026-04-30_hybrid_rule_v3_static_margin0_nominal_sanity.md`
- `output/policy_search/hybrid_rule_v3_static_margin0_stress/summary.json`
- `docs/context/policy_search/reports/2026-04-30_hybrid_rule_v3_static_margin0_stress_slice.md`

Stress-slice result for the selected policy:

| Candidate | Stage | Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---|---|---:|---:|---:|---:|---:|---:|
| `hybrid_rule_v3_static_margin0` | stress_slice | 24 | 0.2917 | 0.0000 | 0.2500 | 4.7441 | 1.6522 |

## Safety Analysis

The selected margin-0 v3 policy had zero collision terminations on nominal sanity. The full-rollout
static clearance check fixed a discovered safety gap where low-speed creep could pass the hard
static-clearance check after the shorter hard horizon. A temporary stop-margin relaxation was
rejected because it introduced a static-collision termination.

## Comfort Analysis

The selected policy has a 0.2222 near-miss rate, worse than the stricter full-margin v3 result
at 0.1667. The selected policy is therefore a success/safety tradeoff, not a comfort improvement.
Doorway and overtaking cases remain the main intrusive-near-miss sources.

## Efficiency Analysis

Timeouts remain the blocker. The selected policy succeeds on all open sanity seeds and two classic
doorway seeds, but still times out on head-on corridor, crossing, overtaking, following-human, and
one doorway seed.

## What Failed

- Full static clearance with a 5 cm extra margin: collision-free, but too conservative in doorway
  cases and lost one success.
- Stop-margin relaxation: rejected because it introduced a static-collision termination.
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
- `LOGURU_LEVEL=WARNING uv run python scripts/validation/run_policy_search_candidate.py --candidate hybrid_rule_v3_static_margin0 --stage smoke --workers 1 --output-dir output/policy_search/hybrid_rule_v3_static_margin0_smoke`
- `LOGURU_LEVEL=WARNING uv run python scripts/validation/run_policy_search_candidate.py --candidate hybrid_rule_v3_static_margin0 --stage nominal_sanity --workers 1 --output-dir output/policy_search/hybrid_rule_v3_static_margin0_nominal`
- `LOGURU_LEVEL=WARNING uv run python scripts/validation/run_policy_search_candidate.py --candidate hybrid_rule_v3_static_margin0 --stage stress_slice --workers 2 --output-dir output/policy_search/hybrid_rule_v3_static_margin0_stress`
