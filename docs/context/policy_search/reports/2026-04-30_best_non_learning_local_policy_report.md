# Best Non-Learning Local Navigation Policy Report

## Executive Summary

The best empirically supported non-learning policy from this iteration is
`hybrid_rule_v3_teb_like_rollout`, after correcting pedestrian velocity frame handling in
`robot_sf/planner/hybrid_rule_local_planner.py`.

It is not promotion-ready. On the 18-episode `nominal_sanity` slice it improves over v0 and remains
collision-free, but it still times out in most dynamic-agent scenarios.

## Final Selected Policy

`hybrid_rule_v3_teb_like_rollout`:

- deterministic DWA-style candidate sampling,
- static footprint clearance filtering,
- dynamic-agent collision prediction with SocNav structured pedestrian velocities converted from
  robot ego frame to world frame,
- route-guide candidate from the occupancy-grid route planner,
- explicit score diagnostics and candidate rejection counters.

## Quantitative Results

| Candidate | Stage | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| `hybrid_rule_v0_minimal` | nominal_sanity | 0.1667 | 0.4444 | 0.2222 | 3.9469 | 1.7281 | reject |
| `hybrid_rule_v3_teb_like_rollout` | nominal_sanity | 0.2778 | 0.0000 | 0.1667 | 3.8516 | 1.6830 | best current |
| `hybrid_rule_v4_recovery_aware` | nominal_sanity | 0.2222 | 0.0000 | 0.1667 | 3.8880 | 1.6510 | reject |
| `hybrid_rule_v3_fast_progress` | nominal_sanity | 0.1667 | 0.0000 | 0.2222 | 3.7599 | 1.5624 | reject |
| `hybrid_rule_v3_dynamic_relaxed` | nominal_sanity | 0.2222 | 0.0000 | 0.1667 | 3.7283 | 1.6644 | reject |
| `mpc_clearance_sampler_v1` | nominal_sanity | 0.1667 | 0.2778 | 0.2222 | 4.0883 | 1.5560 | reject |

Primary artifacts:

- `output/policy_search/hybrid_rule_v3_teb_like_rollout_nominal_final/summary.json`
- `output/policy_search/hybrid_rule_v3_teb_like_rollout_nominal_final/nominal_sanity__hybrid_rule_v3_teb_like_rollout.jsonl`
- `docs/context/policy_search/reports/2026-04-30_hybrid_rule_v3_teb_like_rollout_nominal_sanity.md`

## Safety Analysis

The selected v3 policy had zero collision terminations on nominal sanity. The static footprint
clearance filter removed the v0 static-collision failure mode. The MPC-clearance comparison was
rejected because it introduced a 0.2778 collision rate on the same slice.

## Comfort Analysis

The selected v3 policy still has a 0.1667 near-miss rate. Doorway and overtaking cases remain the
main intrusive-near-miss sources. Faster progress and relaxed dynamic horizons did not improve this.

## Efficiency Analysis

Timeouts remain the blocker. Corrected v3 succeeds on all open sanity seeds and two doorway seeds,
but still times out on head-on corridor, crossing, overtaking, following-human, and one doorway
seed. Recovery-aware variants reduced some fallback counts but did not improve aggregate success.

## What Failed

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
best current hybrid-rule variant remains experimental and timeout-limited.

## Validation

- `uv run ruff check robot_sf/planner/hybrid_rule_local_planner.py tests/planner/test_hybrid_rule_local_planner.py`
- `uv run pytest tests/planner/test_hybrid_rule_local_planner.py -q`
- `uv run pytest tests/planner/test_hybrid_rule_local_planner.py tests/benchmark/test_algorithm_metadata_contract.py tests/benchmark/test_map_runner_preflight_profiles.py -q`
- `LOGURU_LEVEL=WARNING uv run python scripts/validation/run_policy_search_candidate.py --candidate hybrid_rule_v3_teb_like_rollout --stage smoke --workers 1 --output-dir output/policy_search/hybrid_rule_v3_teb_like_rollout_smoke_final`
- `LOGURU_LEVEL=WARNING uv run python scripts/validation/run_policy_search_candidate.py --candidate hybrid_rule_v3_teb_like_rollout --stage nominal_sanity --workers 1 --output-dir output/policy_search/hybrid_rule_v3_teb_like_rollout_nominal_final`
