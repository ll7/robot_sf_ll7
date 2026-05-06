# Full-Matrix H500 Policy Search Analysis (2026-05-05)

## Scope

This note summarizes the tracked `full_matrix_h500` policy-search reports. The initial 23-candidate
matrix was produced from commit `47fecd938482949b7989f1011ec6e34237d8b45d`; the two leader rerun
reports were refreshed by the clean pinned Slurm rerun on commit
`2b796ea92104467d3bc913528801fb8bb11034dd`.

- Stage: `full_matrix_h500`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Candidate reports: 23 tracked reports under `docs/context/policy_search/reports/`
- Episode count: 144 per candidate
- Local source artifacts: `output/policy_search/*/full_matrix_h500/*/summary.json` and combined
  JSONL files referenced by each report

These are policy-search triage artifacts, not paper-facing benchmark claims. The output artifacts
under `output/` are worktree-local; the tracked Markdown reports and this synthesis are the durable
record for handoff.

## Clean Leader Rerun

Slurm job `12339` reran `scenario_adaptive_hybrid_orca_v1` and
`hybrid_rule_v3_fast_progress` with `--clean-pinned` under run id
`policy_search_full_matrix_h500_leaders_clean_20260505_204501`.

- Both array tasks completed successfully (`0:0`) on `auxme-imech172`.
- The clean rerun reproduced the earlier h500 metrics exactly for both candidates.
- The only tracked per-candidate report deltas are provenance updates to the clean summary paths and
  commit hash.
- Log review found repeated global-planner invalid-goal-cell messages, but no Slurm/job failure,
  traceback, integrity contradiction, or benchmark fallback/degraded execution mode. Planner
  diagnostics show adapter execution throughout; internal `EMERGENCY_STOP` command fallbacks remain
  episode-level behavior, not fallback planner substitution.

## Aggregate Ranking

| Rank | Candidate | Success | Collision | Near Miss | Classic Success | Francis Success | Main Failures |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `scenario_adaptive_hybrid_orca_v1` | 0.9097 | 0.0208 | 0.4236 | 0.8696 | 0.9467 | `timeout_low_progress` 6, `static_collision` 3 |
| 2 | `hybrid_rule_v3_fast_progress_static_escape` | 0.9028 | 0.0208 | 0.4236 | 0.8696 | 0.9333 | `timeout_low_progress` 6, `static_collision` 3 |
| 3 | `hybrid_rule_v3_fast_progress` | 0.8264 | 0.0139 | 0.4236 | 0.7391 | 0.9067 | `timeout_low_progress` 11, `static_collision` 2 |
| 4 | `hybrid_rule_v3_progress_2p4` | 0.8056 | 0.0139 | 0.4097 | 0.6957 | 0.9067 | `timeout_low_progress` 14, `static_collision` 2 |
| 5 | `hybrid_rule_v4_recovery_aware` | 0.8056 | 0.0139 | 0.4097 | 0.6957 | 0.9067 | `timeout_low_progress` 14, `static_collision` 2 |
| 6 | `hybrid_rule_v3_dynamic_relaxed` | 0.7778 | 0.0139 | 0.4167 | 0.6667 | 0.8800 | `timeout_low_progress` 15, `static_collision` 2 |
| 7 | `hybrid_rule_v3_waypoint2_route_lookahead8_static02` | 0.7778 | 0.0139 | 0.4097 | 0.6522 | 0.8933 | `timeout_low_progress` 15, `static_collision` 2 |
| 8 | `hybrid_rule_v3_waypoint2_route_lookahead8_static05` | 0.7778 | 0.0139 | 0.4097 | 0.6522 | 0.8933 | `timeout_low_progress` 15, `static_collision` 2 |
| 9 | `hybrid_rule_v3_teb_like_rollout` | 0.7708 | 0.0139 | 0.4097 | 0.6667 | 0.8667 | `timeout_low_progress` 15, `static_collision` 2 |
| 10 | `hybrid_rule_v0_minimal` | 0.7361 | 0.0208 | 0.4167 | 0.6667 | 0.8000 | `timeout_low_progress` 23, `static_collision` 3 |

The remaining route-lookahead and waypoint variants mostly trade extra success for static
collisions. `mpc_clearance_sampler_v1` remains attractive only as a research component because it
has the lowest near-miss rate in this h500 set (`0.1597`) but a very high collision rate (`0.3750`).

## Horizon Effect

The h500 rerun mostly reveals that the short `full_matrix` horizon was truncating route completion,
not that a different planner family suddenly became dominant.

| Candidate | Full Matrix Success | H500 Success | Success Delta | Collision Delta |
|---|---:|---:|---:|---:|
| `scenario_adaptive_hybrid_orca_v1` | 0.2778 | 0.9097 | +0.6319 | +0.0069 |
| `hybrid_rule_v3_fast_progress_static_escape` | 0.2639 | 0.9028 | +0.6389 | +0.0069 |
| `hybrid_rule_v3_fast_progress` | 0.2569 | 0.8264 | +0.5694 | +0.0000 |
| `hybrid_rule_v3_progress_2p4` | 0.2708 | 0.8056 | +0.5347 | +0.0000 |
| `hybrid_rule_v4_recovery_aware` | 0.2500 | 0.8056 | +0.5556 | +0.0000 |

The top two candidates gain the most success but also add one extra collision relative to the
shorter full-matrix run. This matters because the strict `nominal_sanity` gate allows at most
`0.0200` collision rate, while the top two finish at `0.0208` (3 collisions in 144 episodes).

## Promotion Decision

Current h500 leader:

- Promote `scenario_adaptive_hybrid_orca_v1` as the current experimental h500 leader. It has the
  best aggregate success/collision tradeoff (`0.9097` success, `0.0208` collision), with especially
  strong Francis-family success (`0.9467`).

Strict gate-clean h500 candidates:

- Promote `hybrid_rule_v3_fast_progress` as the simplest strict-gate h500 candidate. It is the
  best candidate that passes success `>= 0.80`, collision `<= 0.02`, and the scenario-stratified
  collision guard (`0.8264` success, `0.0139` collision).
- Keep `hybrid_rule_v3_progress_2p4` and `hybrid_rule_v4_recovery_aware` as backup ablations. They
  both pass the same strict gate at `0.8056` success and `0.0139` collision, but neither beats
  `hybrid_rule_v3_fast_progress` on aggregate success.

Track but do not promote as primary:

- `hybrid_rule_v3_fast_progress_static_escape`: nearly tied with the leader (`0.9028` success) but
  has the same one-collision-over-strict-gate issue and no aggregate advantage over the
  scenario-adaptive candidate.
- `hybrid_rule_v3_dynamic_relaxed`, route-lookahead static-margin variants, and
  `hybrid_rule_v3_teb_like_rollout`: useful ablations, but all fall below the `0.80` strict success
  floor.
- `mpc_clearance_sampler_v1`: do not promote; use only as a future clearance-cost/proposer idea
  behind a hard static safety filter.

Before any camera-ready or paper-facing promotion, rerun the recommended candidates from a clean,
pinned worktree and include horizon-specific evidence in the benchmark claim.

## Remaining Scenario Blockers

H500 does not solve every case. The aggregate leader still fails:

- `classic_merging_medium`: 0/3 success, 3 timeouts.
- `classic_station_platform_medium`: 0/3 success, 3 timeouts.
- `francis2023_narrow_doorway`: 0/3 success, 3 timeouts.
- `classic_merging_low`: 1/3 success, with one collision and one timeout.
- `classic_cross_trap_high`: 2/3 success, with one collision.
- `francis2023_circular_crossing`: 2/3 success, with one collision.

The no-success-at-h500 cases should be treated as planner/design blockers rather than horizon
selection problems.

## Research Directions

1. Horizon-aware benchmark contracts. H500 shows route-completion budget is a first-class variable:
   aggregate success rises by roughly 0.5-0.64 for the top rule-based family. Future reports should
   separate planner quality from time-budget sufficiency.
2. Deadlock and route-local-minimum recovery. The remaining hard cases are merging, station
   platform, and narrow doorway scenes where more horizon does not create progress. Work should move
   from local DWA scoring toward deliberate retreat/recenter/re-route actions with hysteresis.
3. Comfort-preserving high-success planning. The best h500 candidates succeed but retain high
   near-miss rates around `0.41-0.42`. Add speed-dependent near-miss/comfort terms or scenario
   affordance gates before treating high success as socially acceptable.
4. Selector with safety accounting. Scenario-adaptive selection is the aggregate leader, but its
   advantage over `hybrid_rule_v3_fast_progress_static_escape` is only one episode. Any next
   selector should require per-scenario evidence that the override improves success without adding
   static collisions.
5. MPC as a proposer, not an executor. `mpc_clearance_sampler_v1` has low near-miss exposure but
   unacceptable static-collision behavior. The promising direction is to reuse its clearance score
   under the hybrid rule planner's hard safety filter.

## Scenario Horizon Policy

Use scenario-specific horizons rather than one fixed budget for every scenario.

Recommended decision rule:

1. For each scenario, collect successful completion step counts from the safe incumbent set
   (`collision <= 0.03`, success `>= 0.75`) across pinned h500 runs.
2. Set the candidate horizon to `ceil(p95_success_steps + 20)` with a small floor for very short
   scenes. Use the route-length/nominal-speed estimate as a sanity check when `shortest_path_len` or
   equivalent path metrics are present.
3. If `p95_success_steps` is near the cap or the scenario still has many h500 timeouts, run an h600
   probe before deciding.
4. If no safe incumbent succeeds by h500, mark the scenario as planner-blocked and keep the horizon
   at h500 only for diagnosis. Do not keep raising the horizon and calling the resulting failure a
   fair route-completion test.

Observed buckets from the h500 evidence:

| Horizon Bucket | Candidate Scenarios | Recommended Handling |
|---|---|---|
| 90-150 steps | elevator/room entry-exit, group crossing, intersections, `francis2023_leave_group` | Short fixed budget is enough; use h150 unless route length changes. |
| 180-280 steps | bottlenecks, corridors, T-intersections, most Francis following/traffic scenes | Use h250-h300, selected by per-scenario p95. |
| 350-475 steps | cross-trap, overtaking, `classic_realworld_double_bottleneck_high`, some doorway seeds | Use h450-h500 and run h600 only if p95 is near 500. |
| Planner-blocked at h500 | `classic_merging_medium`, `classic_station_platform_medium`, `francis2023_narrow_doorway` | Do not solve with horizon; require planner or scenario-contract work. |

## Validation

- Refreshed portfolio overview:
  `uv run python scripts/tools/summarize_policy_search_portfolio.py --output-md docs/context/policy_search/portfolio_overview_2026-05-05.md --output-json docs/context/policy_search/portfolio_overview_2026-05-05.json`
- Parsed the 23 h500 report summary paths and combined JSONL files referenced by those reports to
  compute aggregate ranking, h100-to-h500 deltas, scenario blockers, gate checks, and horizon
  buckets.
- Reviewed clean leader rerun job `12339` with `sacct`; both tasks completed with exit code `0:0`
  in about 24 minutes, and the rerun summaries matched the earlier h500 metrics exactly.
- Refreshed strict h500 gate reports and scenario horizon source paths so the two rerun leaders now
  reference `policy_search_full_matrix_h500_leaders_clean_20260505_204501`.
- Follow-up implementation artifacts:
  - Clean pinned rerun handoff:
    `docs/context/policy_search/SLURM/004_h500_leader_clean_rerun.md`
  - Strict h500 gate reports:
    `docs/context/policy_search/reports/promotions/2026-05-05_full_matrix_h500_strict_gate/`
  - Scenario horizon recommendations:
    `docs/context/policy_search/reports/2026-05-05_h500_horizon_recommendations.md`
    and `configs/policy_search/scenario_horizons_h500.yaml`
  - Research plan:
    `docs/context/policy_search/reasoning/2026-05-05_h500_research_plan.md`
- Tooling checks for the follow-up implementation:
  - `uv run pytest tests/tools/test_promote_policy_search_candidate.py tests/tools/test_suggest_policy_search_horizons.py`
  - `uv run ruff check scripts/tools/suggest_policy_search_horizons.py scripts/tools/promote_policy_search_candidate.py scripts/tools/summarize_policy_search_portfolio.py tests/tools/test_promote_policy_search_candidate.py tests/tools/test_suggest_policy_search_horizons.py`
  - `uv run ruff format --check scripts/tools/suggest_policy_search_horizons.py scripts/tools/promote_policy_search_candidate.py scripts/tools/summarize_policy_search_portfolio.py tests/tools/test_promote_policy_search_candidate.py tests/tools/test_suggest_policy_search_horizons.py`
  - `bash -n scripts/dev/sbatch_policy_search_sweep.sh SLURM/Auxme/policy_search_candidate_stage.sl`
  - `scripts/dev/sbatch_policy_search_sweep.sh --stage full_matrix_h500 --candidates-file configs/policy_search/candidate_sets/h500_leader_clean_rerun.txt --run-id policy_search_full_matrix_h500_leaders_clean_dryrun --throttle 1 --workers 2 --clean-pinned --no-status --dry-run`
  - `git diff --check`
