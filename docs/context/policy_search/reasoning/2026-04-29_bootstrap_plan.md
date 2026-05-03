# Policy Search Bootstrap Plan (2026-04-29)

## Goal

Implement the parts of the policy-search notes that do not require full training runs, keep every result file-based under `docs/context/policy_search/`, and push learning-heavy follow-up work into explicit SLURM handoff notes.

## Boundaries

- Local execution is limited to smoke and other narrow validation slices.
- Full training campaigns and full-matrix promotion work are deferred to `docs/context/policy_search/SLURM/`.
- New work must reuse the existing benchmark stack where possible instead of inventing a parallel execution path.

## Evidence Sources

- `docs/context/policy_search/2026-04-29_policy_search.md`
- `docs/context/policy_search/2026-04-29_broad_policy_search.md`
- `robot_sf/benchmark/map_runner.py`
- Existing experimental planners already present in `robot_sf/planner/`
- `local.machine.md` and `SLURM/AGENTS.md`

## Design Decisions

1. Reuse the existing benchmark and planner interfaces.
   The repo already had ORCA, guarded PPO, MPPI social, NMPC social, and hybrid portfolio surfaces. The missing value was orchestration, gating, and reporting, not a second benchmark framework.

2. Make policy search config-first.
   The policy-search funnel, promotion gates, baselines, and candidates now live under `configs/policy_search/`, with the canonical registry and docs rooted in `docs/context/policy_search/`.

3. Keep all expensive work explicit and deferred.
   Learning-heavy ideas such as learned risk models, shielded PPO repair, and oracle-imitation runs are captured as concrete SLURM handoff notes rather than hidden TODOs.

4. Start with a non-training composite candidate.
   `hybrid_orca_sampler_v1` is implemented as a thin composite over the existing ORCA and MPPI adapters: ORCA remains the default local planner, while MPPI can replace ORCA when short-horizon safety or progress checks show that ORCA is stalling or unsafe.

## Implemented Surfaces

- `docs/context/policy_search/candidate_registry.yaml`
- `configs/policy_search/funnel.yaml`
- `scripts/validation/run_policy_search_candidate.py`
- `scripts/tools/compare_policy_search_candidates.py`
- `scripts/tools/build_policy_search_failure_report.py`
- `scripts/tools/plot_policy_search_pareto_front.py`
- `scripts/tools/promote_policy_search_candidate.py`
- `robot_sf/planner/hybrid_orca_sampler.py`
- `docs/context/policy_search/SLURM/*.md`

## Validation Path

- Unit tests for the shared reporting helpers and runner helpers.
- Planner tests for the new hybrid ORCA sampler logic.
- A real smoke execution of `hybrid_orca_sampler_v1` through the policy-search runner.

## Current Conclusion

The local workflow is now real and runnable, not just documented. The first smoke run proves the new runner, registry, reporting layer, and `hybrid_orca_sampler` integration all work end to end. The current blocker is planner quality, not missing infrastructure: the smoke candidate completes without collisions, but it times out on progress across all three seeds.
