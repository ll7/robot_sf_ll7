# Experimental Planner Guardrails

This repository now distinguishes between:

- `baseline-ready`: allowed in baseline-safe and paper-facing benchmark profiles
- `experimental`: allowed in exploratory benchmark profiles
- `experimental-testing`: planners that are implemented and tested, but should not silently enter
  normal benchmark sweeps until they have demonstrated stable benchmark value

## Planner Contract Boundary

Benchmark-facing planners also publish planner contract metadata through
`robot_sf.benchmark.algorithm_metadata`. The contract records the planner-facing observation mode,
required inputs, normalization label, command space, action keys, and reset convention before a
benchmark runner uses the planner.

This metadata is a compatibility guard only. It makes mismatched observation/action assumptions
fail before benchmark execution, but it is not performance evidence and must not be cited as proof
that a planner is baseline-ready, paper-facing, or high quality.

## Policy-Search Candidate Configs

Policy-search candidate planner configs live under
`configs/policy_search/candidates/`. This is the first place to look for candidate variants such as
`hybrid_rule_v3_static_margin0.yaml`, `hybrid_rule_v3_static_margin0_waypoint2.yaml`,
`hybrid_rule_v3_fast_progress.yaml`, and `hybrid_rule_v3_fast_progress_static_escape.yaml`.

`configs/algos/` is the benchmark-facing algorithm-config directory. It can contain promoted,
compatibility, or direct benchmark wrappers, and some entries intentionally point back to candidate
configs. Do not assume a candidate exists under `configs/algos/` just because a similarly named
policy-search file exists under `configs/policy_search/candidates/`.

Preferred local entrypoints:

```bash
uv run python scripts/validation/run_policy_search_candidate.py \
  --candidate hybrid_rule_v3_fast_progress \
  --stage smoke
```

```bash
uv run python scripts/validation/run_policy_search_step_diagnostics.py \
  --candidate hybrid_rule_v3_static_margin0 \
  --stage smoke
```

The runners resolve candidate names through `docs/context/policy_search/candidate_registry.yaml`,
whose entries point back to the candidate config files. Use the candidate runner for compact
candidate smoke/evaluation loops. Use the step-diagnostics runner when debugging why a candidate
action, clearance, progress, or recovery decision changed. Both commands load repository
dependencies through `uv run python`, so they avoid accidental system-Python import failures.

## Testing-only planners

The following planners are currently treated as testing-only and fail closed by default:

- `risk_dwa`
- `hybrid_rule_local_planner`
- `mppi_social`
- `predictive_mppi`
- `hybrid_portfolio`
- `stream_gap`
- `gap_prediction`
- `sicnav`
- `dr_mpc`
- `trivial_reference` / `reference_adapter` (diagnostic starter template only)

These planners are blocked unless their algo config explicitly contains:

```yaml
allow_testing_algorithms: true
```

This keeps unfinished planner families available for controlled R&D while preventing accidental
inclusion in camera-ready or routine exploratory benchmark runs.

## Why this guard exists

These planners have unit/integration coverage and can be exercised locally, but their benchmark
performance is still unstable or materially below the current champion policy family. The guard
prevents:

- accidental inclusion in broad benchmark matrices
- accidental regression interpretation from incomplete planners
- confusing benchmark tables where low-readiness planners appear alongside promotion candidates

## Promotion rule

A testing-only planner should not lose the guard until it has:

1. repeatable benchmark evidence on the intended suite
2. contradiction-free outputs
3. documented failure modes and runtime limits
4. a clear reason to exist alongside current baselines

When those conditions are met, update `robot_sf/benchmark/algorithm_readiness.py` and remove the
opt-in requirement for that planner.

## Issue 596 stage-gate policy

Issue 596 adds a stricter staging layer before any testing-only planner can be reconsidered for
routine benchmark use:

1. pass the verified-simple suite under
   `configs/scenarios/sets/verified_simple_subset_v1.yaml` at a calibrated bar
2. remain contradiction-free on that suite
3. only then spend time on broader benchmark evidence
4. only remove the guard after the broader benchmark evidence is also accepted

The verified-simple suite is a necessary gate, not sufficient evidence for promotion.

Supporting issue-596 notes:

- `docs/context/issue_596_verified_simple_gate_proposal.md`
- `docs/context/issue_596_atomic_scenario_matrix.md`

## Current planner-specific promotion criteria

The following planner-specific evidence notes are the current citation surface for the testing-only
testing-only planners. None of them currently meet the promotion bar.

| Planner | Evidence note | Current blocker | Promotion criterion | Next proof required |
| --- | --- | --- | --- | --- |
| `risk_dwa` | `docs/context/issue_679_risk_dwa_benchmark.md` | Much lower success and higher collisions than predictive baselines despite faster runtime. | Show a concrete revision that recovers goal-reaching to a competitive level without giving back the reactive safety/runtime advantages. | Rerun benchmark evidence only after a hypothesis-driven change to progress recovery or action selection. |
| `hybrid_rule_local_planner` | `docs/context/policy_search/reports/2026-04-30_best_non_learning_local_policy_report.md` | Best current hybrid-rule configuration is `hybrid_rule_v3_static_margin0_waypoint2`: collision-free on nominal sanity and stress slice, with better stress success/near-miss rates than the previous margin-0 candidate. Follow-up route-lookahead, route-commitment, static-escape, mild-speed, and comfort/progress variants either introduced static collisions or regressed success/stress behavior. | Show broader matrix gains without increasing collisions or severe timeout/freezing. | Investigate doorway comfort and dynamic-agent passage logic before adding broader recovery or ensemble claims. |
| `mppi_social` | `docs/context/issue_677_mppi_social_benchmark.md` | Severe runtime cost plus lower success than predictive baselines. | Demonstrate a materially better runtime-success tradeoff than the current predictive baselines while preserving the observed safety signal. | Provide a concrete runtime-reduction or horizon-simplification hypothesis, then rerun the verified-simple gate and broader benchmark. |
| `predictive_mppi` | `docs/context/issue_675_predictive_mppi_benchmark.md` | Roughly order-of-magnitude runtime penalty plus worse success and collision outcomes. | Show that the control-search change produces either clear outcome gains or a defensible safety/runtime win over the predictive baselines. | Land a new search or sampling hypothesis first; do not rerun the unchanged config. |
| `hybrid_portfolio` | `docs/context/issue_673_hybrid_portfolio_benchmark.md` | Planner switching loses both success and runtime efficiency while still colliding more. | Prove that portfolio arbitration yields a measurable benchmark benefit beyond its component planners, not just a near-miss reduction. | Revisit only with explicit evidence about why the switching logic should improve commitment and runtime behavior. |
| `stream_gap` | `docs/context/issue_681_stream_gap_benchmark.md` | Strong safety/runtime signal but zero goal-reaching on the paper surface. | Recover non-trivial goal-reaching while preserving the low-collision, low-near-miss behavior that makes the planner interesting. | Show a concrete commitment/progress fix on the verified-simple gate before spending more full-benchmark time. |
| `gap_prediction` | `docs/context/issue_671_gap_prediction_benchmark.md` | Over-conservative veto behavior collapses success to zero. | Demonstrate that the veto/prediction interaction can support actual progress rather than only suppressing motion. | Bring a concrete fix for progress recovery, then rerun verified-simple and only escalate to the paper surface if success returns. |
| `sicnav` | `docs/context/issue_4870_sicnav_smoke.md`, `docs/context/issue_771_drmpscnav_assessment.md` | Open-source smoke PASSES: the wrapper drives the pinned upstream `sicnav.policy.campc.CollisionAvoidMPC` (CasADi + bundled IPOPT + python-RVO2; Acados/HSL intentionally absent) and returns valid `{v, omega}` actions. But the steady-state solve is ~197 / 506 / 1454 ms per `step()` for 2 / 5 / 10 neighbors — **2.0×–14.5× over the 100 ms step budget** and scaling with neighbor count, so this redistributable path is real-time campaign-infeasible. | Show the wrapper runs deterministically on the verified-simple subset without fallback-only behavior AND at a per-step solve cost compatible with the real-time step budget. | Do not rerun the uncompiled CasADi/IPOPT path. Revisit only with a compiled Acados + HSL build (or a horizon/solver change) that brings per-step solve under the 100 ms budget, and switch the crowd to ORCA pedestrians (the wrapper's social-force default is not what SICNav models). |
| `dr_mpc` | `docs/context/issue_771_drmpscnav_assessment.md` | External DR-MPC runtime is assessment-only and not yet benchmark-ready in-repo. | Show the wrapper can resolve the upstream contract and produce stable actions without fallback-only behavior. | Keep DR-MPC as a follow-up anchor until the benchmark wrapper is backed by a real upstream runtime. |

## Current status

- Keep all testing-only planners behind `allow_testing_algorithms: true`.
- Treat their existing benchmark notes as negative evidence, not missing paperwork.
- Do not remove the guard for documentation completeness alone.
