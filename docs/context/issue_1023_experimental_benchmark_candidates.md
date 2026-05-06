# Issue 1023 Experimental Benchmark Candidates

Date: 2026-05-06

Related:

- Upstream issue: `ll7/robot_sf_ll7#1023`
- Long-horizon benchmark config:
  `configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500.yaml`
- Scenario horizon schedule: `configs/policy_search/scenario_horizons_h500.yaml`
- Policy-search portfolio:
  `docs/context/policy_search/portfolio_overview_2026-05-05.md`
- Candidate-augmented preflight evidence:
  `docs/context/evidence/issue_1023_candidate_augmented_preflight_2026-05-06/`
- Candidate-augmented local full evidence:
  `docs/context/evidence/issue_1023_candidate_augmented_local_full_2026-05-06/`
- Candidate configs:
  `configs/policy_search/candidates/scenario_adaptive_hybrid_orca_v1.yaml`
  and `configs/policy_search/candidates/hybrid_rule_v3_fast_progress_static_escape.yaml`
- Benchmark fallback policy: `docs/context/issue_691_benchmark_fallback_policy.md`

## Decision

Add `scenario_adaptive_hybrid_orca_v1` and
`hybrid_rule_v3_fast_progress_static_escape` to the h500 scenario-horizon benchmark collection as
experimental candidates. They are not baseline-ready core planners and should not be used as
headline paper claims. The 2026-05-06 candidate-augmented local full run proves they execute
end-to-end in the benchmark path, but it also leaves release-blocking caveats: SNQI contract status
`fail` under warn enforcement, fixed-vs-scenario drift versus the May 4 reference, and experimental
planner provenance.

The maintainer request spelled `hybrid_rule_v3_fast_progress_static_excape`; the tracked candidate
is `hybrid_rule_v3_fast_progress_static_escape`.

## Candidate: scenario_adaptive_hybrid_orca_v1

This planner is a scenario-adaptive classical selector. Its default runtime policy is the
`hybrid_rule_local_planner` based on `configs/algos/hybrid_rule_v3_teb_like_rollout.yaml`, with a
faster progress envelope, static-clearance escape, and static recentering enabled. For
`francis2023_perpendicular_traffic`, it raises the slow-motion allowance and static escape speed.
For `francis2023_leave_group`, it switches the algorithm family to tuned ORCA using
`configs/algos/issue707_orca_tuned.yaml` with narrower ORCA timing, obstacle margin, symmetry, and
head-on bias parameters.

The design intent is pragmatic rather than elegant: keep the strong route-guided hybrid behavior
for most scenarios, but use ORCA where the policy-search evidence indicated that a tuned reciprocal
velocity-obstacle planner resolves the leave-group deadlock better than the hybrid rollout policy.
This is also the main reason to keep it experimental. A scenario-explicit selector can encode
knowledge of the benchmark distribution, so it needs separate overfitting scrutiny before it is
treated as a general planner.

Policy-search inclusion evidence: 144 episodes, success `0.9097`, collision `0.0208`, near miss
`0.4236`. The candidate-augmented local full benchmark then ran 144 benchmark episodes with
success `0.9097`, collisions `0.0278`, near misses `18.7447`, SNQI `-0.0795`, and runtime
`570.8285` seconds. The report records strong route completion and low collision relative to weak
baselines, with remaining failures dominated by long-horizon exposure and hard scenarios. This
evidence justifies inclusion as a challenger row, not promotion to the baseline collection.

## Candidate: hybrid_rule_v3_fast_progress_static_escape

This planner keeps a single `hybrid_rule_local_planner` family across the matrix. It starts from the
route-guided v3 hybrid rollout config and raises the speed and acceleration envelope to improve
long-route progress. It enables static-clearance escape, static recentering, and static corridor
transit so the planner can recover from local static-clearance stalls without disabling the hard
static-collision gate. It also applies a Francis perpendicular-traffic override for slower,
controlled escape motion.

The underlying planner is a deterministic local policy rather than a learned policy. It scores
short-horizon rollout candidates against route progress, static clearance, dynamic safety, and
recovery terms. The static escape and corridor-transit pieces exist because several h500 failures
were not social-navigation failures in the strict sense; the robot was locally stalled near static
geometry while otherwise safe. These additions try to convert those stalls into slow progress while
preserving fail-closed static collision rejection.

Policy-search inclusion evidence: 144 episodes, success `0.9028`, collision `0.0208`, near miss
`0.4236`. The candidate-augmented local full benchmark then ran 144 benchmark episodes with
success `0.9028`, collisions `0.0278`, near misses `20.7778`, SNQI `-0.0874`, and runtime
`576.6053` seconds. It is slightly behind `scenario_adaptive_hybrid_orca_v1` on success but avoids
a scenario-level algorithm switch, making it a cleaner non-learning challenger row. Its remaining
failure taxonomy still includes low-progress timeouts, static collisions, and intrusive near
misses, so it is not a solved planner.

## Benchmark Integration Note

The camera-ready map runner now resolves policy-search candidate manifests directly. During a
benchmark episode it merges `base_config_path` with `params`, applies `scenario_overrides`, and
honors `scenario_algo_overrides`. This matters for `scenario_adaptive_hybrid_orca_v1`; without that
support, the benchmark would list the candidate but would not faithfully execute its ORCA
leave-group override.

The long-horizon benchmark config keeps these rows under `planner_group: experimental` and
`benchmark_profile: experimental`. The 2026-05-06 candidate-augmented preflight proves the
9-planner matrix loads and reports the two candidate rows. The same-date local full campaign proves
the two rows run end-to-end with no failed, unavailable, fallback, or degraded planner
classification.

Do not publish a release tag from this run as-is. The campaign finished successfully, but SNQI
contract status is `fail` with warn enforcement, and the analyzer reports a small SNQI mean
mismatch for `scenario_adaptive_hybrid_orca_v1` between summary row and episode recomputation.
