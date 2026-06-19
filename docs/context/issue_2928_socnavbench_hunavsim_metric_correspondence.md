# Issue #2928 SocNavBench / HuNavSim Metric Correspondence (2026-06-19)

Status: scoped interoperability analysis (`analysis_only`), not benchmark evidence.

Related:
- GitHub issue: https://github.com/ll7/robot_sf_ll7/issues/2928
- SocNavBench repo: https://github.com/CMU-TBD/SocNavBench
- SocNavBench paper: https://publications.ri.cmu.edu/socnavbench-a-grounded-simulation-testing-framework-for-evaluating-social-navigation
- HuNavSim repo: https://github.com/robotics-upo/hunav_sim
- HuNavSim paper: https://arxiv.org/pdf/2305.01303
- Prior mapping seed: [`issue_2459_socnavbench_hunavsim_mapping.md`](issue_2459_socnavbench_hunavsim_mapping.md)

## Scope

This note compares **Robot SF metric families** with **SocNavBench** and **HuNavSim** concepts only.
It does not claim simulator equivalence, planner transferability, or benchmark-score parity.
This output is intentionally designed as analysis input for issue #2930 and is not itself benchmark
evidence.

## Match classes

- **metric parity** — same named implementation family is present in Robot SF and can be traced to the
  vendored SocNavBench source subset (no outcome-level parity proof).
- **approximate analogue** — related intent, but no exact same definition/signature is implemented.
- **unsupported** — no in-repo metric, no fixture-level evidence, or framework-level gap.

## Correspondence matrix

| Metric family | Robot SF metric(s) | SocNavBench analogue | HuNavSim analogue | Match class |
|---|---|---|---|---|
| Safety | `collisions`, `near_misses`, `min_distance` | collision/clearance behavior is represented through collision thresholds and distance-based episode outcomes; not identical to upstream SocNavBench suites | outcome-level safety checks are discussed as part of robot-human social interaction in the HuNavSim paper | approximate analogue |
| Efficiency | `path_efficiency`, `time_to_goal_norm`, `avg_speed` | `path_length_ratio` / path-length-based tradeoff exists; `time_to_goal_norm` is not a direct SocNavBench term | throughput / speed behavior appears in HuNavSim planner evaluations but not as a directly named matching term | approximate analogue |
| Comfort | `comfort_exposure`, `jerk_mean`, `curvature_mean`, `energy`, force quantiles (`force_quantiles`) | SocNavBench report/usage materials mention comfort-adjacent notions (e.g., robot energy, smoothness, speed) and these are comparable as proxies | HuNavSim discusses human comfort-reactivity in robot-aware behavior simulations; no directly matching scalar in-Repo | approximate analogue |
| Personal-space | `min_distance`; no code-backed `personal_space_cost` metric in current Robot SF evidence | SocNavBench `personal_space_objective` is source-parallel conceptually, but no in-repo Robot SF metric currently exposes that name | HuNavSim models are intent-aligned but no documented equivalent objective metric in Robot SF evidence | approximate analogue; unsupported for `personal_space_cost` |
| Path deviation | `socnavbench_path_length`, `socnavbench_path_length_ratio`, `socnavbench_path_irregularity` | direct source name-level parity via vendored SocNavBench metric implementations | HuNavSim does not expose a dedicated path-deviation scalar in current scope | metric parity (for the named metrics), unsupported elsewhere |
| Social compliance | `comfort_exposure` is implemented; `pedestrian_deviation`, `flow_disruption`, `legibility_progress`, and `distributional_inconvenience` are contract-defined in `configs/benchmarks/social_compliance_metric_contract_v1.yaml` but not wired into `compute_all_metrics()` | no single SocNavBench scalar that is definitionally equivalent; only component costs with different composition semantics | no direct HuNavSim counterpart in currently published scope | unsupported (direct scalar), approximate through implemented comfort/efficiency proxies |
| Human-impact | `metrics.pedestrian_impact` (experimental), `force_exceed_events`, `comfort_exposure` | no direct Robot SF-to-SocNavBench human-impact aggregate contract in-repo; only source analogues are component-level force/cost fields | HuNavSim focus is behavior-modeling, not a directly portable human-impact aggregate metric | unsupported |

## Practical interpretation boundary

- `socnavbench_*` metrics are only as strong as the vendored implementation mapping and missing
  fixture-level parity proof.
- Comfort and compliance family rows are useful for **directional comparisons** only.
- HuNavSim rows are documentation-positioning points because this repository currently has no
  in-repo HuNavSim adapter, fixtures, or config path.
- `metrics.social_compliance` is a Robot SF contract with dedicated YAML (`configs/benchmarks/social_compliance_metric_contract_v1.yaml`);
  most named families in that contract are not implemented `compute_all_metrics()` outputs today, so
  this is not a cross-framework parity claim.

## Validation checks used for this note

```bash
test -f robot_sf/benchmark/metrics.py
test -f docs/context/issue_2459_socnavbench_hunavsim_mapping.md
test -f docs/context/issue_2397_socnavbench_control_status_2026-06-06.md
test -f third_party/socnavbench/UPSTREAM.md
```
