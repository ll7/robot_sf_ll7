# Issue #2930 External Benchmark Positioning (2026-06-19)

Status: positioning note (`proposal`), not benchmark evidence.

Related:
- GitHub issue: https://github.com/ll7/robot_sf_ll7/issues/2930
- Metric correspondence input:
  [issue_2928_socnavbench_hunavsim_metric_correspondence.md](issue_2928_socnavbench_hunavsim_metric_correspondence.md)
- Prior concept mapping:
  [issue_2459_socnavbench_hunavsim_mapping.md](issue_2459_socnavbench_hunavsim_mapping.md)
- Benchmark fallback policy:
  [issue_691_benchmark_fallback_policy.md](issue_691_benchmark_fallback_policy.md)
- Artifact vocabulary:
  [artifact_evidence_vocabulary.md](artifact_evidence_vocabulary.md)

## Claim Boundary

This note positions Robot SF relative to adjacent benchmark and simulator ecosystems. It does not
claim metric parity, simulator equivalence, planner-ranking transfer, CARLA replay parity, or
paper-grade evidence. Rows below are best read as a reviewer-facing map of where each tool is useful
and which evidence gates would be needed before stronger claims.

## Positioning Table

| System | Purpose | Fidelity | Actors | Metrics | Reproducibility | Evidence tier in this repo | Best use |
|---|---|---|---|---|---|---|---|
| Robot SF | Fast social-navigation simulation, falsification, AMV-aware diagnostics, and benchmark infrastructure. | 2D map and pedestrian-dynamics abstraction with explicit fail-closed benchmark policy. | AMV/robot ego, pedestrians, static geometry, optional planner adapters, and scenario perturbations. | Success, collision, near-miss, efficiency, comfort, AMV trace/mechanism diagnostics, and selected SocNavBench-derived path metrics. | Config-first scenarios, tracked context notes, durable evidence manifests, and local `output/` treated as disposable unless promoted. | Native Robot SF evidence ranges from diagnostic-only to nominal benchmark evidence depending on the specific command, config, seed policy, and artifact provenance. | Rapid hypothesis falsification, mechanism-level failure diagnosis, AMV actuation studies, and evidence-bound planner comparison inside the Robot SF contract. |
| SocNavBench | Grounded social-navigation benchmark framework and planner/cost ecosystem. | Dataset-backed social-navigation scenarios and upstream objective/cost definitions; Robot SF has only a vendored/adapted subset. | Robot, pedestrians, maps/datasets, and SocNavBench planners/costs. | Path-length and personal-space-style objective families; Robot SF exposes only selected source-parallel path metrics with no outcome-parity proof. | Depends on external assets and upstream runtime assumptions; in-repo use remains gated by asset and adapter readiness. | Adapter/degraded rows are not benchmark-success evidence unless the exact benchmark contract is satisfied. | External social-navigation reference point and terminology source; useful for mapping and future re-entry gates, not as an automatic Robot SF equivalence claim. |
| HuNavSim | ROS 2 human-navigation simulator for human-aware robot navigation experiments. | ROS/Gazebo-style human-navigation simulation with richer human-behavior focus than Robot SF currently imports. | Robot and simulated humans with behavior-modeling emphasis. | Human-aware navigation outcomes and behavior-evaluation concepts described externally; no in-repo HuNavSim metrics, fixtures, or adapter path exist today. | Reproducibility depends on a separate external simulator stack; Robot SF has no hydrated HuNavSim artifact path. | Literature-positioning only in this repository. | Conceptual comparison for human-behavior modeling and future adapter scoping. |
| CARLA | High-fidelity 3D simulator for vehicle/world replay and transfer probes. | 3D simulator, physics, sensors, maps, and actor runtime; Robot SF treats CARLA as an optional external replay/smoke boundary. | Vehicles/robots, pedestrians/actors, static world geometry, sensors when configured. | Replay availability, spawn/alignment, and parity-oriented diagnostics when live CARLA proof exists; failed/not-available rows remain explicit. | Requires pinned CARLA runtime, Docker/Python client compatibility, and live replay proof on capable hosts. | Setup or replay evidence only for the exact command and host/runtime boundary recorded; failed CARLA parity remains `not_available` or `failed`. | Transfer-boundary probes and optional replay diagnostics, not routine fast benchmark iteration. |
| Generic Gymnasium RL envs | Standard RL training/evaluation interface. | Interface fidelity varies by environment; Gymnasium itself does not define social-navigation realism. | Agent plus task-specific observations/actions/rewards. | Reward, return, episode termination, and task-local metrics chosen by the environment. | Strong API convention, but benchmark meaning depends on environment-specific configs and seeds. | Interface support only unless backed by Robot SF benchmark evidence. | Training-loop integration, smoke tests, and algorithm experimentation once the environment contract is explicit. |

## Robot SF Niche

Robot SF is best described as **fast, evidence-bound, falsification-oriented, AMV-aware benchmark
infrastructure**:

- fast enough for local diagnostics, issue-scoped smoke tests, and bounded planner comparisons;
- evidence-bound because benchmark claims must name the command, config, seed policy, execution
  mode, artifact provenance, and fallback/degraded exclusions;
- falsification-oriented because diagnostic and negative results are first-class when they sharpen a
  claim boundary;
- AMV-aware because actuation, trace, and mechanism notes can separate planner logic from vehicle
  feasibility and progress failure modes.

The closest external positioning is not "Robot SF replaces SocNavBench, HuNavSim, CARLA, or
Gymnasium." A safer statement is that Robot SF supplies a repo-native loop for fast social-navigation
experiments with explicit evidence gates. SocNavBench and HuNavSim provide external social-navigation
reference points, CARLA provides optional higher-fidelity replay pressure, and Gymnasium provides the
RL API convention.

## Non-Claims And Follow-Up Gates

- SocNavBench-derived metrics in Robot SF do not establish outcome parity with upstream
  SocNavBench; see the #2928 correspondence note for the current metric boundary.
- HuNavSim remains an external literature and design reference until an adapter, fixture, or
  reproducible setup path exists.
- CARLA parity remains separate from social-navigation framework positioning and requires live
  replay/alignment proof before any parity claim.
- Generic Gymnasium compatibility does not by itself prove social-navigation benchmark quality.

Any future claim stronger than this positioning note should point to executable evidence and classify
rows using `native`, `adapter`, `fallback`, `degraded`, `failed`, or `not_available` semantics from
the benchmark fallback policy.

## Validation

Cheap docs validation for this note should verify the linked context paths and run diff-level docs
proof consistency. No benchmark, planner, metric, or simulator behavior changes are introduced here.
