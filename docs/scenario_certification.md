# Scenario Certification

[← Back to Documentation Index](./README.md)

`scenario_cert.v1` is the first machine-readable certification surface for generated and curated
scenario manifests. It is intentionally conservative: malformed scenarios, missing inflated paths,
kinodynamic violations, and clearly blocked dynamic setups are excluded before they can support
benchmark claims.

For authored scenario intent before execution, use
[`scenario_contract.v1`](./scenario_contracts.md). A scenario contract records ODD assumptions,
actor models, invariants, observables, termination semantics, and provenance; it does not replace
the fail-closed feasibility and eligibility checks described here.

## Contract

The public schema lives at
[`robot_sf/benchmark/schemas/scenario_cert.v1.json`](../robot_sf/benchmark/schemas/scenario_cert.v1.json).
Each certificate includes:

- `schema_version`: always `scenario_cert.v1`.
- `scenario_id` and `source`: the scenario name/id and manifest or programmatic source.
- `classification`: one of `valid`, `invalid`, `geometrically_infeasible`,
  `kinodynamically_infeasible`, `dynamically_overconstrained`, `knife_edge`, or
  `hard_but_solvable`.
- `benchmark_eligibility`: `eligible`, `stress_only`, or `excluded`.
- `checks`: deterministic geometry, route, planner, kinodynamic, and dynamic checks.
- `route_certificates`: per-route evidence for every applicable robot route.
- `evidence`: optional scenario metadata and scenario-difficulty provenance.

Benchmark inclusion policy:

- `valid` and `hard_but_solvable` are benchmark-eligible.
- `knife_edge` is stress-only and should not be promoted as headline benchmark evidence without
  an explicit benchmark issue.
- `invalid`, `geometrically_infeasible`, `kinodynamically_infeasible`, and
  `dynamically_overconstrained` are excluded.

## Checks

The v1 certifier uses the repository scenario loader, so `map_id`, `map_file`, route overrides,
single-pedestrian overrides, and robot config parsing follow the same path as training and
benchmark tools.

Geometry checks:

- finite start and goal coordinates within map bounds,
- start/goal not inside static obstacles,
- inflated global path existence using the classic A* planner with no inflation fallback,
- **continuous swept-envelope validation of the planned A* path** (issue #6139): after A*
  returns a collision-free grid path, the certifier re-validates the planned polyline
  against the same parsed obstacle geometry and robot envelope the simulator uses. A
  grid-inflated A* path can still cut a diagonal corner that the continuous robot disc
  cannot pass, so the certifier measures the full-polyline clearance
  (``LineString(path).distance(obstacles) - robot_radius``) and fails closed as
  ``geometrically_infeasible`` when the swept envelope clips an obstacle corner
  (negative clearance), when a planned vertex is clipped, or when the geometry is
  invalid, empty, or otherwise unverifiable. The occupancy-grid/A* verdict
  (``inflated_collision_free_path``) and the continuous swept-disc verdict
  (``swept_envelope``), and an executable runtime collision verdict
  (``simulator_obstacle_collision``) are recorded together for the discriminating
  check. The runtime verdict replays conservative path samples through
  ``ContinuousOccupancy.is_obstacle_collision`` using the parsed simulator obstacle
  segments; it is distinct from the exact swept-envelope calculation. Every accepted
  path keeps finite non-negative full-polyline clearance for the declared radius.
- shortest inflated path length,
- path length ratio against direct start-goal distance,
- authored route static clearance against obstacles.

Kinodynamic checks:

- differential-drive and holonomic robots are considered command-feasible when their existing
  settings validate, because they can rotate in place,
- bicycle-drive routes are excluded when the authored route contains a turn tighter than the
  configured `wheelbase / tan(max_steer)` limit.

Dynamic checks:

- moving single pedestrians are optional hardness evidence,
- static single pedestrians whose start position blocks the inflated robot route corridor classify
  the scenario as `dynamically_overconstrained`.

Infrastructure checks:

- map definitions may include optional `infrastructure_zones` metadata for public-space semantics
  such as `pedestrian_only`, `stairs`, `pedestrian_exit`, `signalized_crossing_zone`, or
  `shared_space_lane`,
- each infrastructure zone names its polygon vertices and `allowed_actor_types`,
- robot/AMV routes that intersect a zone whose allowed actors do not include `amv`, `robot`,
  `vehicle`, or `all` fail closed as `invalid`,
- these zones are certification-only metadata and do not change simulator physics, obstacle
  collision, route planning, or benchmark execution by themselves.

Scenario difficulty and planner residual analysis from
[`docs/context/issue_692_scenario_difficulty_analysis.md`](./context/issue_692_scenario_difficulty_analysis.md)
is linked as diagnostic evidence only. It is not treated as a replacement for validity or
feasibility checks.

## CLI

Generate a batch JSON document:

```bash
uv run python scripts/tools/certify_scenarios.py \
  configs/scenarios/sets/atomic_navigation_validation_fixtures_v1.yaml \
  --output output/scenario_cert/atomic_validation.json
```

Generate one JSON object per line:

```bash
uv run python scripts/tools/certify_scenarios.py \
  configs/scenarios/sets/atomic_navigation_validation_fixtures_v1.yaml \
  --jsonl
```

Use `--scenario-id <name>` to certify one scenario. Use `--fail-on-excluded` in gates that should
exit non-zero when any scenario is excluded.

## Library API

```python
from pathlib import Path

from robot_sf.scenario_certification import certify_scenario_file, certificate_to_dict

certificates = certify_scenario_file(
    Path("configs/scenarios/sets/atomic_navigation_validation_fixtures_v1.yaml")
)
payload = [certificate_to_dict(certificate) for certificate in certificates]
```

Programmatic tests can call `certify_map_definition(...)` directly with a `MapDefinition`.

## Adversarial scenario verdicts

Falsification callers can combine the serialized certificate with the existing feasibility
oracle, predicate contract, and named execution/replay records through
`robot_sf.adversarial.classify_scenario_admissibility(...)`. Pass the candidate's canonical
`scenario_artifact_path` with the evidence. The output contract is
[`scenario_admissibility.v1`](../robot_sf/benchmark/schemas/scenario_admissibility.v1.json),
and `partition_candidates_by_admissibility(...)` retains cases by verdict for search
stratification.

The adapter has five outcomes: `structurally_invalid`,
`geometric_or_kinodynamic_impossibility`, `admissible_feasibility_unknown`,
`empirically_feasible`, and `planner_specific_failure`. It rejects only explicit structural or
geometry/kinodynamic exclusions. An oracle exclusion is scoped to its recorded robot envelope;
the envelope and certificate assumptions remain attached to the verdict. A
`dynamically_overconstrained` certificate, a blocked or truncated oracle, missing provenance, or
conflicting evidence remains `admissible_feasibility_unknown`. Since `scenario_cert.v1` reports the
highest-severity route at the scenario level, geometry or kinematic exclusion requires every
applicable route certificate to support the same excluded classification; a different or usable
route keeps the whole case unknown. Each route's reason must also match check data from the
canonical certifier: a blocked inflated path, a validated swept-envelope clearance failure, a
validated runtime collision with the same first-collision sample, or a bicycle turning-radius or
steering-limit failure with matching kinematic values. The scenario-level reason list must match
the route-level reasons. Labels without those checks, empty reasons, or contradictory values stay
`admissible_feasibility_unknown`; an unsupported robot model does not establish kinodynamic
impossibility.

Execution inputs are normalized records with `case_id`, `scenario_id`, `scenario_variant`,
`planner_id`, `run_status`, the explicit boolean `fallback_or_degraded`, `route_complete`, `seed`,
`horizon_steps`, SHA-256 hashes for the scenario, robot model, simulator config, planner config, and
environment, plus `planner_checkpoint_sha256`, `source_commit`, and `evidence_ref`. The fallback
boolean must be false; the adapter also applies the canonical runtime fallback/degraded detector to
the full normalized record, so nested fallback flags, unavailable/fallback/degraded statuses, and
positive fallback counters cannot establish an outcome. Missing or malformed fallback state stays
unknown. Callers derive this summary from the complete canonical runtime metadata rather than
guessing from the episode's terminal status. The checkpoint field is either a SHA-256 hash or the
explicit value `not_applicable` for a known checkpoint-free planner; missing checkpoint provenance
stays unknown. Only `scenario_variant: original`, `run_status: ok`, and non-fallback/non-degraded
records can establish an outcome. Replay records additionally require
`determinism_check_status: pass` and `resimulated: true`; callers adapt canonical episode rows and
the existing replay provenance sidecar into this input shape. Incomplete records stay visible in
`evidence` and do not establish feasibility.

Artifact provenance is bound across evidence sources: the certificate `source` and oracle
`scenario_manifest` references must resolve to bytes with the same SHA-256 as
`scenario_artifact_path`, and each execution's `scenario_sha256` must equal that digest. A
missing or unavailable canonical artifact, an unresolvable source reference, or any mismatch
leaves that evidence unusable for rejection or feasibility classification while preserving the
captured input and reason code. For multi-scenario manifests, callers should pass the exact
scenario artifact used for the named case and adapt evidence hashes to those same bytes; a shared
scenario ID alone does not establish artifact identity.

An observed reference or replay completion is empirical evidence for that named case and run, not
a proof that every planner can solve it. Replay counts only after simulator resimulation with a
passing determinism check. A target-planner failure alone does not establish scenario
infeasibility; `planner_specific_failure` requires a completed reference run and an incomplete
target run bound to the same scenario, seed, horizon, source revision, and configuration hashes,
plus a deterministic replay by the target planner that reproduces the incomplete route under the
same scenario bindings and matching planner-config and checkpoint hashes as the target run. Missing,
mismatched, or successful replay leaves planner-specific failure attribution unconfirmed. A valid
completed reference run still establishes empirical feasibility for that same named scenario, and
the target's `route_incomplete` outcome remains separate from that feasibility verdict. This helper
does not change benchmark denominators or establish real-world safety.

## Limits

`scenario_cert.v1` is a first-pass fail-closed contract, not a high-budget oracle planner. It does
not generate adversarial scenarios, promote stress cases into headline benchmarks, or prove that a
planner will solve a valid scenario. It only certifies that the scenario geometry and currently
exposed robot/dynamic constraints are not malformed or impossible under the v1 checks.

`scenario_contract.v1` may explain what a scenario is meant to exercise, but a valid intent
contract does not make an excluded or uncertified scenario benchmark evidence.

For h500 interpretation, layer planner-failure classification on top of certification rather than
using h500 failures as certification evidence by themselves. Excluded or unresolved-certification
scenarios cannot support planner-failure attribution. Eligible and `hard_but_solvable` scenarios may
support planner follow-ups when the same mechanism recurs across seeds or planners. See
`docs/context/issue_1056_h500_failure_classification.md` for the current h500 classification
vocabulary.
