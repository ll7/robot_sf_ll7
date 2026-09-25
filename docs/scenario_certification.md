# Scenario Certification

[← Back to Documentation Index](./README.md)

`scenario_cert.v1` is the first machine-readable certification surface for generated and curated
scenario manifests. It is intentionally conservative: malformed scenarios, completed no-path
results, kinodynamic violations, and clearly blocked dynamic setups are excluded before they can
support benchmark claims. Planner errors remain unknown.

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
  `hard_but_solvable`, or `unknown` when certification cannot determine a result.
- `benchmark_eligibility`: `eligible`, `stress_only`, or `excluded`.
- `checks`: deterministic geometry, route, planner, kinodynamic, and dynamic checks.
- `route_certificates`: per-route evidence for every applicable robot route.
- `evidence`: optional scenario metadata and scenario-difficulty provenance. File-based
  certificates also record `source_artifact_sha256`, captured when the certifier reads the
  manifest, and `effective_input_sha256` plus `effective_input_identity_stable`. The effective
  identity covers included manifests and the selected scenario's resolved map and route-override
  files, including the resolved `map_id` path and parser selected by its suffix. If a scenario
  omits both `map_file` and `map_id`, the closure includes every SVG loaded into the default
  `MapDefinitionPool`. Certification and actor-free oracle execution compare the source hashes
  returned by the actual map and route parsers with this declared closure; a missing or different
  consumed input leaves runtime identity unavailable. Validation loading records each manifest
  digest from the same byte buffer it parsed. File-based certification and feasibility reports
  reuse that parsed validation report when computing input identity, so an include that changes
  and is restored during loading cannot be paired with a digest from a later parse. Explicit map
  definitions are cached by source-content digest and geometry contract, and SVG/serialized-map
  parsers consume the same immutable bytes used for that digest. A legacy single-row identity is
  available only when the full runtime input closure, including the default map pool when used, can
  be resolved and matched to the consumer snapshots.

Benchmark inclusion policy:

- `valid` and `hard_but_solvable` are benchmark-eligible.
- `knife_edge` is stress-only and should not be promoted as headline benchmark evidence without
  an explicit benchmark issue.
- `invalid`, `geometrically_infeasible`, `kinodynamically_infeasible`, and
  `dynamically_overconstrained` are excluded.
- `unknown` is retained for adversarial screening but marked `stress_only` for benchmark
  selection. A planner exception is recorded as `inflated_path_planner_error` and does not prove
  geometric impossibility; only a completed planner result with no path is a geometric exclusion.

## Checks

The v1 certifier uses the repository scenario loader, so `map_id`, `map_file`, route overrides,
single-pedestrian overrides, and robot config parsing follow the same path as training and
benchmark tools.

Geometry checks:

- finite start and goal coordinates within map bounds,
- start/goal not inside static obstacles,
- inflated global path existence using the classic A* planner with no inflation fallback,
- a planner exception is an unknown certification result, not evidence that the route is blocked,
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
stratification. Single-objective search rejects only explicit `search_disposition: reject` verdicts
before planner evaluation. QD callers can supply an `admissibility_precheck` to `run_map_elites`;
explicit exclusions are recorded and skipped before its evaluator, while unknown or missing verdicts
continue to evaluation. A verdict attached only to a completed QD evaluation prevents archive
admission but does not save that evaluation. Unknown, failed, or unavailable certification is not
rejected by this boundary and continues through the existing certification gate.

The adapter has five outcomes: `structurally_invalid`,
`geometric_or_kinodynamic_impossibility`, `admissible_feasibility_unknown`,
`empirically_feasible`, and `planner_specific_failure`. It rejects only explicit structural or
geometry/kinodynamic exclusions. Structural exclusions require corroborating producer checks: an
empty route inventory with the producer's no-route reason, a waypoint count below two with null
endpoints, or a non-finite endpoint serialized as null/malformed. Reason labels without matching
checks remain unknown. The current certificate does not include map bounds, obstacle geometry, or
enough infrastructure policy data to verify outside-map, obstacle, or infrastructure labels, so
those invalid certificates remain `admissible_feasibility_unknown`. An oracle exclusion is scoped to its recorded robot envelope;
the envelope and certificate assumptions remain attached to the verdict. A
`dynamically_overconstrained` certificate, a blocked or truncated oracle, missing provenance, or
conflicting evidence remains `admissible_feasibility_unknown`. A positive actor-free oracle result
requires a `passed` completion with route completion true, no blocker, explicit
`fallback_or_degraded: false`, a successful termination reason, positive completion steps within
the horizon, a matching horizon margin, raw observed completion true, and no rollout blocker or
fallback marker. The oracle report also records the source manifest digest at report production
and whether the source bytes remained stable during the report; the adapter requires both fields to
match the candidate artifact before accepting feasibility or an exclusion. Missing digest,
changed source bytes, missing fallback status, contradictory completion status/termination, or
inconsistent steps remain blocked or unknown. A geometric oracle exclusion
must carry the producer's no-path completion record; a contradictory positive completion cannot be
overridden by the geometric label. Since `scenario_cert.v1` reports the
highest-severity route at the scenario level, geometry or kinematic exclusion requires every
applicable route certificate to support the same excluded classification; a different or usable
route keeps the whole case unknown. Each route's reason must also match check data from the
canonical certifier: a blocked inflated path, a validated swept-envelope clearance failure, a
validated runtime collision with the same first-collision sample, or a bicycle turning-radius or
steering-limit failure with matching kinematic values. The scenario-level reason list must match
the route-level reasons. Labels without those checks, empty reasons, or contradictory values stay
`admissible_feasibility_unknown`; an unsupported robot model does not establish kinodynamic
impossibility.

Before planner evaluation, the target outcome is `not_evaluated`. After an evaluation attempt, the
search updates it to `route_completed`, `route_incomplete`, or `unavailable` and records the episode
JSONL digest, episode/scenario/planner IDs, seed, source commit, and terminal outcome. A completed or
incomplete outcome is recorded only when the episode matches the selected candidate and configured
planner config, its episode integrity and termination fields agree, runtime availability is native
and clean, and the scenario's manifest/map/route input identity still matches the pre-evaluation
snapshot. The default evaluator copies the selected planner config bytes to
`planner_config.snapshot.yaml` in the candidate bundle and runs the benchmark against that snapshot;
the target observation compares the episode's effective config with the captured selection and
records its source digest. Missing or inconsistent records remain `unavailable`. This observation
never changes the feasibility verdict by itself: target failure alone does not establish
infeasibility or `planner_specific_failure`.

Whenever a certificate, oracle, reference, target, or replay row is supplied, callers must also pass
the expected `scenario_id`. The adapter does not infer that binding from a certificate or execution
row; named evidence without it remains unknown.

Execution inputs are normalized records with `case_id`, `scenario_id`, `scenario_variant`,
`planner_id`, `run_status`, the explicit boolean `fallback_or_degraded`, `route_complete`, `seed`,
`horizon_steps`, SHA-256 hashes for the scenario, robot model, simulator config, planner config, and
environment, plus `planner_checkpoint_sha256`, `source_commit`, and `evidence_ref`. Every execution
record requires its source `episode_id` and a valid `source_episodes_jsonl_sha256`. `evidence_ref`
resolves to a local artifact: reference and target rows point to the canonical episode JSONL store,
while replay rows point to the canonical `replay_provenance.json` sidecar. Relative references
require `evidence_root`; absolute references are accepted when readable. The adapter reads and
hashes the bytes, validates the selected row against `episode.schema.v1`, requires one matching
episode ID, and compares its scenario, planner, seed, source revision, runtime status, and
route-completion outcome with the normalized row. A source run may also have an adjacent
`episodes.jsonl.provenance.json` manifest using `benchmark_result_provenance.v1`. When present, the
adapter binds its digest to the exact episode-store bytes and selected JSONL line, then reads the
run ID, numerical execution-context digest, scenario-matrix digest, simulator settings, and
planner-config input digest from that producer record. It rehashes the scenario and planner-config
bytes and checks the recorded horizon and row identity. For same-case comparison, it recomputes a
planner-independent digest from the episode's producer-recorded `scenario_params`, after verifying
the canonical 16-character map-runner config hash. The complete row config hash remains planner
specific and is compared between target and replay. An `execution_context` extension in an episode
row or replay result is not producer authority. If the producer manifest or a required field is
absent, the captured episode outcome remains visible, but it cannot establish candidate-case
feasibility unless the producer-bound scenario-matrix digest and planner-independent case identity
are both valid. Missing or unavailable scenario/case identity, and every explicit run-context
mismatch, leave the scenario verdict unknown. Missing execution-context or checkpoint dimensions
are recorded as unknown and block planner-specific attribution; a completed episode can still
support empirical feasibility for the named case when its producer-bound scenario and case
identity are valid.

Replay admission reads the sidecar's source episode-store path and digest, identity, determinism
status, and resimulation marker. Planner-specific attribution additionally requires the sidecar to
reference a separately hashed
[`target_planner_replay_result.v1`](../robot_sf/benchmark/schemas/target_planner_replay_result.v1.json)
artifact. That result must identify the target planner, source episode-store digest, replay outcome,
and the path and digest of the replay output's adjacent producer manifest. It must also reference a
separate canonical replay episode JSONL store by path, SHA-256, and episode ID. The adapter reads
that store, validates its selected row and byte-level line binding against the canonical episode
schema and producer manifest, and checks the output producer run ID differs from the target run ID.
Episode IDs may remain stable across replay; the distinct output artifact and producer run establish
that a separate run was recorded. Planner-config bytes, numerical execution context, simulator
settings, scenario-matrix bytes, planner-independent case identity, source revision, and horizon
must match the target run. The normalized robot-model, simulator-config, and environment digests
must also agree across the reference, target, and replay records. Caller-supplied context objects
in an episode row or result cannot substitute for producer metadata. A result
without this separate, producer-bound row, or with a runtime error, cannot establish
planner-specific failure. These checks establish internal byte and record consistency; the JSON
artifacts are not signed attestations and do not cryptographically prove which external process
produced them. A determinism pass from the existing episode visualization tool is diagnostic only:
it may compare final position and use a generic goal policy, so it does not establish that the
named target planner repeated its failure. Unreadable, malformed, stale, or conflicting artifacts
leave execution evidence unknown. The fallback
boolean must be false; the adapter also applies the canonical runtime fallback/degraded detector to
the full normalized record, so nested fallback flags, unavailable/fallback/degraded statuses, and
positive fallback counters cannot establish an outcome. Missing or malformed fallback state stays
unknown. Callers derive this summary from the complete canonical runtime metadata rather than
guessing from the episode's terminal status. The checkpoint field is either a SHA-256 hash or the
explicit value `not_applicable` for a known checkpoint-free planner. The current producer manifest
does not bind learned-checkpoint bytes, so hashed checkpoint inputs remain unknown for planner
attribution until a producer-owned checkpoint field is available. Only `scenario_variant: original`,
`run_status: ok`, and non-fallback/non-degraded
records can establish an outcome. Replay records additionally require
`determinism_check_status: pass` and `resimulated: true`; those fields alone cannot establish a
repeated target-planner outcome. For a target-planner failure replay, the result artifact's episode,
scenario, seed, source revision, planner/configuration, producer manifest, and terminal outcome must
match the target run and its source episode store. A replay from another episode or source artifact
leaves planner-specific failure attribution unconfirmed, even when its scenario, seed, and planner
configuration match. Incomplete records stay visible in `evidence` and do not establish feasibility.

Artifact provenance is bound across evidence sources: the certificate `source` and oracle
`scenario_manifest` references must resolve to bytes with the same SHA-256 as
`scenario_artifact_path`, and each execution's `scenario_sha256` must equal that digest. When a
scenario uses included manifests, a map file, or route overrides, certificates, oracle cells, and
normalized execution records must also carry the matching `effective_input_sha256`; producers must
record that the referenced bytes stayed stable while evidence was generated. The search checks the
same input closure again after planner evaluation before recording a target outcome. A
certificate's captured `evidence.source_artifact_sha256` must also equal the manifest digest, so
reusing a certificate after editing the source file at the same path leaves it unbound. A
missing or unavailable canonical artifact, an unresolvable source reference, or any mismatch
leaves that evidence unusable for rejection or feasibility classification while preserving the
captured input and reason code. For multi-scenario manifests, callers should pass the exact
scenario artifact used for the named case and adapt evidence hashes to those same bytes; a shared
scenario ID alone does not establish artifact identity.

An observed reference or replay completion is empirical evidence for that named case and run, not
a proof that every planner can solve it. Replay counts for planner attribution only when a separate
target-planner result records the route outcome and binds its producer-owned planner-config and
runtime metadata to a distinct run. A target-planner failure alone does not establish scenario
infeasibility; `planner_specific_failure` requires a completed reference run and an incomplete
target run bound to the same scenario, seed, horizon, source revision, effective input identity,
numerical execution context, and simulator settings, plus a target-planner replay result that
reproduces the incomplete route under the same scenario bindings and matching producer-owned
planner-config as the target run. The replay must also bind to the target episode ID and source
episode-store digest. Missing, mismatched, or successful replay leaves planner-specific failure
attribution unconfirmed. A valid completed
reference run still establishes empirical feasibility for that same named scenario, and the target's
`route_incomplete` outcome remains separate from that feasibility verdict. This helper does not
change benchmark denominators or establish real-world safety.

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
