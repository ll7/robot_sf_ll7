# Issue #2468 Adversarial Scenario Generation Roadmap (2026-06-07)

Status: proposal/synthesis evidence only. This note does not train a generator, produce new
benchmark rows, or claim that generated scenarios are benchmark evidence.

Related surfaces:
- GitHub issue: https://github.com/ll7/robot_sf_ll7/issues/2468
- Existing adversarial protocol: `docs/context/issue_1457_adversarial_generation_protocol.md`
- Frozen adversarial manifest: `docs/context/issue_1500_adversarial_manifest.md`
- Adversarial failure archive contract: `docs/context/issue_1237_adversarial_failure_archive.md`
- RL adversarial pedestrian proposal: `docs/context/issue_2470_rl_adversarial_pedestrian_policy.md`
- Diffusion scenario-generation feasibility proposal:
  `docs/context/issue_2471_diffusion_scenario_generation_feasibility.md`
- Scenario certification: `docs/scenario_certification.md`
- Scenario zoo adversarial row: `docs/scenario_zoo/index.md`
- Proposal artifact: `docs/context/evidence/issue_2468/adversarial_generation_roadmap.yaml`

## Result

Robot SF already has an executable first adversarial lane: bounded search over explicit scenario
and route perturbations. The next research direction should not jump directly to RL or diffusion
training. It should first define a common `adversarial_scenario_manifest.v1` envelope that every
generator family must emit before any planner run is interpreted.

The recommended first spike is a manifest/schema plus fail-closed validator smoke. It should
normalize one existing search-generated candidate, one mock RL action-to-config fixture, and one
mock diffusion/proposal fixture into the same manifest envelope, then reject malformed and
degenerate variants before simulation. This is proposal/interface evidence, not benchmark evidence.

## Method Comparison

| Method | Feasibility in this repo | Data and compute need | Interpretability | Claim boundary |
|---|---|---|---|---|
| Bounded search / random / Optuna TPE / guided route search | Highest. Existing configs and scripts cover crossing/TTC and head-on route-search stress cases. | Low to moderate. Runs over explicit parameter bounds and planner rows. | High, because each candidate is an explicit manifest/config with seed, objective, and row classification. | Development stress-test evidence only until separately certified and promoted. Fallback/degraded rows are caveats, not successes. |
| RL adversary over pedestrian controls | Plausible after a deterministic action-to-config smoke. Existing `PedestrianEnv` is ego-pedestrian RL, not this NPC-adversary path. | Moderate to high. Needs reward design, frozen planner population, invalid-action penalties, and repeatable evaluation. | Medium. Actions are interpretable if constrained to high-level spawn/route/timing/speed fields; policy behavior still needs mechanism traces. | Proposal until the transformer, reward decomposition, fail-closed invalidity, and at least one smoke episode are proven. |
| Diffusion / generative model over scenario perturbation manifests | Later-stage only. The useful first step is the archive/manifest/certification interface, not model training. | High. Needs a durable failure/near-miss archive, split discipline, generator provenance, inference seeds, and certification. | Medium to low unless generated samples remain explicit manifests with source archive lineage and rejection reasons. | Proposal/interface evidence until archive lineage, determinism, certification, and planner smoke pass. No realism claim from generation alone. |
| Learned proposal model from failure/near-miss archives | Middle path after the archive is mature. Could rank or propose explicit perturbation manifests without full diffusion training. | Moderate. Needs labeled failure archive rows, features, train/validation split, and duplicate/degeneracy controls. | Medium to high if the model emits bounded manifest fields and records feature attribution or nearest-neighbor support. | Diagnostic proposal evidence until held-out archive behavior and deterministic manifest validation are demonstrated. |
| Hybrid LLM-to-structured-manifest assistant | Useful for brainstorming and triage only. Natural-language proposals must compile to explicit YAML/JSON and pass the same validators. | Low compute but high validation risk if unconstrained. | Medium after compilation; low before deterministic manifest validation. | Never benchmark evidence by itself. Accepted artifacts are only the validated structured manifests and their executable proof. |

## Common Manifest Direction

Each generator family should emit an `adversarial_scenario_manifest.v1` bundle with these fields:

| Field group | Minimum fields | Why it matters |
|---|---|---|
| Source and target | `source_map`, `source_scenario`, `planner_target`, `scenario_family` | Prevents orphaned generated cases and keeps planner-specific weaknesses explicit. |
| Generator provenance | `generator_family`, `generator_id`, `generator_commit`, `generator_seed`, `search_or_training_config` | Separates random/search/RL/diffusion/LLM-assisted rows and makes regeneration auditable. |
| Candidate controls | `pedestrians`, `robot_start`, `robot_goal`, route overrides, timing, speed, waits, phase offsets, objective weights | Gives every method the same bounded knobs and avoids hidden natural-language state. |
| Validity record | schema result, route reachability, map bounds, collision/overlap checks, temporal bounds, duplicate hash | Makes invalid and degenerate candidates fail before they become misleading runs. |
| Execution status | `not_run`, `valid_pending_run`, `valid_behavioral_failure`, `success`, `invalid_candidate`, `simulation_error`, `fallback`, `degraded`, `not_available` | Reuses the fail-closed row vocabulary from adversarial comparison work. |
| Evidence boundary | `development_stress_test`, `diagnostic_only`, `benchmark_candidate`, `promoted_benchmark_evidence` | Keeps exploratory stress cases from being mistaken for benchmark-strengthening evidence. |

## Controllable Parameters

The first shared control surface should stay close to fields that existing Robot SF scenario and
adversarial tooling can already materialize:

| Parameter | Initial bound or type | Notes |
|---|---|---|
| `scenario_seed` / `generator_seed` | integer, recorded per candidate | Required for deterministic replay and duplicate detection. |
| pedestrian start/spawn zone | discrete zone or explicit pose on walkable map | Must pass map bounds, obstacle clearance, and overlap checks. |
| pedestrian goal / route choice | discrete route id or explicit reachable goal | Must pass route reachability. |
| `start_delay_s` / phase offset | bounded float | Useful for TTC/crossing stress without mutating maps. |
| `speed_m_s` | bounded float | Reject zero-speed stalls and unrealistic extremes unless explicitly scoped. |
| wait/pause duration | bounded float | Reject temporal overflow and infinite blocking. |
| group size / role / cohesion mode | bounded enum and small count | v0 can stay single-NPC; multi-pedestrian variants need group consistency checks. |
| robot start/goal override | explicit pose or route id | Keep in v1 only when already supported by the scenario family and route validator. |
| objective weights | collision, near miss, timeout, low progress, comfort | Search and learned proposal methods should expose the objective, not hide it. |
| map geometry mutation | deferred | Only a v2 lane after stronger map mutation validators; v1 should prefer route/start-state and scenario-parameter perturbations. |

## Validity And Degeneracy Rules

A generated candidate is valid only when all of the following hold:

- The manifest parses against the explicit schema and names a durable source map/scenario.
- All start, goal, spawn, and route points are in-bounds, obstacle-clear, and reachable.
- Candidate timing, speed, wait, phase, density, and group fields are within configured bounds.
- The candidate records generator family, seed, objective, source config, and duplicate hash.
- The selected planner is available in native/adapter mode required by the row contract.
- Simulation setup completes without exception, fallback, or degraded substitution unless the row
  is explicitly scoped to measure fallback behavior.
- The row is classified fail-closed as `invalid_candidate`, `simulation_error`, `fallback`,
  `degraded`, or `not_available` when any required contract is missing.

Reject degeneracy before planner interpretation:

- zero-length or unreachable pedestrian/robot route;
- zero-speed or infinite-wait blocking that wins only by freezing the scene;
- start/goal overlap, obstacle overlap, or spawn inside the robot footprint;
- duplicate candidate under the same normalized control hash;
- candidates whose objective can be satisfied only by invalid geometry, missing planner support,
  or fallback execution;
- unconstrained natural-language scenario edits that have not compiled to a structured manifest.

## Recommended First Spike

Implement a common adversarial scenario manifest validator and fixture smoke:

1. Define `adversarial_scenario_manifest.v1` as a small schema or typed validator around the fields
   above.
2. Add three tiny fixtures: an existing bounded-search candidate, a mock RL action-to-config
   candidate following issue #2470, and a mock diffusion/proposal candidate following issue #2471.
3. Prove that valid fixtures pass schema/provenance/route-control checks without running a full
   benchmark.
4. Prove that malformed, unreachable, duplicate, and temporal-overflow fixtures fail closed with
   explicit rejection reasons.
5. Emit a compact validation summary that says `evidence_tier: proposal_interface_smoke`.

Stop rule: stop the spike after schema validation and fail-closed rejection proof. Do not train RL
or diffusion, do not publish planner rankings, and do not promote generated scenarios into
benchmark evidence until a follow-up issue supplies certification plus executable planner proof on
durable inputs.

## Remaining Risks

The largest risk is claim inflation: generated cases are tempting to treat as stress-test wins even
when they only expose invalid geometry, planner unavailability, or fallback behavior. The shared
manifest should make those non-evidence states first-class. A second risk is over-generalizing the
control space before the existing route/start/timing lane is fully exploited; map mutation and
learned generators should remain behind explicit validator gates.
