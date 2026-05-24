# Issue #1457 Adversarial Map And Start-State Generation Protocol (2026-05-23)

Date: 2026-05-23

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1457>

Predecessor notes:

- [Issue #1433 Bounded Adversarial Edge-Case Scenario Search Design](issue_1433_adversarial_edge_case_search_design.md)
- [Issue #1237 Adversarial Failure Archive](issue_1237_adversarial_failure_archive.md)
- [Artifact Evidence Vocabulary](artifact_evidence_vocabulary.md)

Evidence:

- [issue_1457_adversarial_generation_smoke_summary.json](evidence/issue_1457_adversarial_generation_protocol_2026-05-23/issue_1457_adversarial_generation_smoke_summary.json)

## Goal

Issue #1457 asks how Robot SF should use generative methods, controlled map mutations, and
start-state or environment modifications to create adversarial planner stress tests without
weakening reproducibility or benchmark provenance.

The v1 recommendation is to start with **bounded start-state and route search on existing durable
maps**, then add constrained map mutation only after the validation and promotion boundary is
reviewed. This keeps the first slice executable today while preserving the distinction between
development stress tests and certified benchmark evidence.

## Compared Approaches

### Start-State And Route Search First

This approach keeps map geometry fixed and searches over robot route overrides, pedestrian route
overrides, route pairings, start/goal placements, timing, and density parameters. It reuses existing
scenario manifests, SVG maps, route planners, and feasibility checks.

Recommendation: **use this as the first implementation slice**. It has the lowest provenance risk,
can fail closed with current planner/map validation, and already has an executable anchor in
`scripts/tools/generate_adversarial_routes.py`.

### Constrained Map Mutation

This approach starts from existing SVG maps and applies bounded edits such as narrowing a corridor,
moving an obstacle, adding a blocker, or reshaping a crossing zone. It can expose geometry-sensitive
planner failures but requires a stronger validator than route search: SVG parseability, polygon
validity, minimum clearance, route reachability, semantic map metadata, and source-map lineage all
need to pass before any simulation run is considered valid.

Recommendation: defer to v2. Use it only after a mutation manifest schema and map validator can
reject invalid geometry before planner evaluation.

### Synthetic Scenario-Generator Extension

This approach adds adversarial parameter families to `robot_sf/benchmark/scenario_generator.py` or a
nearby generator, then emits explicit scenario manifests for bounded sweeps. It is useful when the
search dimensions are mostly scenario-level fields such as density, start delay, speed, and route
assignment.

Recommendation: useful after the route-search slice. Keep generated manifests in a stress-test
namespace and require review before promoting any case into `configs/scenarios/`.

### Black-Box Optimizer

This approach uses evolutionary search, Bayesian optimization, or another black-box optimizer over
the same validated parameters. It can find harder cases faster than grid/random search, but it also
increases overfitting risk if the objective is too planner-specific.

Recommendation: acceptable as the search engine for a bounded parameter space, not as the boundary
definition. Every candidate still needs the same deterministic seed, source config, validation
status, and objective-component record.

### LLM-Assisted Proposal With Validator-Controlled Execution

This approach uses a generative model to propose map/scenario variants, then accepts only variants
that compile to explicit YAML/SVG and pass repository validators. It can help brainstorm unusual
cases, but hidden prompt/model dependencies are a provenance risk.

Recommendation: exploratory only. Do not use LLM proposals as benchmark-strength evidence unless
the accepted artifact is fully explicit, deterministic, and validator-approved.

## Valid Versus Invalid Cases

An adversarial case is valid only when all of the following hold:

- The source map/scenario is durable and repo-addressable.
- Generated routes or start/goal pairs are inside map bounds and pass planner point validation.
- Required routes are available with native planner execution and no inflation fallback unless the
  run is explicitly scoped to measure that fallback.
- The candidate records the source scenario, source map, generator/config version, seed, objective,
  mutation/search parameters, planner/config, and objective components.
- The execution mode is clearly labeled as `development_stress_test`; generated cases are not mixed
  into nominal benchmark aggregate rows.

Candidates are rejected fail-closed when any of the following occur:

- Geometry is unparsable, self-intersecting, outside map bounds, below minimum clearance, or missing
  required semantic zones.
- Start or goal points are invalid, degenerate, unreachable, or require unapproved inflation
  fallback.
- Route generation fails, route availability is missing, or the planner cannot run natively.
- Simulation reset/step raises an exception. This is a `simulation_error`, not archive-eligible
  failure evidence.
- Planner execution is fallback or degraded outside an issue that explicitly measures fallback mode.
- Objective score is missing, invalid, or computed from a rejected candidate.

## V1 Prototype Command

The current executable anchor is:

```bash
uv run python scripts/tools/generate_adversarial_routes.py \
  --config configs/adversarial_routes/default.yaml
```

Run context:

- Commit: `4196918c271f501ed89740a8661b2e52d422d831`
- Scenario: `configs/scenarios/classic_interactions.yaml`
- Scenario id: `classic_head_on_corridor_low`
- Source map: `maps/svg_maps/classic_head_on_corridor.svg`
- Planner: `ClassicGlobalPlanner` with `algorithm: theta_star_v2`
- Optimizer: `optuna_tpe`
- Seed: `123`
- Trial count: `20`
- Feasibility filter: `true`
- Inflation fallback: `false`

Observed smoke result:

- Best score: `0.3474276984592932`
- Valid trials: `5 / 20`
- Failed trials: `15`
- Rejection counts: `invalid_start_or_goal: 15`
- Objective components:
  - `failure_proxy: 0.22222222222222224`
  - `delay_proxy: 0.38971079383717305`
  - `path_inefficiency: 0.0`
  - `near_miss_stress: 1.0`

Interpretation: the prototype proves that a bounded, seeded route/start-state search path runs
against an existing map and records useful feasibility diagnostics. It does **not** prove planner
robustness, benchmark coverage, or a certified adversarial case. The high rejection count is useful
evidence that v1 needs explicit feasibility rejection reporting and a minimum-valid-trial gate.

## Artifact Decision

The prototype wrote raw outputs under:

```text
configs/adversarial_routes/output/adversarial_routes/classic_head_on_corridor_low_20260523_130332_773186/
```

That directory remains ignored and worktree-local. The tracked evidence copy is a compact,
repo-relative summary JSON under `docs/context/evidence/`; it intentionally omits absolute local
paths, images, and raw route-override bundles.

No generated case is promoted into `configs/scenarios/` by this issue. Promotion would need a
separate review with explicit scenario certification and coverage/uncertainty context.

## Validation

Commands run locally:

```bash
uv sync --all-extras
uv run python scripts/tools/generate_adversarial_routes.py \
  --config configs/adversarial_routes/default.yaml
```

Additional validation run before handoff:

```bash
uv run pytest tests/test_adversarial_route_generation.py -q
scripts/dev/check_docs_proof_consistency_diff.sh
uv run python -m json.tool \
  docs/context/evidence/issue_1457_adversarial_generation_protocol_2026-05-23/issue_1457_adversarial_generation_smoke_summary.json \
  >/dev/null
git diff --check
```

Observed results:

```text
13 passed in 13.21s
OK docs/proof consistency check passed for 4 changed file(s).
json.tool exited 0.
git diff --check exited 0.
```

Planned PR freshness gate before opening the PR:

```bash
BASE_REF=origin/main scripts/dev/pr_ready_check.sh
```

## Follow-Up Boundary

The next implementation issue should add a first-class adversarial generation manifest only if it
can keep these guarantees:

- explicit source map/scenario lineage,
- deterministic seed and generator version,
- fail-closed validation for geometry and start/goal feasibility,
- native planner availability labels,
- objective components separated from nominal benchmark aggregates,
- small reviewable evidence copies instead of raw `output/` promotion.

Map mutation should remain deferred until the validator can reject invalid geometry before route or
planner scoring. LLM-assisted proposals should remain exploratory unless the final accepted variant
is fully explicit and validator-controlled.
