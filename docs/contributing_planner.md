# Contributing A Planner

This is the minimum path for adding a Robot SF planner without overclaiming its status. It is a
checklist for a first useful contribution, not a substitute for the detailed benchmark, policy
search, or artifact-provenance docs linked below.

Use this guide when a planner should become runnable through the repository, visible to reviewers,
and clear about whether it is smoke-only, diagnostic-only, or benchmark-ready.

## Start From The Right Entry Point

Choose the narrowest integration surface that matches the planner:

- General planner protocol: implement the `PlannerProtocol`, `ObservationContract`,
  `ActionContract`, and `PlannerMetadata` shape in
  [`robot_sf/baselines/interface.py`](../robot_sf/baselines/interface.py).
- Map-runner local adapter: start from
  [`docs/dev/planner_adapter_template.md`](./dev/planner_adapter_template.md). The current
  diagnostic reference is `robot_sf.planner.socnav.TrivialReferencePlannerAdapter`, exposed as
  `trivial_reference` with the `reference_adapter` alias.
- Learned or imported policy adapter: read the learned-policy eligibility and adapter notes before
  adding code:
  [`docs/context/policy_search/contracts/learned_local_policy_eligibility.md`](./context/policy_search/contracts/learned_local_policy_eligibility.md)
  and
  [`docs/context/issue_1618_learned_policy_adapter_interface.md`](./context/issue_1618_learned_policy_adapter_interface.md).

Do not add a new runner or CLI path unless the existing protocol or adapter route cannot express the
planner. If optional dependencies, checkpoints, or external runtimes are missing, the planner must
fail closed or report an explicit not-available/diagnostic status rather than silently falling back
to a different planner.

## Minimum Contribution Flow

1. Add the planner or adapter behind an existing public entry point.
   For map-benchmark adapters this usually means a planner module under `robot_sf/planner/`, a
   `_build_policy()` branch in [`robot_sf/benchmark/map_runner/map_runner.py`](../robot_sf/benchmark/map_runner/map_runner.py),
   and no new benchmark semantics beyond that dispatch.

2. Declare algorithm metadata and readiness.
   Update [`robot_sf/benchmark/algorithm_metadata.py`](../robot_sf/benchmark/algorithm_metadata.py)
   with category, policy semantics, planner id, observation contract, action contract, command
   bounds, reset convention, adapter/projection notes, and any required scenario inputs. Update
   [`robot_sf/benchmark/algorithm_readiness.py`](../robot_sf/benchmark/algorithm_readiness.py) with
   the canonical key, aliases, tier, note, and opt-in requirement.

3. Document the observation and action contract.
   State exactly what the planner consumes, what it ignores, and what it outputs. Use
   [`docs/dev/observation_contract.md`](./dev/observation_contract.md),
   [`docs/dev/holonomic_action_contract.md`](./dev/holonomic_action_contract.md), and
   [`docs/benchmark_observation_visibility.md`](./benchmark_observation_visibility.md) instead of
   rewriting the contracts.

4. Add a config-first invocation path.
   Put stable planner settings under `configs/algos/` or, for policy-search candidates, under
   `configs/policy_search/candidates/`. The command should identify the config path rather than
   depending on reviewer-only CLI flags.

5. Add smoke or contract coverage.
   At minimum, cover command shape/bounds, reset behavior, failure status, metadata emission, and
   map-runner dispatch. The reference adapter coverage lives in:
   `tests/test_socnav_planner_adapter.py`, `tests/benchmark/test_map_runner_utils.py`,
   `tests/benchmark/test_algorithm_metadata_contract.py`, and
   `tests/benchmark/test_algorithm_readiness_contract.py`.

6. Set the benchmark boundary before running comparisons.
   If the planner has benchmark-routed stages in
   [`docs/context/policy_search/candidate_registry.yaml`](./context/policy_search/candidate_registry.yaml),
   add `benchmark_track`. If it is intentionally not benchmark evidence, add an explicit
   `claim_scope: diagnostic_only` or `non_benchmark_boundary`. Keep the fail-closed fallback policy
   from [`docs/context/issue_691_benchmark_fallback_policy.md`](./context/issue_691_benchmark_fallback_policy.md)
   in force.

7. Add candidate-registry metadata when the row should be routed by policy search.
   Add `candidate_config_path`, family, status, training requirement, hypothesis, promotion gate,
   required stages, and claim boundary in the policy-search registry. The registry is for
   implemented or concrete runnable Robot SF candidates, not source-only ideas. Use
   [`docs/context/policy_search/README.md`](./context/policy_search/README.md),
   [`docs/context/policy_search/candidate_registry_summary.md`](./context/policy_search/candidate_registry_summary.md),
   and [`docs/planner_zoo/index.md`](./planner_zoo/index.md) for routing vocabulary.

8. Record artifact and provenance status.
   Learned checkpoints, normalizers, external source assets, and generated evidence need durable
   pointers before they support benchmark or paper-facing claims. Use
   [`docs/context/artifact_evidence_vocabulary.md`](./context/artifact_evidence_vocabulary.md) and
   [`docs/model_registry_publication.md`](./model_registry_publication.md) for those boundaries.

## Status Vocabulary

Use these terms consistently in issue text, docs, and PR validation:

- Smoke proof: a narrow execution or contract test showed the planner can run through the intended
  adapter/config path and produce valid bounded actions. This is wiring evidence, not a ranking.
- Diagnostic-only: the planner is useful for mechanism inspection, feasibility, or failure analysis,
  but its outputs are not benchmark-strength evidence. Registry rows should say so explicitly.
- Benchmark readiness: the planner has the required metadata, config, fail-closed behavior, scenario
  or benchmark-track identity, and reproducible benchmark evidence for the stated suite. A smoke run
  alone never promotes a planner to this status.

For broader benchmark interpretation, use
[`docs/benchmark_experimental_planners.md`](./benchmark_experimental_planners.md),
[`docs/benchmark_planner_family_coverage.md`](./benchmark_planner_family_coverage.md), and
[`docs/context/issue_1627_learned_policy_transfer_benchmark.md`](./context/issue_1627_learned_policy_transfer_benchmark.md).

## Minimal Example Path

New to the planner interface? Start with the runnable
[`37_custom_planner_protocol.py`](../examples/advanced/37_custom_planner_protocol.py) tutorial:
one deterministic stateless planner implementing `plan`, keyword-only seeded `reset`,
`diagnostics`, and idempotent `close` against `LocalPlannerProtocol`, with action validation,
a short headless episode, and a negative fixture. It is educational only — not a benchmark
reference and not a production recommendation.

For a diagnostic adapter contribution, compare against the existing reference adapter before adding
new surfaces:

```bash
uv run robot_sf_bench run \
  --matrix configs/scenarios/planner_sanity_matrix_v1.yaml \
  --out output/benchmarks/reference_adapter_smoke/episodes.jsonl \
  --algo reference_adapter \
  --repeats 1 \
  --horizon 300 \
  --workers 1 \
  --no-video \
  --benchmark-profile experimental \
  --algo-config configs/algos/reference_adapter.yaml
```

For a policy-search candidate example using an already available planner family, inspect
[`configs/policy_search/candidates/hybrid_rule_v3_fast_progress.yaml`](../configs/policy_search/candidates/hybrid_rule_v3_fast_progress.yaml)
and its registry row, then run the registered smoke path:

```bash
uv run python scripts/validation/run_policy_search_candidate.py \
  --candidate hybrid_rule_v3_fast_progress \
  --stage smoke
```

That example is useful because it shows the config-first candidate path and registry routing. It
does not mean a new variant is benchmark-ready; promotion still depends on the declared stages,
benchmark-track metadata, and accepted evidence.

## Minimum Validation

Run the narrowest validation that matches the change. A typical new planner adapter should include:

```bash
uv run pytest \
  tests/test_socnav_planner_adapter.py \
  tests/benchmark/test_map_runner_utils.py \
  tests/benchmark/test_algorithm_metadata_contract.py \
  tests/benchmark/test_algorithm_readiness_contract.py \
  -q
uv run python scripts/validation/validate_policy_search_registry.py
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
```

Add a smoke command, such as the `reference_adapter` or policy-search candidate smoke above, before
claiming the planner runs in this repository. For docs-only guide updates, `git diff --check`, path
existence checks, and docs proof consistency are enough unless the guide changes a command contract.

## Hybrid feasibility diagnostics

The hybrid evaluator supports `debug_candidate_evaluator: true` (default: false).
Each decision then includes `candidate_evaluator_debug`: rejection counts grouped
by constraint and actual threshold, the highest-scoring feasible moving command,
and `FORCED` versus `PREFERRED` translation stops. A translation command with
absolute speed at most 1e-9 m/s is a stop; turning in place is included. Goal stops
are labeled `GOAL`. The proximity speed cap and conditional constraint thresholds
are also recorded. These diagnostic fields are absent when the flag is absent.

For hybrid v4 with bound exact-map geometry and continuous static checking,
`physical_static_exclusion_enabled: true` (default: false) separates physical
body exclusion from the soft static clearance preference. Hard exclusion uses
the observed body radius; clearance preference uses the occupancy-grid estimate
of surface gap divided by `desired_static_clearance`, clamped to [0, 1]. The
coarse estimate affects preference, while exact geometry determines wall contact.
Every rollout interval uses the observed plant timestep, trapezoidal velocity,
and midpoint heading. The swept segment is expanded by an arc bound:
`arc_length * abs(turn_angle) / 8`, which is at least the circular-arc sagitta.
This also covers the first committed step. Unbound geometry and older variants
retain the existing exclusion policy.

`goal_next_validity_enabled: true` is a separate, default-false planner flag.
Use it with the default-false environment option `include_goal_next_valid: true`.
The sensor then emits `goal.next_valid` (float32 array of shape (1,), 0 for absent,
1 for present). The hybrid and its grid route guide use that bit; a legitimate
successor at world origin remains valid. The production flattened reader and
map observation bridge preserve the optional bit. Without the field, they retain legacy
selection. Physical exclusion alone does not change goal selection. Default
observations, observation spaces and frozen 0.0.8 policy remain unchanged.

The wall flag also checks the complete committed-step-plus-braking tail against
physical map geometry. Limited reverse from main is respected by this sweep.
The platform candidate-injection experiment was removed after round 3: with the
current-position braking cap enforced it adds no meaningful success benefit and
increases near misses. Historical measurements and the braking audit remain in
`docs/validation/hybdiag/`; they are not an adoption recommendation.

The 1.60 m threshold in these scenarios is a **candidate rejection radius applied
to constant-velocity predicted pedestrian positions at rollout endpoints**.
It is not a hard limit on executed separation. Plant/pedestrian evolution can
differ from prediction; report actual minimum separation and near misses rather
than inferring them from this threshold. The physical radius sum here is 1.40 m.
These experimental flags require paired collision, pedestrian-separation and
empty-world validation; none changes the release config.

The development-only driver uses the benchmark environment and action adapter:

```bash
uv run python -m scripts.validation.run_hybrid_feasibility_diagnostics \
  --scenarios francis2023_narrow_doorway_width_2p20 --seeds 1001 \
  --arms off static_only static_plus_goal_validity --workers 1 --output output/hybrid_feasibility
```

It validates resolved seeds before environment creation and permits only
1001-1030. Local runs permit at most two workers; Slurm runs permit at most 32.
The default comparison budget is 600 steps at 0.1 s, explicitly independent of
release authored budgets. `--empty` uses the existing actor-removal hook.
Summary JSON and compressed per-step JSONL retain forced/preferred decisions,
actual-plant stopped-time fraction (speed <= 0.05 m/s), time without a feasible
moving command, longest stationary interval, and diagnostic freezing (stationary
for >10 s, or net displacement <0.5 m across a window longer than 10 s).
The three comparison arms are `off`, `static_only` (only
`physical_static_exclusion_enabled`), and `static_plus_goal_validity` (also
`goal_next_validity_enabled` and `include_goal_next_valid`). Native pedestrian minimum center separation
and near-miss steps/events are retained. A near miss has a surface gap in
`[0, 0.50)` m; an event starts when the per-step near-miss indicator turns on.

ORCA output separately reports the first rejected *executed* endpoint segment
and the first rejected hypothetical constant-command horizon. The latter is not
an executed-path counterexample. Dynamic segment replay uses the hybrid's
constant-velocity pedestrian prediction. Feasible unsampled forward probes
explain candidate omissions without entering the action-selection set. Outputs
are diagnostic-only development evidence and do not alter release inputs.
