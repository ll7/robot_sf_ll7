# Issue #6921 — Matched-budget reactive versus open-loop preflight

Status: diagnostic-only preflight; execution remains held.
Evidence grade: launch-packet and local capability evidence, not benchmark evidence.

## Plain-language summary

This revision turns the original declarative packet into a preflight contract
for two real repository seams:

- the open-loop arm uses `run_adversarial_search` with its production candidate
  evaluator, the nominal `social_force` policy, and the explicit
  `minimize_episode_min_robot_distance` episode-record objective;
- the reactive arm uses `FiniteGridSearchPolicy` and
  `BoundedResidualAdversary`.

Both arms emit the shared `matched_compute_trace.v1` accounting schema. The
trace also records `evidence_status` and `simulator_steps_source`, so the
local canary's synthetic episode records and one-step controller snapshot
cannot be mistaken for observed production simulator evidence. It fails
closed on missing, fallback, degraded, unavailable, or non-finite accounting.
Accepted, rejected, and invalid candidate counts are a disjoint partition of
`candidate_evaluations`; failed open-loop evaluations are rejected and cannot
also be counted as accepted.
The probe validates the returned `SearchRunResult` counters against this
partition and fails closed on an inconsistent legacy valid-count field.

The packet still runs no comparison campaign, submits no SLURM job, and makes
no benchmark, stress-strength, safety, planner-ranking, superiority, or
paper-facing claim. Domain-aware approval remains required before execution.

## What ships

- `configs/adversarial/issue_6921_matched_compute_packet.yaml`
  - `matched_compute_packet.v2`, explicitly revised from v1;
  - frozen crossing/time-to-collision (TTC) template identity and seeds;
  - packet-scoped `issue_6921_crossing_ttc_space.yaml`, preserving the
    approved crossing/TTC bounds while freezing every candidate's scenario
    seed to the template seed `123`;
  - `minimize_predicted_robot_distance` as the validated reactive objective;
  - a separately named `minimize_episode_min_robot_distance` open-loop
    projection, with no claim that the two projections are equivalent;
  - native runner bindings for both arms;
  - 9 candidate evaluations per reactive macro-action, 90 per arm per episode;
  - shared trace fields for arm, seeds, execution mode, simulator steps and
    step source, evidence status, macro-actions, candidate evaluations,
    validity counts, and status;
  - explicit execution, fallback, degraded, and claim exclusions.
- `robot_sf/adversarial/matched_compute.py`
  - shared trace/accounting dataclasses;
  - native reactive and open-loop seam adapters;
  - fail-closed accounting validation for the preflight canary.
- `tests/adversarial/test_matched_compute_packet.py`
  - packet schema, runner-binding, budget, provenance, seed, bounds, and gate
    checks.
- `tests/adversarial/test_matched_compute_runtime.py`
  - deterministic adapter probes using injected evaluators plus the actual
    `run_adversarial_search` seam with a synthetic evaluator;
  - native seam identity and shared accounting-schema checks;
  - malformed and non-native accounting rejection tests.

## Frozen contract

The scenario remains `crossing_ttc_template` from
`configs/scenarios/templates/crossing_ttc.yaml`, seed `123`, with search seed
and residual-adversary seed `42`. The residual bounds remain identical across
arms, and `target_ped_idx: [0]` remains the deliberate single-target choice.
The packet-scoped search space is derived from
`configs/adversarial/crossing_ttc_space.yaml`; its only semantic difference is
that `scenario_seed` is frozen to `123`, so the native open-loop runner cannot
silently sample seeds outside the packet.

The reactive arm evaluates the 3×3 residual grid at each of 10 macro-action
boundaries: 9 candidates per boundary and 90 candidate evaluations per
episode. The open-loop arm binds its existing scenario-search budget to 90
candidate evaluations per episode. The reactive one-step predicted-distance
proxy and the open-loop completed-episode minimum-distance objective are
different projections. These are declared budget fields, not observed
execution results; the shared trace records actual simulator work separately
so a later approval decision can reject an incomparable mapping.

## Native runner bindings

The open-loop binding is:

```text
robot_sf.adversarial.search.run_adversarial_search
robot_sf.adversarial.search.production_candidate_evaluator
policy: social_force
objective: minimize_episode_min_robot_distance
budget: 90, horizon: 50, dt: 0.1
```

The reactive binding is:

```text
robot_sf.ped_npc.residual_search.FiniteGridSearchPolicy
robot_sf.ped_npc.residual_adversary.BoundedResidualAdversary
```

The preflight does not substitute the former one-shot residual-controller
stand-in for the open-loop scenario search. The adapters report `native` for
code-path identity only; the current injected open-loop canary is
`diagnostic_only_preflight` with `synthetic_episode_fixture` step provenance,
and the reactive snapshot canary is `diagnostic_only_preflight` with
`controller_snapshot` provenance. Neither can populate observed simulator
physics steps. A `production_observed` trace is reserved for a later canary
that runs the actual evaluator/simulator and records its provenance. The
current probe never emits `production_observed`: omitting the optional
runner/evaluator hooks proves only callable selection, not per-candidate
native/no-fallback provenance or complete episode provenance.

## Canonical local validation

```bash
uv run pytest tests/adversarial/test_matched_compute_packet.py \
  tests/adversarial/test_matched_compute_runtime.py \
  tests/adversarial/test_matched_compute_objective.py -v
```

The open-loop unit coverage uses deterministic injected evaluators, and one
canary invokes the actual `run_adversarial_search` runner with a synthetic
per-candidate evaluator. The synthetic episode-step totals remain in metadata
only; the canonical simulator-step field is unavailable and the trace is
explicitly diagnostic-only. The reactive test uses a one-step controller
snapshot and likewise does not observe a simulator tick. These tests do not
launch benchmark batches or campaigns. A future production execution canary
must be run only after the domain-approval gate is granted; its output must be
promoted and provenance-checked before any stronger claim.

## Claim boundary and deferred work

This slice establishes only that the packet names real seams and that local
accounting validation is available. Preflight candidate counts and fixture or
controller counters are not observed simulator evidence. The distinct
objective projections are intentionally not treated as equivalent. It does
not establish stress strength, realism, safety, planner ranking, superiority,
benchmark performance, or a paper-facing result. Fallback/degraded execution
is never success evidence.

Deferred until explicit approval and an independent evidence review:

- comparison execution on the frozen scenario and seeds;
- a no-fallback production-path canary with observed candidate and simulator-
  step accounting for both arms;
- observed budget-parity decision from native traces;
- stress-case validity/strength metrics;
- CMA-ES/MCTS expansion and any learned/PPO adversary;
- benchmark or paper-facing synthesis.
