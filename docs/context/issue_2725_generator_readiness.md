# Issue #2725 - Generator-Readiness Contract for RL and Diffusion Scenario Generation

**Status:** not yet training-ready
**Parent issue:** #2725
**Schema:** `robot_sf/benchmark/schemas/generated_scenario_candidate.v1.json`
**Fail-closed:** yes - no benchmark-strength or training-ready claim is made until all prerequisite gates pass.

---

## 1. Scope

This document defines the readiness contract for generated scenario candidates produced by
three generator families:

1. **Heuristic perturbation** - bounded parameter-space search (random, TPE, grid) over
   existing scenario templates with typed perturbation families.
2. **RL adversary** - learned policy that proposes perturbation parameters or route
   overrides to maximize a difficulty objective.
3. **Diffusion / generative prior** - diffusion model or other generative prior that
   samples full scenario configurations (pedestrian poses, routes, timing) from a learned
   distribution.

A generated scenario candidate is **not** a benchmark result. It is a proposal that must
pass validity preflight, trace lineage, and evidence-boundary checks before it may be
considered for training data or benchmark promotion.

---

## 2. Scenario Encoding Prerequisites

Every generated candidate must reference a source scenario that satisfies:

| Prerequisite | Requirement |
|---|---|
| Scenario contract | Valid `scenario_contract.v1` with `certification.required_before_benchmark_claim: true` |
| Scenario cert | `scenario_cert.v1` eligibility must be `eligible` or `stress_only` - `unknown` is insufficient |
| Map registry | `map_id` resolves to an existing SVG file via `maps/registry.yaml` |
| Actor encoding | At least one `robot` and one `pedestrian` actor defined in the scenario contract |
| Motion models | Every actor has a named motion model that is loadable by the simulation backend |
| Seed determinism | Source scenario must be reproducible from recorded seed + config hash |

Candidates that reference scenarios missing any prerequisite are rejected at validation time.

---

## 3. Action / Perturbation Space

### 3.1 Heuristic Perturbation Families

The following families are supported by the `generated_scenario_candidate.v1` schema:

| Family | Parameters | Bounds Source |
|---|---|---|
| `robot_route_offset` | `dx_m`, `dy_m`, `max_magnitude_m` | Search-space YAML |
| `pedestrian_route_offset` | `dx_m`, `dy_m`, `max_magnitude_m` | Search-space YAML |
| `single_pedestrian_speed_offset` | `speed_delta_m_s`, `max_abs_speed_delta_m_s` | Search-space YAML |
| `single_pedestrian_start_delay_offset` | `dt_s`, `max_abs_dt_s` | Search-space YAML |
| `single_pedestrian_wait_duration_offset` | `wait_delta_s`, `max_abs_wait_delta_s` | Search-space YAML |
| `single_pedestrian_trajectory_waypoint_offset` | `dx_m`, `dy_m`, `max_magnitude_m`, `pedestrian_id` | Search-space YAML |
| `pedestrian_density_offset` | `density_delta`, `max_abs_density_delta` | Search-space YAML |
| `noop` | none | - |

Unsupported families are rejected by the schema (`additionalProperties: false` on the
perturbation object).

### 3.2 RL Adversary Perturbation Families

RL adversaries operate in the same parameter space as heuristic perturbations but propose
parameters via a learned policy. The candidate must record:

- `generator_family: "rl_adversary"`
- `generator_config_ref` pointing to the adversary training config
- `adversary_checkpoint_ref` pointing to the specific checkpoint used

RL adversary candidates that propose parameters outside the search-space bounds are
invalidated by the same validity constraints as heuristic candidates.

### 3.3 Diffusion / Generative Prior Families

Diffusion models may generate:

- Full scenario configurations (pedestrian spawns, goals, timing)
- Route overrides for existing scenarios
- Hybrid outputs that combine generated and template parameters

The candidate must record:

- `generator_family: "diffusion_prior"` or `"generative_prior"`
- `generator_config_ref` pointing to the model config
- `model_checkpoint_ref` pointing to the specific model weights
- `sample_params` (temperature, guidance scale, etc.)

Diffusion-generated candidates must still satisfy all validity constraints. The schema
rejects candidates that claim diffusion-origin but use heuristic-only perturbation fields
without the required generator metadata.

---

## 4. Validity Constraints

Every candidate is subject to:

| Constraint | Rule | Failure Action |
|---|---|---|
| Route offset magnitude | `<= max_magnitude_m` from search-space config | Reject candidate |
| Speed delta | `<= max_abs_speed_delta_m_s` and resulting speed `<= max_speed_m_s` | Reject candidate |
| Start delay | `<= max_abs_dt_s` | Reject candidate |
| Wait duration | `<= max_abs_wait_delta_s` | Reject candidate |
| Density delta | `<= max_abs_density_delta` and resulting density `<= max_ped_density` | Reject candidate |
| Scenario certification | Source must have `required_before_benchmark_claim: true` | Reject candidate |
| Map existence | `map_id` must resolve to an existing SVG | Reject candidate |
| Spawn collision | Pedestrian spawn must not overlap existing agents | Reject candidate |
| Goal reachability | Pedestrian goal must be on navigable surface | Reject candidate |

Validity preflight is **not** benchmark execution evidence. It is a minimum gate for
candidate admission.

---

## 5. Severity Metrics

Severity metrics quantify how challenging a generated candidate is for planners.

| Metric | Description | Unit |
|---|---|---|
| `ttc_min_s` | Minimum time-to-collision across all agent pairs | seconds |
| `min_clearance_m` | Minimum agent-to-agent clearance | meters |
| `comfort_force_max_N` | Maximum social force on the ego robot | Newtons |
| `near_miss_count` | Number of near-miss events (clearance < threshold) | count |
| `collision_count` | Number of collision events | count |
| `timeout_risk` | Fraction of episodes that timeout for baseline planner | fraction [0, 1] |
| `objective_value` | Search objective score (e.g., `worst_case_snqi`) | scalar |

Severity metrics are **placeholders** until evaluated. The schema requires that
severity metrics fields are present but allows `null` values to indicate unevaluated
status.

---

## 6. Diversity Metrics

Diversity metrics ensure the generated candidate pool covers the scenario space.

| Metric | Description | Unit |
|---|---|---|
| `param_distance_min_m` | Minimum parameter-space distance to nearest existing candidate | meters |
| `param_distance_mean_m` | Mean parameter-space distance to nearest K existing candidates | meters |
| `unique_scenario_families` | Number of distinct scenario families covered | count |
| `coverage_fraction` | Fraction of search-space dimensions with non-trivial exploration | fraction [0, 1] |
| `dedup_rate` | Fraction of candidates within dedup threshold of another candidate | fraction [0, 1] |

Diversity metrics are **placeholders** until the candidate pool is assembled. The schema
allows `null` values for unevaluated diversity metrics.

---

## 7. Trace Lineage

Every candidate must carry full trace lineage:

| Field | Description |
|---|---|
| `candidate_id` | Unique identifier for this candidate |
| `generator_family` | One of `heuristic_perturbation`, `rl_adversary`, `diffusion_prior`, `generative_prior` |
| `generator_run_id` | Identifier for the generator run that produced this candidate |
| `source_scenario_ref` | Reference to the source scenario (name, config path, hash) |
| `source_split` | Which split the source scenario belongs to (`train`, `validation`, `test`) |
| `perturbations` | Typed list of perturbation operations applied |
| `config_hash` | SHA-256 hash of the generator config used |
| `seed` | Random seed used for this candidate |
| `generated_at_utc` | ISO-8601 timestamp of generation |
| `git_hash` | Repository commit at generation time |
| `provenance` | Additional provenance metadata (checkpoint path, model version, etc.) |

Missing trace lineage is a hard rejection. The schema uses `required` fields and
`additionalProperties: false` to enforce completeness.

---

## 8. Train / Validation Split and Leakage Policy

| Rule | Requirement |
|---|---|
| Source split declaration | Every candidate must declare `source_split` for its source scenario |
| Split isolation | Generated candidates must not be used to train a planner that is then evaluated on scenarios from the same source split |
| Validation split | Candidates derived from validation-split scenarios may only be used for validation metrics, never for training data selection |
| Test split | Candidates derived from test-split scenarios are held out entirely from any training pipeline |
| Cross-split leakage check | Before training, a leakage audit must confirm no generated candidate shares a `source_scenario_ref` with the evaluation set |
| Pedestrian identity leakage | If a generated candidate reuses specific pedestrian spawn/goal identities from a training scenario, it must be flagged for potential identity leakage |

The schema records `source_split` and `generator_run_id` to enable automated leakage
detection. Claims of "no leakage" require a positive audit result, not just schema
compliance.

---

## 9. Benchmark Promotion Rules

A generated candidate may be promoted to benchmark-strength evidence only when:

| Gate | Requirement |
|---|---|
| Schema validity | Candidate passes `generated_scenario_candidate.v1` validation |
| Validity preflight | All validity constraints pass |
| Trace lineage | Complete lineage with no gaps |
| Replay determinism | Re-generation from recorded seed + config produces identical candidate (within tolerance) |
| Severity evaluation | All severity metrics are non-null and computed by a validated evaluator |
| Diversity evaluation | Pool-level diversity metrics are non-null and above minimum thresholds |
| Leakage audit | Cross-split leakage audit passes with zero violations |
| Benchmark config | Candidate is included in a frozen benchmark manifest with explicit scope |
| Evidence boundary | `promotion_status` is set to `promoted` by an authorized gate - default is `not_promoted` |

**Fail-closed default:** `promotion_status` defaults to `"not_promoted"`. Candidates that
have not passed all gates remain in `not_promoted` status and must not be cited as
benchmark evidence.

---

## 10. Stop Rules

| Condition | Action |
|---|---|
| Validity preflight fails | Reject candidate; do not evaluate severity |
| Trace lineage incomplete | Reject candidate; do not evaluate severity |
| Source scenario not certified | Reject candidate |
| Replay determinism fails | Reject candidate; investigate generator |
| Leakage audit fails | Block training; do not promote candidates |
| Severity metrics null | Candidate is not promotion-eligible |
| Diversity metrics below threshold | Candidate pool is not promotion-eligible |
| Any prerequisite status is `false` or missing | Status remains `not yet training-ready` |

---

## 11. Not Yet Training-Ready

**This generator-readiness contract is not training-ready.** The following conditions
must be met before any generated scenario candidate may be used for training data or
promoted to benchmark evidence:

1. **Executable schema validator** - The `generated_scenario_candidate.v1` JSON Schema
   must be backed by an executable Python validator with the same rejection semantics.
2. **Severity evaluator** - A validated evaluator must compute all severity metrics for
   each candidate.
3. **Diversity evaluator** - A pool-level diversity evaluator must compute coverage and
   dedup metrics.
4. **Leakage audit** - An automated leakage audit must confirm cross-split isolation.
5. **Replay determinism proof** - At least one candidate from each generator family must
   pass replay determinism verification.
6. **Benchmark manifest inclusion** - Candidates must be included in a frozen benchmark
   manifest with explicit scope and comparison axes.

Until all six conditions are satisfied, the status is **not yet training-ready** and no
benchmark-strength claims may be made.

---

## 12. Generator Family Distinctions

| Property | Heuristic Perturbation | RL Adversary | Diffusion / Generative Prior |
|---|---|---|---|
| Parameter source | Random/grid/Bayesian search | Learned policy | Learned distribution |
| Search space | Explicit YAML config | Same param space (or extended) | Full scenario config |
| Diversity mechanism | Search-space coverage | Exploration bonus / entropy | Distributional diversity |
| Severity mechanism | Search objective (e.g., `worst_case_snqi`) | Adversarial reward | Implicit via distribution |
| Reproducibility | Deterministic from seed + config | Deterministic from seed + checkpoint | Deterministic from seed + model + params |
| Schema `generator_family` | `"heuristic_perturbation"` | `"rl_adversary"` | `"diffusion_prior"` or `"generative_prior"` |
| Extra required fields | None | `adversary_checkpoint_ref` | `model_checkpoint_ref`, `sample_params` |
| Maturity | Available (v1) | Not yet implemented | Not yet implemented |
