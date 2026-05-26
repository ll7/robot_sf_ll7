# Issue #1488 Bounded Adversarial Search Methodology

Related issue: [#1488](https://github.com/ll7/robot_sf_ll7/issues/1488)  
Branch: `issue-1488-adversarial-methodology`

## Executive Summary

Issue `#1488` is scientifically promising as a methodology contribution, not as a new optimizer
contribution. Its value is in fail-closed status accounting, bounded budget normalization,
replay-backed adversarial evidence, and stress-coverage reporting for social-navigation planners.
The issue asks for random, TPE-style, and guided adversarial search under fixed seeds, fixed
budgets, and compact evidence artifacts.

The main design risk is denominator corruption. Invalid candidates, simulation errors,
fallback/degraded planner rows, and non-replayable failures can make an adaptive search method
appear more effective than it is. The parent issue already identifies this risk and excludes
fallback, degraded, invalid, and simulation-error outcomes from successful benchmark evidence.

The first campaign should include exactly three search engines:

- `seeded_random`
- `optuna_tpe`
- `guided_route_start_state`

AST, STL falsification, full coverage-guided fuzzing, MAP-Elites, and LLM-guided scenario
generation are useful background but too heavy for the first bounded Robot SF campaign unless they
already have comparable budget and replay interfaces.

The primary fairness normalization should be equal attempted-candidate budget per
`(scenario_family, planner_row, search_engine, seed)`, with a fixed per-candidate episode horizon
and secondary reporting per valid candidate, per simulation-hour, and per wall-clock-hour.
Valid-candidate normalization should not be primary because it hides invalid-candidate generation.

Failure discovery should be reported as replayable unique behavioral failures, not raw first-pass
failures. The minimum report should include cumulative failure-discovery curves, first-failure
index, failures per attempted candidate, failures per valid candidate, failures per
simulation-hour, and top-k failure quality under a pre-registered severity score.

Failure diversity can be implemented compactly without a new ML pipeline using a deterministic
failure signature:

```text
(scenario_family, planner_row, failure_mode, trigger_class, relative_geometry_bin,
 min_distance_bin, TTC_bin, progress_bin)
```

This supports unique-failure counting, duplicate suppression, entropy over failure modes, and
planner-specific versus planner-agnostic failure separation.

A compact campaign can support the claim that one search engine found more replayable stress
failures than another within the tested scenario families and fixed budget. It cannot support
claims about planner safety in general, real-world risk, nominal benchmark superiority, or failure
probability under an operational design domain.

## Literature Map

### Random Search / Seeded Random Search

Random search samples candidate scenario parameters from a fixed distribution. Its scientific role
is a baseline with minimal modeling assumptions. Bergstra and Bengio showed in hyperparameter
optimization that random trials are often stronger than grid search when only a subset of dimensions
matters; for Robot SF, the analogous point is that random search is a credible baseline when
route/start-state parameters have sparse stress-relevant dimensions.

Seeded random search is the required control condition. It exposes whether TPE or guided
adversarial search provides real sample-efficiency gains rather than merely benefiting from a
permissive denominator. It may waste budget on invalid candidates or non-stressful valid cases,
especially if scenario constraints are tight, but that weakness gives a conservative baseline.

### Bayesian Optimization / TPE / Optuna-Like Search

Sequential model-based optimization uses past trials to bias future candidates toward promising
regions. TPE models good and bad regions using density estimators and selects candidates by an
expected-improvement-like criterion; Optuna provides a lightweight define-by-run optimization
framework and TPESampler implementation.

TPE fits a compact black-box stress campaign because it can optimize a scalar pre-registered
`stress_score` without access to planner internals. It also handles mixed continuous/categorical
scenario parameters better than a Gaussian-process BO implementation in many small engineering
search spaces. It is sensitive to objective definition, startup trials, invalid trials, and small
budgets. It should not receive a hidden advantage from objective tuning after observing failures.
Freeze startup-trial count, sampler seed, parameter ranges, and objective before execution.

### Adaptive Stress Testing

Adaptive stress testing frames failure discovery as a sequential decision problem, often an MDP, and
uses reinforcement learning, Monte Carlo tree search, or stochastic optimization to find likely
failure trajectories. AST is relevant conceptually because Robot SF is also looking for planner
failures in a stochastic interactive environment.

Full AST requires a sequential action interface, a disturbance model, reward design, and often large
simulation budgets. It is a poor first-campaign fit unless Robot SF already exposes search over
sequential human-agent perturbations rather than static route/start-state parameters. Treat AST as
background justification and future extension.

### Falsification And Temporal-Logic-Based Testing

Falsification tools such as S-TaLiRo, Breach, and VerifAI search for counterexamples to
temporal-logic specifications or robustness metrics over simulated traces. These tools justify
robustness-valued safety and comfort objectives such as minimum clearance, time-to-collision,
proxemic violation margin, goal-progress failure, and jerk/comfort threshold violations.

They require formal property definitions and trace-monitoring infrastructure. If Robot SF only has
metric summaries rather than STL monitors, implementing falsification now would expand scope. Use
falsification ideas to define objective functions and failure oracles, but do not include a separate
falsification engine in the first campaign.

### Scenario Parameter Search And Autonomous-Driving Fuzzing

AV-Fuzzer perturbs traffic participant behaviors and uses a genetic algorithm with domain knowledge
to find autonomous-driving safety violations; DriveFuzz mutates driving scenarios and guides search
using driving-quality metrics and test oracles. Robot SF's guided route/start-state search is
closer to this family than to full formal falsification.

The transferable idea is that scenario generation must separate test generation, oracle evaluation,
failure status, and artifact replay. Many AV fuzzers assume vehicle traffic rules,
CARLA/Apollo/Autoware stacks, or high-fidelity driving-simulator APIs. Direct adoption would be
mismatched for micromobility/social-navigation unless converted to social-agent interactions and
proxemic metrics. Include only the existing Robot SF guided route/start-state search path.

### Guided Route/Start-State Adversarial Search

This family uses domain-specific constraints and heuristics to generate route conflicts, initial
geometries, crowd interactions, or start-goal configurations likely to stress the planner. It is
likely the strongest first-campaign method because it uses repository-specific knowledge without
requiring a new RL/falsification pipeline.

It can overfit to known planner weaknesses or to one scenario family. It must be compared under the
same attempted-candidate budget and must report invalid candidates. Include it as the domain-guided
treatment condition.

### Coverage-Guided Testing

Coverage-guided testing uses explicit coverage criteria to steer or evaluate test generation.
Internal DNN neuron coverage is probably irrelevant unless Robot SF evaluates learning-based
planners with accessible internals. Feature-space coverage is highly relevant: coverage over
declared stress axes is exactly what `stress_uncertainty_coverage.v1` should report.

Include coverage reporting, not coverage-guided search.

### Diversity-Seeking Failure Discovery

Novelty search and quality-diversity algorithms avoid returning a single best failure by exploring
behavior space. Failure diversity is central to `#1488`; a single repeated collision geometry is
much weaker evidence than multiple distinct failure signatures. Full MAP-Elites or
quality-diversity search adds another optimizer and archive design. For a compact campaign,
diversity should first be a reporting metric, not a fourth search engine.

### Social-Navigation Evaluation And Metrics

Social-navigation evaluation literature emphasizes safety, efficiency, comfort, smoothness,
human-awareness, and scenario/task context. Robot SF should not reduce failure to collision alone.
Valid behavioral failures should include deadlock, severe social-space intrusion, excessive
discomfort/jerk, timeout, path inefficiency under safety constraints, and planner instability.
These metrics should be used as stress evidence, not direct claims about human acceptance.

### Scenario Standards And Scenario-Based Safety Assessment

Scenario-based testing is widely used for automated-driving validation. Even if Robot SF does not
use OpenSCENARIO, it should adopt the same discipline: declared scenario families, logical
parameter ranges, concrete sampled candidates, replayable manifests, and explicit coverage gaps.
Use the principle, not the full standard.

### Recent LLM-Guided Scenario Generation

LLM-guided scenario generation is relevant for future semantic scenario generation and
interpretation notes, especially if Robot SF later needs natural-language scenario descriptions. It
adds prompt nondeterminism, model-version dependence, and another validity layer, so it is not
appropriate for the first bounded reproducibility campaign.

## Method Comparison Table

| Method family | Examples / representative papers | Optimization target | Budget type | Strengths | Weaknesses | Robot SF fit | First-campaign role |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| Seeded random search | Bergstra and Bengio random search baseline | None or uniform stress parameter sampling | Attempted candidates | Simple, reproducible, strong baseline, no modeling assumptions | Inefficient for rare failures; may generate invalid candidates | High | Mandatory baseline |
| TPE / Optuna-like search | Bergstra et al. TPE; Optuna TPESampler | Scalar `stress_score` or severity objective | Attempted candidates, plus wall-clock | Lightweight adaptive black-box search; handles mixed parameters | Objective-sensitive; small-budget instability; invalid-trial handling matters | High | Main adaptive comparator |
| Adaptive stress testing | Koren et al.; Lee et al.; POMDPStressTesting.jl | Most likely failure trajectory under disturbance model | Simulation calls or simulation steps | Strong theoretical fit for sequential rare-event failure search | Requires MDP/disturbance interface and large budget | Medium | Background and future extension |
| Temporal-logic falsification | S-TaLiRo, Breach, VerifAI | Robustness margin of temporal property | Simulation calls | Precise oracles; strong property semantics | Requires STL/MTL monitors and property engineering | Medium | Use ideas for failure oracles, not separate engine |
| AV fuzzing / scenario parameter search | AV-Fuzzer, DriveFuzz | Safety violation, driving-quality degradation | Candidate scenarios or simulation time | Practical scenario generation; strong bug-finding precedent | Vehicle-stack assumptions; may not fit social navigation directly | Medium | Conceptual support for guided search |
| Existing guided route/start-state search | Robot SF issue context | Domain-specific conflict/stress heuristic | Attempted candidates | Repository-specific, compact, likely effective | Overfitting and invalid-candidate risk | High | Main guided treatment |
| Coverage-guided DNN testing | DeepXplore, DeepTest, DeepHyperion | Neuron coverage or feature-space illumination | Generated inputs | Strong coverage vocabulary; interpretable feature maps | Neuron coverage mismatched for black-box planners | Medium | Use feature-space coverage reporting |
| Diversity-seeking search | Novelty search, MAP-Elites, diverse AST | Novelty plus failure quality | Archive cells / candidate budget | Prevents duplicate failure overclaiming | Adds optimizer complexity | Medium | Report diversity; do not add new engine yet |
| Scenario DSL / standards | Scenic, ASAM OpenSCENARIO DSL | Declarative scenario distributions and constraints | Sampling budget | Good manifest discipline and scenario reuse | Automotive semantics may not map directly | Medium | Inform manifest design |
| LLM-guided scenario generation | Gao et al. 2025 | Semantic risk evaluation and adversarial generation | Prompt/model calls plus simulation | Emerging semantic generation path | Versioning, nondeterminism, prompt bias | Low | Exclude from first campaign |

## Recommended Experimental Design

### Search Engines

Use exactly three engines in the first bounded comparison:

1. `seeded_random`: uniform or declared-distribution sampling over the frozen parameter ranges.
2. `optuna_tpe`: Optuna TPESampler with fixed sampler seed, fixed startup trial count, fixed
   objective, and no post hoc objective tuning.
3. `guided_route_start_state`: the existing Robot SF guided adversarial route/start-state search
   path, exposed through the same candidate interface as the other engines.

Do not include AST, STL falsification, MAP-Elites, LLM generation, or new fuzzers in the first
campaign. They are scientifically relevant but would turn `#1488` from a bounded comparison into an
optimizer-development project.

### Budget Normalization

Use equal attempted-candidate budget as the primary normalization:

```text
primary_budget_cell =
  scenario_family x planner_row x search_engine x campaign_seed
  -> N_attempted_candidates fixed
```

This is the correct primary denominator because invalid candidates, simulation errors, and
fallback/degraded rows are part of the search method's practical behavior. A method that proposes
many invalid cases should be penalized rather than normalized away.

Report secondary normalizations:

```text
failures_per_valid_candidate
failures_per_simulation_hour
failures_per_wall_clock_hour
first_failure_candidate_index
AUC_failure_discovery_curve
```

Do not use equal valid-candidate budget as the primary comparison.

### Scenario-Family Count

Use a two-stage design:

- Smoke stage: `1` scenario family.
- Bounded comparison: `2` scenario families.

Choose families with stable replay contracts and different stress mechanisms, for example:

```text
family_A: crossing_conflict / proxemic conflict
family_B: bottleneck / crowd-flow / route-conflict deadlock
```

The two families should differ in failure mechanism, not merely parameter values.

### Planner-Row Count

Use:

- Smoke stage: `1` planner row if runtime is constrained; `2` if both are already stable.
- Bounded comparison: `2` planner rows.

Planner rows should be selected before search and should include only runnable planners satisfying
the fail-closed benchmark contract. A fallback planner row is not a valid substitute for the
intended planner row.

### Seed Policy

Use hierarchical deterministic seeds:

```text
campaign_seed
  search_seed
  scenario_seed
  simulator_seed
  planner_seed
  replay_seed
```

Derive child seeds from a stable hash:

```text
seed_child = hash64(campaign_id, cell_id, seed_role, seed_index)
```

Record all seeds in the manifest. Do not use ambient RNG state, wall-clock timestamps, process IDs,
or nondeterministic seed allocation.

### Candidate Budgets

Recommended compact defaults:

```text
smoke:
  families = 1
  planner_rows = 1
  search_engines = 3
  campaign_seeds = 3
  attempted_candidates_per_cell = 20
  total_attempted = 180

two_family:
  families = 2
  planner_rows = 2
  search_engines = 3
  campaign_seeds = 3
  attempted_candidates_per_cell = 30
  total_attempted = 1080
```

If SLURM capacity is available, increase only `attempted_candidates_per_cell` to `50`, keeping the
same families, planner rows, and seeds.

### Replay Policy

Replay all discovered failures if the count is small. If the count is large, replay:

```text
top_k_by_severity = 5 per search_engine x scenario_family x planner_row
top_k_by_diversity = 5 additional unique failure signatures if available
replay_repetitions = 3
```

A failure is `replayable` only if all three replays reproduce the same failure status and failure
signature within pre-registered metric tolerances.

### Artifact Policy

Commit only compact durable artifacts:

```text
manifest.yaml
run_summary.csv
status_counts.csv
failure_index.csv
topk_replay_manifest.yaml
stress_uncertainty_coverage.v1.json
checksums.sha256
interpretation_note.md
```

Do not commit raw episode dumps, large simulator logs, videos, generated maps, uncompressed
trajectories, temporary `output/` folders, planner debug traces, screenshots, or local caches. If
raw artifacts are needed for audit, store them outside git and commit only URI, checksum, byte size,
generation command, and retention note.

### Failure Status Taxonomy

Use the taxonomy below. The key rule is:

```text
Only valid_behavioral_failure intersection replayable intersection non_duplicate
counts as primary failure evidence.
```

### Smoke-Stage Acceptance Criteria

Accept the smoke stage only if:

```text
manifest validates against schema
all three search engines execute from the same manifest
status taxonomy is populated for every attempted candidate
valid_trial_rate is reported per engine
invalid_candidate_rate is reported per engine
simulation_error_rate is reported per engine
fallback/degraded/unavailable rows are explicit
no simulation_error row is counted as failure evidence
top-k replay path executes if at least one failure is found
stress_uncertainty_coverage.v1 is emitted
no raw artifacts are committed
```

Reject or revise if:

```text
any engine lacks comparable candidate-budget accounting
simulation_error_rate > 20% in any engine without diagnosis
valid_trial_rate < 50% in any engine without diagnosis
fallback/degraded execution is aggregated with success
failures are reported without denominator counts
```

Finding no failure in the smoke stage is not a failure of the smoke stage. The smoke stage tests
execution, accounting, and replay plumbing.

### Two-Family-Stage Acceptance Criteria

Accept the bounded comparison only if:

```text
two scenario families complete for all planned search engines
at least two planner rows complete or unsupported rows are explicit
all candidate statuses are denominator-accounted
all replayable failures have replay manifests and checksums
unique-failure counts use deterministic duplicate suppression
failure-discovery curves are reported per engine
coverage reports separate candidate coverage, valid coverage, and failure coverage
uncertainty intervals are reported over campaign seeds
interpretation note states allowed and disallowed claims
```

Reject or revise if:

```text
any search engine receives broader parameter ranges
objective changes after observing failures
early stopping removes denominator comparability
non-replayable failures are counted as primary evidence
valid-only normalization is presented as the primary result
```

## Metrics And Reporting Schema

### Valid-Trial Metrics

Let:

```text
N_attempted = all generated candidates
N_valid_non_failure = valid candidates with no behavioral failure
N_valid_failure = valid candidates with behavioral failure
N_invalid = candidates rejected by scenario constraints
N_sim_error = candidates that fail due to simulator/runtime errors
N_unavailable = planner unavailable or row cannot execute
N_fallback = fallback/degraded execution rows
```

Report:

```text
valid_trial_rate =
  (N_valid_non_failure + N_valid_failure) / N_attempted

invalid_candidate_rate =
  N_invalid / N_attempted

simulation_error_rate =
  N_sim_error / N_attempted

fallback_degraded_rate =
  N_fallback / N_attempted

availability_rate =
  1 - (N_unavailable / N_attempted)
```

Do not remove invalid or simulation-error candidates from the primary denominator.

### Failure-Efficiency Metrics

Primary:

```text
replayable_unique_failures_per_attempted_candidate =
  N_replayable_unique_failures / N_attempted
```

Secondary:

```text
replayable_unique_failures_per_valid_candidate =
  N_replayable_unique_failures / (N_valid_non_failure + N_valid_failure)

replayable_unique_failures_per_sim_hour =
  N_replayable_unique_failures / total_simulation_hours

first_failure_index =
  min(candidate_index where replayable_unique_failure discovered)

AUC_failure_discovery =
  normalized area under cumulative unique replayable failures vs candidate index

top_k_failure_quality =
  severity_score of top-k replayable unique failures
```

The `severity_score` must be pre-registered. A compact social-navigation score could be:

```text
severity_score =
  w_collision * collision_indicator
  + w_clearance * max(0, clearance_threshold - min_clearance)
  + w_ttc * max(0, ttc_threshold - min_ttc)
  + w_progress * max(0, required_progress - achieved_progress)
  + w_comfort * max(0, jerk_or_accel_margin)
```

The weights must be frozen before the campaign.

### Failure-Diversity Metrics

Use a compact deterministic signature:

```text
failure_signature =
  scenario_family
  failure_mode
  trigger_class
  actor_relation_class
  relative_bearing_bin
  initial_gap_bin
  min_clearance_bin
  min_ttc_bin
  progress_bin
  planner_row
```

Report:

```text
unique_failure_count = count(unique(failure_signature))

duplicate_failure_count =
  N_replayable_failures - unique_failure_count

failure_mode_entropy =
  Shannon entropy over failure_mode labels

scenario_family_diversity =
  count(families with >=1 unique replayable failure)

planner_specific_failures =
  signatures occurring in exactly one planner row

planner_agnostic_failures =
  signatures occurring in >=2 planner rows under same replay policy
```

Recommended failure-mode labels:

```text
collision
near_miss
proxemic_violation
deadlock
timeout
goal_noncompletion
comfort_violation
oscillation_or_instability
rule_or_constraint_violation
```

### Replay-Determinism Metrics

Report:

```text
replay_success_rate =
  N_failures_reproduced_3_of_3 / N_failures_selected_for_replay

status_stability_rate =
  N_replays_same_status_3_of_3 / N_replay_attempts_grouped

metric_tolerance_pass_rate =
  N_replays_within_metric_tolerance / N_replay_attempts_grouped

non_replayable_failure_rate =
  N_non_replayable_failures / N_failures_selected_for_replay
```

Metric tolerances should be frozen, for example:

```text
min_clearance_tolerance_m = 0.05
min_ttc_tolerance_s = 0.10
completion_time_tolerance_s = 0.50
path_length_tolerance_m = 0.25
```

### Stress-Coverage Metrics

Define coverage over declared bins, not over an implicit unbounded scenario space:

```text
stress_axes:
  scenario_family
  crowd_density_bin
  initial_gap_bin
  crossing_angle_bin
  route_conflict_bin
  robot_speed_bin
  human_speed_bin
  occlusion_or_visibility_bin
  proxemic_zone_bin
```

Report three separate coverages:

```text
candidate_coverage =
  bins_sampled_by_any_candidate / bins_declared

valid_coverage =
  bins_with_valid_trials / bins_declared

failure_coverage =
  bins_with_replayable_unique_failure / bins_declared
```

Also report:

```text
invalid_only_bins
sim_error_only_bins
uncovered_bins
valid_safe_bins
valid_failure_bins
```

This prevents an engine from claiming coverage by sampling regions that only produce invalid
candidates.

### Uncertainty And Caveat Reporting

For small campaigns:

```text
rate intervals:
  Wilson or exact binomial intervals for valid_trial_rate,
  invalid_candidate_rate,
  simulation_error_rate,
  replay_success_rate

engine comparison:
  bootstrap over campaign seeds for difference in
  replayable_unique_failures_per_attempted_candidate

failure-discovery curve:
  median and seed-wise envelope

ranking sensitivity:
  report planner rank under nominal metrics,
  planner rank under adversarial stress metrics,
  and rank-change magnitude
```

Do not estimate real-world failure probability from adversarial samples. Adaptive and guided
searches intentionally bias the sample distribution.

## Failure Status Taxonomy

| Status | Definition | Counts as valid trial? | Counts as primary failure evidence? | Required fields |
| --- | --- | ---: | ---: | --- |
| `valid_behavioral_failure` | Candidate is valid, intended planner runs, and pre-registered oracle detects behavioral failure | Yes | Only after replay and duplicate filtering | failure mode, metrics, oracle, severity, signature |
| `valid_non_failure` | Candidate is valid and intended planner completes without failure | Yes | No | metrics, completion status |
| `invalid_candidate` | Candidate violates scenario construction constraints before valid simulation | No | No | violated constraint, parameter values |
| `simulation_error` | Simulator/runtime fails independently of planner behavior | No | No | error class, log checksum, reproducibility note |
| `planner_unavailable` | Intended planner row cannot run due to missing dependency/configuration | No | No | planner row, reason |
| `fallback_degraded_row` | Execution uses fallback, degraded, adapted, or substituted planner behavior | No for primary benchmark comparison | No | fallback type, trigger, intended planner |
| `non_replayable_failure` | First-pass failure cannot be reproduced under replay policy | Yes as first-pass valid candidate, no as replayable evidence | No | replay attempts, divergence reason |
| `duplicate_failure` | Replayable failure matching an existing failure signature | Yes | No for unique failure count | duplicate signature, canonical failure ID |

Fail-closed rule:

```text
Unknown status -> not evidence.
Ambiguous failure -> not primary evidence.
Simulation error -> not failure.
Fallback/degraded -> not success.
Non-replayable -> not primary failure evidence.
```

## Threats To Validity

### Internal Validity

Search engines may receive unequal objectives, constraints, warm-start behavior, or
candidate-rejection handling. Guided search may encode knowledge unavailable to random/TPE search.
TPE may benefit from post hoc objective shaping unless the objective is frozen before execution.
Simulator nondeterminism can produce false failures or hide replay fragility. Fallback/degraded
planner execution can contaminate planner-row comparisons. Early stopping after discovering
failures corrupts denominator comparability.

Mitigation:

```text
fixed manifest
fixed objective
equal attempted-candidate budget
frozen seed schedule
explicit status taxonomy
top-k replay
no early stopping except fail-closed infrastructure abort
```

### External Validity

Two scenario families cannot represent the full social-navigation operating domain. Simulated human
agents may not reflect real pedestrian behavior. Micromobility planner performance may differ under
perception noise, actuation limits, real road surfaces, or regulatory constraints. A small
planner-row set cannot generalize to all planning architectures.

Mitigation:

```text
state tested families explicitly
avoid ODD-level claims
report uncovered stress bins
describe simulator and planner assumptions
```

### Construct Validity

Collision, minimum distance, time-to-collision, jerk, and progress are proxies for safety, comfort,
and efficiency. Proxemic thresholds are context-dependent. Failure signatures approximate
diversity; they are not guaranteed root-cause labels. A scalar `stress_score` may overemphasize one
failure class.

Mitigation:

```text
separate safety, comfort, and efficiency metrics
report failure-mode taxonomy
freeze severity formula
include interpretation notes
```

### Statistical Validity

Adaptive-search samples are not IID. Small seed counts produce wide uncertainty intervals. Multiple
metrics create cherry-picking risk. Failure counts may be zero-inflated or duplicate-heavy. Planner
ranking sensitivity may be unstable under few failures.

Mitigation:

```text
report seed-wise results
show intervals and not only point estimates
use cumulative curves
pre-register primary metric
do not claim population failure probability
```

### Artifact/Provenance Validity

Replays may depend on uncommitted code, environment variables, simulator versions, map files, or
hidden generated artifacts. Raw traces may be too large to commit, causing later irreproducibility
if not checksummed. Git commit hashes alone are insufficient if generated assets are not versioned.
Local-only `output/` files are not durable evidence.

Mitigation:

```text
manifest-only replay where possible
checksums for every generated compact artifact
external URI + checksum for raw audit data if needed
dirty-tree flag
container/simulator version recording
```

## Implementation Implications For Robot SF

### Manifest Fields

Recommended `adversarial_campaign_manifest.v1` fields:

```yaml
schema_version: adversarial_campaign_manifest.v1
campaign_id: robot_sf_1488_two_family_v1
parent_issue: 1488
child_issue: 1502

repository:
  url: https://github.com/ll7/robot_sf_ll7
  commit: "<git_sha>"
  dirty_tree: false
  branch: "<branch_name>"

environment:
  os: "<name/version>"
  ros_distro: "<ros2_distro>"
  python_version: "<version>"
  simulator_version: "<version>"
  container_image: "<image_digest_or_null>"
  hardware_class: "local|slurm|other"

budget:
  attempted_candidates_per_cell: 30
  max_episode_seconds: 60
  max_wall_clock_seconds_per_cell: 7200
  replay_repetitions: 3
  early_stop_policy: "none_except_fail_closed_abort"

scenario_families:
  - family_id: crossing_conflict_v1
    parameter_ranges: {}
    validity_constraints: []
    coverage_bins: {}
  - family_id: bottleneck_deadlock_v1
    parameter_ranges: {}
    validity_constraints: []
    coverage_bins: {}

planner_rows:
  - planner_id: "<planner_a>"
    config_path: "<path>"
    fallback_allowed: false
  - planner_id: "<planner_b>"
    config_path: "<path>"
    fallback_allowed: false

search_engines:
  - engine_id: seeded_random
    seed_role: search_seed
    config: {}
  - engine_id: optuna_tpe
    sampler: TPESampler
    startup_trials: 10
    objective: stress_score_v1
    config: {}
  - engine_id: guided_route_start_state
    config: {}

seeds:
  campaign_seeds: [1001, 1002, 1003]
  derivation: "hash64(campaign_id, cell_id, seed_role, seed_index)"

oracles:
  failure_oracle_version: failure_oracle.v1
  severity_score_version: stress_score.v1
  metric_tolerances:
    min_clearance_m: 0.05
    min_ttc_s: 0.10
    completion_time_s: 0.50

status_taxonomy:
  version: failure_status_taxonomy.v1

artifact_policy:
  commit_compact_artifacts_only: true
  forbidden_git_paths:
    - output/
    - videos/
    - raw_episodes/
    - generated_maps/
```

### Replay Manifest Fields

```yaml
schema_version: adversarial_replay_manifest.v1
campaign_id: robot_sf_1488_two_family_v1
failure_id: "<canonical_failure_id>"
source_candidate_id: "<candidate_id>"
failure_signature: "<signature_hash>"
scenario_family: "<family_id>"
planner_row: "<planner_id>"
search_engine: "<engine_id>"
campaign_seed: 1001

candidate_parameters:
  route_id: "<id>"
  start_state: {}
  human_agent_parameters: {}
  environment_parameters: {}

replay:
  repetitions: 3
  replay_seeds: ["<seed1>", "<seed2>", "<seed3>"]
  expected_status: valid_behavioral_failure
  expected_failure_mode: "<mode>"
  metric_tolerances: {}

artifacts:
  compact_summary_path: "<path>"
  checksum_sha256: "<sha>"
  raw_artifact_uri: null
  raw_artifact_checksum_sha256: null

result:
  replay_outcomes:
    - repetition: 1
      status: valid_behavioral_failure
      metrics: {}
    - repetition: 2
      status: valid_behavioral_failure
      metrics: {}
    - repetition: 3
      status: valid_behavioral_failure
      metrics: {}
  replay_classification: replayable
```

### Compact Artifact Bundle

Recommended tracked bundle:

```text
docs/context/evidence/issue_1488/
  campaign_manifest.yaml
  campaign_summary.md
  run_summary.csv
  status_counts.csv
  failure_index.csv
  replay_manifest.yaml
  stress_uncertainty_coverage.v1.json
  checksums.sha256
  claim_boundary.md
```

### Deterministic Seed Handling

Use one seed derivation function and write it into both code and manifest:

```python
def derive_seed(campaign_id: str, cell_id: str, role: str, index: int) -> int:
    payload = f"{campaign_id}:{cell_id}:{role}:{index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)
```

Every candidate should record:

```text
campaign_seed
search_seed
scenario_seed
simulator_seed
planner_seed
candidate_index
candidate_id
```

### Result Table Fields

Minimum `run_summary.csv` columns:

```text
campaign_id
cell_id
scenario_family
planner_row
search_engine
campaign_seed
candidate_index
candidate_id
status
valid_trial
failure_detected
failure_mode
failure_signature
duplicate_of
replay_selected
replay_status
severity_score
min_clearance_m
min_ttc_s
collision
proxemic_violation
goal_reached
completion_time_s
path_length_m
mean_speed_mps
max_accel_mps2
max_jerk_mps3
sim_time_s
wall_clock_s
invalid_reason
simulation_error_class
fallback_degraded_reason
artifact_checksum
```

## Recommended Issue Edits

### Parent Issue `#1488`

Add a primary denominator clause:

```markdown
Primary comparison denominator: attempted candidates per
(scenario_family, planner_row, search_engine, campaign_seed).
Invalid candidates, simulation errors, unavailable rows, and fallback/degraded rows
remain in the attempted-candidate denominator and are reported separately.
Valid-candidate and simulation-hour normalizations are secondary only.
```

Add a primary evidence clause:

```markdown
Primary failure evidence is limited to replayable, non-duplicate,
valid behavioral failures. First-pass non-replayable failures and duplicate
failures are reported but not counted as primary unique failures.
```

Add a no optimizer expansion clause:

```markdown
The first campaign is limited to seeded random, Optuna/TPE-style,
and existing guided route/start-state search. AST, STL falsification,
MAP-Elites, LLM-guided generation, and new fuzzers are literature-grounded
future extensions, not first-campaign search engines.
```

### Child Issue `#1501`

Add smoke-specific acceptance thresholds:

```markdown
Smoke acceptance requires all three search engines to execute from the frozen
manifest with status accounting for every attempted candidate. A smoke run may
pass with zero discovered failures if manifest execution, status accounting,
coverage report generation, and replay plumbing are verified.
```

Add failure criteria:

```markdown
Revise or reject if any engine lacks attempted-candidate accounting,
if fallback/degraded rows are aggregated as success, if simulation_error rows
are counted as failure evidence, or if valid_trial_rate < 50% without diagnosis.
```

### Child Issue `#1502`

Add explicit campaign size:

```markdown
Default bounded comparison:
2 scenario families x 2 planner rows x 3 search engines x 3 campaign seeds
x 30 attempted candidates per cell = 1080 attempted candidates.
If runtime permits, increase only attempted candidates per cell to 50.
```

Add replay requirement:

```markdown
Replay all failures if few are found; otherwise replay top-5 by severity and
top-5 additional diversity representatives per search_engine x scenario_family x
planner_row. Primary evidence requires 3/3 replay agreement under frozen
metric tolerances.
```

Add budget guard:

```markdown
No early stop after first failure. Budget exhaustion, manifest invalidity,
or fail-closed infrastructure abort are the only stop conditions.
```

### Child Issue `#1503`

Add required report sections:

```markdown
The synthesis report must include:
1. status-count table by engine/family/planner/seed,
2. cumulative unique replayable failure curves,
3. failures per attempted candidate, per valid candidate, and per simulation-hour,
4. failure-signature diversity table,
5. replay determinism table,
6. candidate/valid/failure coverage by stress bin,
7. uncertainty intervals over seeds,
8. allowed and disallowed claim statements.
```

Add duplicate rule:

```markdown
Duplicate failures are grouped by deterministic failure_signature and are
reported separately from unique replayable failures.
```

Add caveat rule:

```markdown
The report must state that adversarial-search samples are biased stress/probing
samples and must not be interpreted as nominal benchmark coverage or real-world
failure probability.
```

## Claim Boundary

Allowed:

> Under a fixed attempted-candidate budget, guided route/start-state search discovered more
> replayable unique behavioral failures than seeded random search in the tested Robot SF scenario
> families.

Allowed:

> In the bounded two-family campaign, TPE produced a higher replayable-unique-failure rate per
> attempted candidate than random search, but the confidence interval over seeds was wide.

Allowed:

> The campaign found stress evidence for specific planner-row weaknesses under the declared
> scenario families, seeds, and simulator configuration.

Allowed:

> Invalid-candidate and simulation-error rates differed across search engines and were included in
> the primary denominator.

Allowed:

> Stress coverage was incomplete; uncovered bins are listed and no nominal benchmark generality is
> inferred.

Allowed:

> Some first-pass failures were excluded from primary evidence because they were non-replayable or
> duplicate failures.

Disallowed:

> Guided search proves the planner is unsafe in general.

Disallowed:

> TPE is universally better than random search for social-navigation validation.

Disallowed:

> The adversarial campaign replaces the nominal Robot SF benchmark.

Disallowed:

> No discovered failure implies the planner is safe.

Disallowed:

> Simulation-error rows are evidence of planner failure.

Disallowed:

> Fallback/degraded execution demonstrates benchmark success.

Disallowed:

> Failure rate under adversarial search estimates real-world failure probability.

Disallowed:

> A small two-family campaign covers the Robot SF operating domain.

The issue is strong enough for a methodology-paper contribution if framed as:

```text
A bounded, replay-backed, status-accounted adversarial stress-search protocol
for social-navigation benchmark evidence.
```

The core contribution would be:

```text
not a new optimizer,
but a conservative evaluation protocol that compares stress-search methods
without denominator leakage, replay leakage, or nominal-claim leakage.
```

Defensible claims after the compact campaign:

```text
sample-efficiency comparison under fixed budget
replayable unique failure evidence
status-accounted invalid/simulation-error rates
stress-axis coverage and uncovered regions
planner ranking sensitivity under stress evidence
```

Out-of-scope claims:

```text
general planner safety
real-world safety assurance
ODD-level failure probability
optimizer superiority beyond tested families
human acceptance of the navigation behavior
nominal benchmark replacement
```

## References

- James Bergstra and Yoshua Bengio. "Random Search for Hyper-Parameter Optimization." Journal of
  Machine Learning Research, 2012.
  <https://jmlr.org/papers/v13/bergstra12a.html>
- James Bergstra, Remi Bardenet, Yoshua Bengio, and Balazs Kegl. "Algorithms for Hyper-Parameter
  Optimization." NeurIPS, 2011.
  <https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization>
- Optuna contributors. "optuna.samplers.TPESampler."
  <https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html>
- Mark Koren, Saud Alsaif, Ritchie Lee, and Mykel J. Kochenderfer. "Adaptive Stress Testing for
  Autonomous Vehicles." IEEE Intelligent Vehicles Symposium / arXiv, 2018/2019.
  <https://arxiv.org/abs/1902.01909>
- Ritchie Lee, Ole J. Mengshoel, Anshu Saksena, Ryan W. Gardner, Daniel Genin, Joshua Silbermann,
  Michael Owen, and Mykel J. Kochenderfer. "Adaptive Stress Testing: Finding Likely Failure Events
  with Reinforcement Learning." Journal of Artificial Intelligence Research, 2020.
  <https://www.jair.org/index.php/jair/article/view/12190>
- Robert J. Moss. "POMDPStressTesting.jl: Adaptive Stress Testing for Black-Box Systems." Journal
  of Open Source Software, 2021.
  <https://www.theoj.org/joss-papers/joss.02749/10.21105.joss.02749.pdf>
- Anthony Corso, Robert J. Moss, Mark Koren, Ritchie Lee, and Mykel J. Kochenderfer. "A Survey of
  Algorithms for Black-Box Safety Validation of Cyber-Physical Systems." arXiv, 2021.
  <https://arxiv.org/abs/2005.02979>
- Yashwanth Annpureddy, Che Liu, Georgios Fainekos, and Sriram Sankaranarayanan. "S-TaLiRo: A Tool
  for Temporal Logic Falsification for Hybrid Systems." TACAS, 2011.
  <https://home.cs.colorado.edu/~srirams/papers/sTaliro-tacas11.pdf>
- Alexandre Donze. "Breach, A Toolbox for Verification and Parameter Synthesis of Hybrid Systems."
  Computer Aided Verification, 2010.
  <https://link.springer.com/chapter/10.1007/978-3-642-14295-6_17>
- Tommaso Dreossi, Daniel J. Fremont, Shromona Ghosh, Edward Kim, Hadi Ravanbakhsh, Marcell
  Vazquez-Chanlatte, and Sanjit A. Seshia. "VERIFAI: A Toolkit for the Design and Analysis of
  Artificial Intelligence-Based Systems." CAV / arXiv, 2019.
  <https://arxiv.org/abs/1902.04245>
- Daniel J. Fremont, Tommaso Dreossi, Shromona Ghosh, Xiangyu Yue, Alberto L.
  Sangiovanni-Vincentelli, and Sanjit A. Seshia. "Scenic: A Language for Scenario Specification
  and Scene Generation." PLDI / arXiv, 2019.
  <https://arxiv.org/abs/1809.09310>
- Guanpeng Li, Yiran Li, Saurabh Jha, Timothy Tsai, Michael Sullivan, Siva Kumar Sastry Hari,
  Zbigniew Kalbarczyk, and Ravishankar Iyer. "AV-FUZZER: Finding Safety Violations in Autonomous
  Driving Systems." IEEE ISSRE, 2020.
  <https://research.nvidia.com/publication/2020-10_av-fuzzer-finding-safety-violations-autonomous-driving-systems>
- Seulbae Kim, Major Liu, Junghwan Rhee, Yuseok Jeon, Yonghwi Kwon, and Chung Hwan Kim.
  "DriveFuzz: Discovering Autonomous Driving Bugs through Driving Quality-Guided Fuzzing." ACM CCS
  / arXiv, 2022.
  <https://arxiv.org/abs/2211.01829>
- Kexin Pei, Yinzhi Cao, Junfeng Yang, and Suman Jana. "DeepXplore: Automated Whitebox Testing of
  Deep Learning Systems." SOSP / arXiv, 2017.
  <https://arxiv.org/abs/1705.06640>
- Yuchi Tian, Kexin Pei, Suman Jana, and Baishakhi Ray. "DeepTest: Automated Testing of
  Deep-Neural-Network-driven Autonomous Cars." ICSE, 2018.
  <https://dl.acm.org/doi/10.1145/3180155.3180220>
- Tahereh Zohdinasab, Vincenzo Riccio, Alessio Gambi, and Paolo Tonella. "DeepHyperion: Exploring
  the Feature Space of Deep Learning-Based Systems through Illumination Search." ISSTA / arXiv,
  2021.
  <https://arxiv.org/abs/2107.06997>
- Jean-Baptiste Mouret and Jeff Clune. "Illuminating search spaces by mapping elites." arXiv, 2015.
  <https://arxiv.org/abs/1504.04909>
- Joel Lehman and Kenneth O. Stanley. "Abandoning objectives: Evolution through the search for
  novelty alone." Evolutionary Computation, 2011.
  <https://dl.acm.org/doi/10.1162/EVCO_a_00025>
- Peter Du and Katherine Driggs-Campbell. "Finding Diverse Failure Scenarios in Autonomous Systems
  Using Adaptive Stress Testing." SAE International Journal of Connected and Automated Vehicles,
  2019.
  <https://saemobilus.sae.org/downloads/articles/12-02-04-0018/Full%20Text%20PDF>
- Abhijat Biswas, Allan Wang, Glenn Silvera, Aaron Steinfeld, and Henny Admoni. "SocNavBench: A
  Grounded Simulation Testing Framework for Evaluating Social Navigation." ACM THRI, 2022.
  <https://harplab.github.io/assets/pubs/biswas_thri21.pdf>
- Yao Gao and collaborators. "Evaluation of Socially-Aware Robot Navigation." 2022.
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC8791647/>
- Thibault Kruse, Amit Kumar Pandey, Rachid Alami, and Alexandra Kirsch. "Human-Aware Robot
  Navigation: A Survey." Robotics and Autonomous Systems, 2013.
  <https://hal.science/hal-01684295v1/document>
- Lukas Westhofen, Christian Neurohr, Tjark Koopmann, Martin Butz, Barbara Schutt, Fabian Utesch,
  Birte Neurohr, Christian Gutenkunst, and Eckard Bode. "Criticality Metrics for Automated Driving:
  A Review and Suitability Analysis of the State of the Art." Archives of Computational Methods in
  Engineering / arXiv, 2023.
  <https://arxiv.org/abs/2108.02403>
- Stefan Riedmaier, Thomas Ponn, Dieter Ludwig, Bernhard Schick, and Frank Diermeyer. "Survey on
  Scenario-Based Safety Assessment of Automated Vehicles." IEEE Access, 2020.
  <https://opus4.kobv.de/opus4-hs-kempten/files/703/Survey_on_Scenario-based_Safety_Assessment.pdf>
- ASAM e.V. "ASAM OpenSCENARIO DSL."
  <https://www.asam.net/standards/detail/openscenario-dsl/>
- Yuan Gao, Mattia Piccinini, Korbinian Moller, Amr Alanwar, and Johannes Betz. "From Words to
  Collisions: LLM-Guided Evaluation and Adversarial Generation of Safety-Critical Driving
  Scenarios." arXiv, 2025.
  <https://arxiv.org/abs/2502.02145>
- Per Runeson and Martin Host. "Guidelines for conducting and reporting case study research in
  software engineering." Empirical Software Engineering, 2009.
  <https://d-nb.info/1288443331/34>
- ACM SIGSOFT. "Empirical Standards for Software Engineering."
  <https://www2.sigsoft.org/EmpiricalStandards/>
- Roberto Verdecchia, Patricia Lago, Ivano Malavolta, and colleagues. "Threats to validity in
  software engineering research." Information and Software Technology, 2023.
  <https://www.sciencedirect.com/science/article/abs/pii/S0950584923001842>
