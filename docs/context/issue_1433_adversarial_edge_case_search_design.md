# Issue #1433 Bounded Adversarial Edge-Case Scenario Search Design (2026-05-22)

Date: 2026-05-22

Related issues:

- <https://github.com/ll7/robot_sf_ll7/issues/1433> (this design)
- <https://github.com/ll7/robot_sf_ll7/issues/1434> (uncertainty / coverage reporting)
- Predecessors:
  [issue_1236_optimizer_adversarial_sampler.md](issue_1236_optimizer_adversarial_sampler.md),
  [issue_1237_adversarial_failure_archive.md](issue_1237_adversarial_failure_archive.md),
  [issue_1240_scenario_coverage_entropy.md](issue_1240_scenario_coverage_entropy.md)

## Goal

Design a bounded v1 adversarial edge-case search that turns the existing crossing/TTC template and
search-space configs into a reproducible, budget-constrained stress-test generator, without
promoting generated cases to benchmark evidence and without waiting for full uncertainty/coverage
reporting infrastructure.

## What Can Be Piloted Before Issue #1434

The following surfaces are stable enough to pilot today because they depend only on authored
scenario/config fields and deterministic simulation replay:

1. **Parametric candidate generation** within the existing `adversarial-search-space.v1` bounds.
2. **Invalid-candidate filtering** using `min_start_goal_distance_m` and search-space scalar clamps.
3. **Scripted adversarial pedestrian decisions** (fixed speed, delay, and spawn-time overrides).
4. **Deterministic replay bundles** with recorded budget, seed, objective, and replay metadata.
5. **Failure-mode classification** into a small, stable taxonomy rooted in current
   `failure_attribution.primary_failure` values.
6. **Compact archive manifests** per `adversarial_failure_archive.v1` that point back to source
   candidate bundles without copying raw JSONL or video.

## What Depends on Issue #1434

The following capabilities require uncertainty quantification or coverage entropy reporting and
**must not block** the v1 pilot:

1. **Adaptive stopping rules** based on coverage-entropy saturation or novelty-threshold triggers.
2. **Learned adversarial pedestrian decisions** (policy-conditioned or model-based) because
   training/selection requires a coverage baseline to avoid overfitting to a narrow perturbation
   subspace.
3. **Automated scenario promotion** from development stress tests to certified benchmark candidates,
   because promotion requires evidence that the generated case fills a genuine coverage gap rather
   than repeating a known failure cluster.
4. **Cross-search comparison claims** (e.g., "optuna finds failures in fewer evaluations than
   random") on real benchmark surfaces, because statistical efficiency claims need confidence
   intervals and coverage fields whose report contract belongs in Issue #1434.

## Parameter Bounds

The v1 search space is the existing `configs/adversarial/crossing_ttc_space.yaml`:

| Variable | Min | Max | Unit | Rationale |
|---|---|---|---|---|
| `start_x` | 1.0 | 3.0 | m | Pedestrian spawn lateral range on the near side of the crossing trap |
| `start_y` | 2.0 | 4.0 | m | Pedestrian spawn longitudinal range before the robot path |
| `goal_x` | 7.0 | 9.0 | m | Pedestrian goal lateral range on the far side |
| `goal_y` | 2.0 | 4.0 | m | Pedestrian goal longitudinal range after crossing |
| `spawn_time_s` | 0.0 | 2.0 | s | When the adversarial pedestrian enters the scene |
| `pedestrian_speed_mps` | 0.8 | 1.4 | m/s | Adversarial pedestrian walking speed |
| `pedestrian_delay_s` | 0.0 | 2.0 | s | Extra delay before the pedestrian begins moving |
| `scenario_seed` | 100 | 999 | int | Deterministic simulation seed |

**Hard constraint:** `min_start_goal_distance_m: 2.0` rejects any candidate whose Euclidean
start-goal distance falls below this bound. Samplers must still emit the candidate so the failure
is recorded as an invalid trial; the bound must not silently clip coordinates.

**V1 boundary:** These bounds are frozen for the pilot. Expanding them (e.g., multi-pedestrian,
obstacle-conditioned, or map-variant extensions) is deferred to a later implementation issue.

## Invalid-Candidate Handling

The v1 contract reuses the invalid-trial pattern from #1236:

- **Search-space violations** (out-of-bounds scalars) are rejected at proposal time with a clear
  validation reason and recorded as an `invalid_candidate` trial.
- **Constraint violations** (e.g., `min_start_goal_distance_m`) are evaluated as a failed trial
  with objective `None` and `termination_reason: invalid_candidate`.
- **Simulator exceptions** during reset or step (e.g., spawn collision, unreachable goal) are caught,
  logged, and recorded as `termination_reason: simulation_error`.
- Invalid candidates must **never** be promoted into the failure archive; they are retained in the
  raw manifest for debugging and budget-audit purposes only.

## Scripted vs Learned Adversarial Pedestrian Decision

**V1 scope is scripted only.**

The adversarial pedestrian behavior is fully determined by the search-space parameters above:
fixed speed, fixed delay, fixed spawn time, and a straight-line goal. There is no inner-loop
policy or learned trajectory optimizer.

**Rationale:**

- A learned adversarial pedestrian requires a coverage baseline (Issue #1434) to ensure the learned
  policy explores the intended failure modes rather than memorizing a single low-diversity
  collision trajectory.
- The existing `crossing_ttc_template` (`configs/scenarios/templates/crossing_ttc.yaml`) is already
  marked `behavior: generated_stress_test` and `generated_cases_are_benchmark_evidence: false`,
  which is the correct container for scripted adversarial candidates.
- Keeping the pedestrian decision scripted preserves deterministic replay: the same candidate spec
  + seed always produces the same episode trace.

**Deferred boundary:** A learned adversarial-decision follow-up can be scoped once Issue #1434
defines the uncertainty/coverage fields needed to judge whether a training distribution is
representative. Until then, any "adversarial pedestrian policy" work remains an experimental spike
with explicit non-benchmark marking.

## Execution Contract

Every v1 adversarial search run must record the following metadata in the emitted manifest
(`adversarial-search-manifest.v1`):

| Field | Type | Description |
|---|---|---|
| `budget` | int | Total candidate evaluations allowed for this search |
| `seed` | int | Global search seed (separate from per-candidate `scenario_seed`) |
| `objective` | string | Objective name, e.g., `min_snqi` or `max_collision_indicator` |
| `scenario_template` | string | Path to the base template YAML |
| `search_space` | string | Path to the search-space YAML |
| `sampler` | string | Sampler class name, e.g., `RandomCandidateSampler`, `OptunaCandidateSampler` |
| `replay_metadata` | object | `command`, `commit`, `timestamp`, `uv_lock_hash` (if available) |
| `start_time_iso` | string | ISO-8601 start timestamp |
| `end_time_iso` | string | ISO-8601 end timestamp |

**Budget rule:** v1 runs are capped at `budget <= 100` for local development and `budget <= 1000`
for preflight SLURM jobs. Anything larger requires a maintainer decision and a separate issue.

**Seed rule:** The global search seed controls sampler determinism; each candidate still receives
its own `scenario_seed` from the search space. This two-level seeding lets researchers vary the
search trajectory while keeping individual candidates replayable.

## Failure-Mode Classes

V1 uses the existing `failure_attribution` vocabulary. The primary classes expected from the
crossing/TTC search are:

1. **`collision`** — physical contact between robot and adversarial pedestrian.
2. **`near_miss`** — proximity below a configured threshold without contact.
3. **`timeout`** — episode reaches `max_episode_steps` without goal reach.
4. **`comfort_violation`** — force-exposure or jerk exceeds a threshold (non-collision).
5. **`invalid_candidate`** — search-space or constraint violation (not an episode failure).
6. **`simulation_error`** — spawn or runtime exception (not an episode failure).

**Archive eligibility:** Only candidates with `primary_failure` in {`collision`, `near_miss`,
`timeout`, `comfort_violation`} and a valid objective score may enter the
`adversarial_failure_archive.v1`. Invalid candidates and simulation errors remain in the raw
manifest for budget auditing but are excluded from clustering and representative selection.

## Artifact Policy

The artifact policy is inherited from #1237 with one v1 addition:

- **Raw candidate bundles** (episode JSONL, trajectories, videos) live in `output/` and are
  git-ignored.
- **Search manifests** (`adversarial-search-manifest.v1`) are also git-ignored unless promoted as
  small, reviewable evidence copies into `docs/context/evidence/`.
- **Failure archive** (`adversarial_failure_archive.v1`) is a compact JSON manifest (no raw
  episodes) and may be promoted into `docs/context/evidence/` when it supports a design decision
  or bug report.
- **Design docs** (this note) are tracked in git and are the durable record of scope and
  boundaries.

**V1 rule:** Do not commit raw search outputs. If a generated case is needed for a bug
reproduction, record the candidate spec + seed + replay command in the issue or context note
instead of checking in the JSONL.

## Fail-Closed Benchmark-Success Boundary

Generated adversarial cases are **explicitly not benchmark evidence**.

- The template config marks `generated_cases_are_benchmark_evidence: false`.
- Any claim that an adversarial search result "strengthens the benchmark" or "proves planner
  robustness" is out of scope for v1.
- If a search run produces a candidate that looks like a useful new scenario, the correct next
  step is manual scenario certification (`scenario_cert.v1`) via a separate issue, not automatic
  promotion from the adversarial archive.
- Benchmark reports that include adversarial search rows must label them as
  `mode: development_stress_test` and must not mix them with certified benchmark distributions.

This boundary prevents the adversarial search from inadvertently distorting benchmark-success
metrics by injecting non-representative, hand-optimized failure cases into a claim distribution.

## Dependency Relationship to Issue #1434

Issue #1434 (uncertainty / coverage reporting) is a **sibling dependency**, not a parent blocker.

- The v1 pilot can execute, archive, and classify failures without Issue #1434.
- Issue #1434 becomes **required** before any of the "Depends on Issue #1434" capabilities above are
  implemented.
- When Issue #1434 lands, the v1 design should be updated to reference the new uncertainty/coverage
  reporting contract and to add adaptive-stopping or learned-decision extensions in a follow-up
  implementation issue.
- Until then, v1 search results must carry an explicit caveat in any report or note:
  "Coverage and uncertainty metrics are not yet available; results are exploratory stress tests
  only."

## Follow-Up Boundary for Later Implementation Issue

A future issue (not created now) should scope the v2 expansion once Issue #1434 is available:

- Multi-pedestrian adversarial candidates (building on #923/#936/#944 multi-ped schema work).
- Learned adversarial pedestrian decisions with coverage-entropy regularization.
- Adaptive budget stopping based on failure-mode saturation or coverage novelty.
- Cross-sampler statistical comparison on real benchmark surfaces with confidence intervals.
- Scenario-certification handoff path: from archived failure to `scenario_cert.v1` promotion.

That follow-up issue should reference this design note, the Issue #1434 coverage contract, and the
existing multi-ped adversarial runtime notes (#870, #1015).
