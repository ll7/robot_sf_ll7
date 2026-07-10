# Issue #4933 — Virtual-Validation Mitigation Techniques — Feasibility and Effort Assessment

This assessment evaluates five mitigation techniques from the autonomous-driving virtual-validation
literature, mapped onto robot_sf_ll7's existing infrastructure. For each technique: what it adds,
which components it reuses, estimated effort, and a recommendation.

## Summary Table

| # | Technique | Effort | Recommendation |
|---|-----------|--------|----------------|
| 1 | Statistical robustness / repeatability | **S** | **Do now** — standardize existing distributional reporting |
| 2 | Fidelity & calibration | **L** | **Later** — blocked on external reference data |
| 3 | Accelerated stress testing | **M** | **Do now** — rare-event sampling exists; orchestration gap only |
| 4 | Uncertainty quantification | **M** | **Later** — stubs exist; needs learned-component integration |
| 5 | Hybrid testing (replay + perturbation) | **L** | **Later** — CARLA replay exists; native perturbation replay is new |

---

## 1. Statistical Robustness / Repeatability

### What it adds
Repeated seeded runs with distributional reporting instead of single pass/fail. Makes benchmark
claims reproducible and exposes seed variance that single-run reporting hides.

### Existing components reused
- **`robot_sf/benchmark/camera_ready/campaign.py`**: already computes `seed_variability_records`
  and `seed_variability_by_scenario` across multi-seed runs.
- **`scripts/benchmark/build_headline_ci_rank_stability_report_issue_3216.py`**: rank-stability
  analysis across seeds.
- **`robot_sf/benchmark/reactivity_ablation.py`**: paired-seed common-random-numbers design.
- **`scripts/tools/seed_sufficiency_gate.py`**: post-run CI half-width gate for seed count.
- **`robot_sf/research/statistics.py`**: distributional helpers for research reports.

### What is missing
Standardization of the distributional reporting surface. Currently each campaign type
(camera-ready, fidelity-sensitivity, reactivity-ablation) computes seed variability differently.
A unified `seed_distribution_report.v1` schema and a single report builder would close the gap.

### Effort: **S**
Mostly glue: define the schema, write a thin adapter that normalizes per-campaign seed data into
the common format, and update the leaderboards to show CI alongside point estimates.

### Recommendation: **Do now**
Low effort, high payoff. Makes every existing multi-seed campaign more trustworthy without
changing any simulation code. Natural first slice.

---

## 2. Fidelity & Calibration

### What it adds
Diversified pedestrian/road-user behavior parameters and tuned sim/sensor parameters against
reference measurements, with explicit calibration-status reporting.

### Existing components reused
- **`robot_sf/benchmark/pedestrian_flow_validation.py`**: diagnostic-only pedestrian flow
  validation with descriptive metrics (issue #3971).
- **`robot_sf/research/ped_model_assumption_inventory.py`**: tracks which pedestrian model
  assumptions are synthetic-only vs. calibrated.
- **`robot_sf/research/amv_command_response_trace_manifest.py`**: AMV calibration-ingest
  contract, explicitly blocked on external data.
- **`robot_sf/planner/stream_gap_gate_calibration.py`**: stream-gap calibration module.
- **`robot_sf/representation/scenario_belief.py`**: exposes `calibration_status` field.
- **`scripts/benchmark/run_ped_model_sensitivity_issue_3950.py`**: pedestrian model
  sensitivity sweep.
- **`scripts/benchmark/build_fidelity_sensitivity_smoke_report.py`**: fidelity sensitivity
  reporting.

### What is missing
- **External reference data**: the AMV trace manifest explicitly states calibration is "blocked
  on external data." Real-world pedestrian trajectory datasets, sensor noise profiles, or
  field-measured road-user behavior parameters are not staged.
- **Calibration pipeline**: no end-to-end ingest → fit → report → validate calibration exists;
  the current modules are inventories and staging contracts only.
- **Pedestrian behavior diversity**: the scenario configs support archetype variation but there
  is no calibrated distribution over archetype parameters.

### Effort: **L**
The existing inventory and staging infrastructure is solid, but the actual calibration work
requires: (1) staging external reference data (trajectory datasets, sensor profiles), (2)
building a calibration pipeline that fits sim parameters to reference data, and (3) reporting
calibration status alongside benchmarks. Steps 1–2 are blocked on data acquisition and are
genuinely large.

### Recommendation: **Later — blocked on external data**
The infrastructure scaffolding exists, but meaningful calibration is impossible without real
reference data. Promote the existing inventories and staging contracts; do not build a
calibration pipeline until reference data is staged.

---

## 3. Accelerated Stress Testing (Importance / Rare-Event Sampling)

### What it adds
Bias sampling toward the failure boundary so critical episodes appear far more often per compute
unit than naive random. Overlaps with the scenario-generation epic (#4932).

### Existing components reused
- **`robot_sf/benchmark/rare_event_sampling.py`**: importance-sampling primitives including
  `ParameterDistribution`, likelihood-ratio bookkeeping, scenario mutation, and estimator math
  (schema `rare_event_sampling.v1`). This is the core engine.
- **`scripts/benchmark/run_rare_event_estimation_issue_4163.py`**: CLI entry point for
  rare-event smoke runs.
- **`configs/benchmarks/rare_event/`**: YAML specs for crossing and static-constriction
  rare-event scenarios.
- **`robot_sf/scenario_certification/feasibility_diagnostics.py`**: feasibility checks ensure
  mutated scenarios remain valid.
- **`scripts/benchmark/run_scenario_criticality_optimization.py`**: criticality search as
  baseline for learned/adaptive stress-testing approaches.

### What is missing
- **Orchestration layer**: the rare-event sampling engine exists but is a smoke harness only.
  A full campaign runner that: (a) generates proposal scenarios, (b) runs them through the
  benchmark runner, (c) feeds outcomes back into the importance sampler, and (d) produces a
  failure-rate estimate with proper confidence intervals is not yet built.
- **Integration with scenario-generation epic (#4932)**: the rare-event sampler should be a
  pluggable sampling strategy within the scenario-generation loop, not a standalone script.
- **Failure-rate estimation discipline**: the current estimator is a smoke proof-of-concept;
  production use needs variance monitoring, stopping rules, and budget allocation.

### Effort: **M**
The core sampling engine exists. The gap is orchestration (plugging it into the benchmark runner
loop, adding adaptive budget allocation, and producing campaign-grade failure-rate reports).
This is a well-scoped integration task, not new algorithmic work.

### Recommendation: **Do now (orchestration slice) — coordinate with #4932**
The rare-event engine is ready. The next slice is a thin campaign runner that wraps the existing
primitives with the benchmark runner and produces a failure-rate report. This naturally slots
into #4932's stage 4 plan. Coordinate there to avoid duplication.

---

## 4. Uncertainty Quantification

### What it adds
Flag episodes where a learned component (predictor, policy, value function) extrapolates beyond
its training distribution. Serves as a fallback/monitoring trigger.

### Existing components reused
- **`robot_sf/nav/uncertainty_envelope.py`**: stub interface for conformal-prediction inflation
  policy; explicitly states "not conformal calibration or safety certification."
- **`robot_sf/planner/guarded_ppo.py`**: exposes `uncertainty_conformal_radius_m` and
  `calibration_metadata` fields; conformal buffer intrusion mode exists.
- **`robot_sf/planner/safety_shield.py`**: carries `calibration_status` metadata.
- **`robot_sf/representation/uncertainty_source_generalization.py`**: generalizes uncertainty
  sources across learned components.
- **`robot_sf/benchmark/forecast_conformal_pilot.py`**: conformal prediction pilot for
  forecast quality.
- **`scripts/benchmark/run_issue_4232_uncertainty_envelope_alpha_sweep.py`**: alpha sweep
  for uncertainty envelope parameters.
- **`robot_sf/nav/baseline_probabilistic_predictor.py`**: probabilistic predictor with
  explicit "no calibration claim" boundary.

### What is missing
- **Runtime OOD detection**: the existing components provide parameter-level uncertainty
  (conformal radius, alpha sweep) but no runtime out-of-distribution detector that flags
  episodes in real time.
- **Learned-component integration**: the uncertainty envelope is a planning-layer concept;
  hooking it into learned predictors (e.g., the diffusion policy, learned proposal model)
  requires per-component OOD detectors.
- **Calibration evidence**: every existing uncertainty component explicitly disclaims
  calibration. Runtime UQ is only useful after calibration, which circles back to technique 2.

### Effort: **M**
The stub infrastructure is substantial. The main work is: (1) defining what "extrapolation"
means for each learned component (predictor, policy), (2) implementing a lightweight OOD
detector per component, and (3) wiring the detector into the episode logger to flag episodes.
Each component detector is small, but there are multiple learned components to cover.

### Recommendation: **Later — after calibration infrastructure (technique 2) lands**
UQ without calibration is a false sense of security. The existing stubs are well-designed;
promote them to implementation once the calibration pipeline from technique 2 provides the
ground truth needed to validate OOD detectors.

---

## 5. Hybrid Testing (Replay + Amplification of Recorded Near-Misses)

### What it adds
Replay recorded near-miss scenarios and apply controlled perturbations (metamorphic-style) to
check behavior consistency. Amplify critical moments to stress-test planner robustness.

### Existing components reused
- **`robot_sf_carla_bridge/live_replay.py`**: T1 oracle live replay against CARLA server,
  including trajectory replay, metric extraction, and comparison.
- **`robot_sf_carla_bridge/export.py`**: T0 neutral replay export for scenario serialization.
- **`robot_sf/benchmark/reactivity_ablation.py`**: paired reactive/replay arms with
  common-random-numbers design; "replay" is robot→ped force-off in live sim.
- **`robot_sf/benchmark/reactivity_replay_preflight.py`**: plan-level preflight for
  reactivity-vs-replay rank study.
- **`robot_sf/adversarial/seed_sensitivity.py`**: perturbation sensitivity analysis across
  seed ranges.
- **`robot_sf/scenario_certification/perturbation_preflight.py`**: perturbation variant
  generation with variant seeds.
- **`scripts/benchmark/run_reactivity_ablation_campaign_issue_3573.py`**: reactivity
  ablation campaign runner.

### What is missing
- **Native near-miss replay**: the existing replay infrastructure is CARLA-specific (T0/T1
  export → CARLA replay). There is no native robot_sf replay that can take a recorded
  near-miss episode and re-run it with perturbations entirely within the robot_sf simulator.
- **Near-miss extraction pipeline**: no automated pipeline to identify near-miss episodes from
  campaign output and extract them as replayable sub-scenarios (overlaps with #4932's stage 2).
- **Metamorphic perturbation framework**: the perturbation infrastructure exists for scenario
  certification but is not wired into a metamorphic testing workflow (given scenario S, generate
  perturbed variants S', S'' and assert consistency).

### Effort: **L**
The CARLA replay path is mature, but native near-miss replay within robot_sf is a new
capability. Building: (1) a native episode recorder and replayer, (2) a near-miss extraction
pipeline, and (3) a metamorphic perturbation framework is a substantial multi-slice effort.

### Recommendation: **Later — after scenario-generation epic (#4932) stages 1–3 land**
The near-miss extraction pipeline is stage 2 of #4932. The metamorphic perturbation framework
builds on top of that. Do not start until #4932's extraction pipeline exists; then the
perturbation + replay work becomes a natural extension.

---

## Cross-Cutting Observations

1. **Technique 3 overlaps #4932**: coordinate there. The rare-event sampling engine
   (`robot_sf/benchmark/rare_event_sampling.py`) is the natural stage 4 implementation for
   that epic.

2. **Technique 2 blocks technique 4**: UQ needs calibration ground truth. Sequence: 2 → 4.

3. **Technique 5 depends on #4932**: near-miss extraction is #4932 stage 2. Sequence: #4932
   stages 1–3 → 5.

4. **Technique 1 is independent and cheap**: can proceed immediately with no dependencies.

5. **All techniques benefit from technique 1**: distributional reporting is the foundation
   that makes every other technique's evidence trustworthy.

## Recommended Implementation Order

1. **Technique 1 (statistical robustness)** — do now, S effort, no dependencies.
2. **Technique 3 (accelerated stress testing)** — do now, M effort, coordinate with #4932.
3. **Technique 2 (fidelity & calibration)** — later, L effort, blocked on external data.
4. **Technique 4 (uncertainty quantification)** — later, M effort, blocked on technique 2.
5. **Technique 5 (hybrid testing)** — later, L effort, blocked on #4932 stages 1–3.
