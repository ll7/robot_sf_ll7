# Issue #1542 Manuscript Claim Evidence Map

Related issues:

- <https://github.com/ll7/robot_sf_ll7/issues/1542>
- AMV actuation boundary update: <https://github.com/ll7/robot_sf_ll7/issues/1570>

Date: 2026-05-26

Latest AMV claim-boundary update: Issue #1570 on 2026-05-31.

AMV actuation evidence ladder:
[issue_2230_amv_actuation_evidence_ladder.md](issue_2230_amv_actuation_evidence_ladder.md).

## Purpose

This note maps current manuscript candidate claims to durable evidence, negative evidence, blockers,
and next actions. It is a conservative evidence-synthesis artifact, not a manuscript draft.

Each claim area reports: claim text, issue/PR/evidence links, evidence tier, metrics used, known
caveats, blocker/next action, and verdict.

Verdict categories:

- `supported`: durable evidence supports the claim with known caveats.
- `negative`: durable evidence contradicts or weakens the claim.
- `blocked`: explicit blocker prevents claim readiness.
- `insufficient`: evidence exists but is too weak/narrow for the claim.
- `not_ready`: evidence collection or validation is incomplete.

Observation-track metadata boundary: most historical benchmark evidence cited by this map predates
first-class `benchmark_track` and `track_schema_version` metadata. Unless an evidence row
explicitly records both fields, treat it as `legacy_track_unknown`: useful for within-surface
interpretation, but not safe to aggregate with track-aware results or compare across observation
tracks. See
[issue_1721_benchmark_track_metadata_audit.md](issue_1721_benchmark_track_metadata_audit.md).

## Claim Area 1: AMV Benchmark Evidence

### Candidate Claim

"Robot-SF provides a paired nominal/stress AMV protocol with primary baseline planners (`goal`,
`social_force`, `orca`) and broader baselines including learned planners (`ppo`,
`prediction_planner`) across differential-drive and omnidirectional kinematics."

### Evidence Links

- Primary protocol: [issue_1344_paired_amv_protocol_report.md](issue_1344_paired_amv_protocol_report.md)
- Primary evidence:
  [issue_1344_paired_amv_primary_2026-05-20/](evidence/issue_1344_paired_amv_primary_2026-05-20/)
- Broader baseline preflight:
  [issue_1353_broader_amv_preflight.md](issue_1353_broader_amv_preflight.md)
- Broader baseline evidence:
  [issue_1353_broader_amv_2026-05-26/](evidence/issue_1353_broader_amv_2026-05-26/)
- Synthetic actuation stress design:
  [issue_1546_amv_actuation_envelope_stress_slice.md](issue_1546_amv_actuation_envelope_stress_slice.md)
- Synthetic actuation implementation and smoke verdict:
  [issue_1556_amv_actuation_stress_slice.md](issue_1556_amv_actuation_stress_slice.md)
- Compact actuation smoke evidence:
  [issue_1569_amv_actuation_smoke_2026-05-27/](evidence/issue_1569_amv_actuation_smoke_2026-05-27/)
- Paper-facing actuation calibration gate: issue
  [#1559](https://github.com/ll7/robot_sf_ll7/issues/1559)
- Primary configs:
  - `configs/benchmarks/issue_1344_paired_nominal_v1_primary.yaml`
  - `configs/benchmarks/issue_1344_paired_stress_primary.yaml`
- Broader configs:
  - `configs/benchmarks/issue_1353_paired_nominal_v1_broader_baselines.yaml`
  - `configs/benchmarks/issue_1353_paired_stress_broader_baselines.yaml`
- Cross-kinematics config: `configs/benchmarks/paper_cross_kinematics_v1.yaml`

### Evidence Tier

`stress` + `full_matrix` (broader baselines)

Synthetic actuation slice tier: `diagnostic_smoke` / `not paper-facing`. The Issue #1569 compact
smoke proves that the synthetic actuation-envelope slice is executable locally and emits actuation
diagnostics, but it is not calibrated AMV evidence and does not upgrade this claim area to
paper-facing.

### Metrics Used

- Success rate
- Collision rate
- Near-miss rate (implicitly tracked, not durably preserved in all summaries)
- SNQI v3 (baseline, weights, diagnostics)
- Mean minimum distance
- AMV coverage dimensions (observed as unavailable/missing in Issue #1344 and Issue #1353
  nominal/stress)
- Synthetic actuation diagnostics for Issue #1556/Issue #1569 only: `command_clip_fraction`,
  `yaw_rate_saturation_fraction`, `signed_braking_peak_m_s2`, planner command-space/projection
  metadata, and compact scenario AMV rows.

### Key Results

**Primary protocol (Issue #1344):**

| Planner | Nominal success | Stress success | Nominal collisions | Stress collisions | Nominal SNQI | Stress SNQI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `goal` | 0.2500 | 0.0000 | 0.3333 | 0.2500 | -0.0967 | -0.1752 |
| `orca` | 0.2500 | 0.1667 | 0.0833 | 0.0764 | -0.2999 | -0.2466 |
| `social_force` | 0.0000 | 0.0000 | 0.0000 | 0.2500 | -1.0435 | -0.8591 |

**Broader baselines (Issue #1353, 2026-05-26):**

| Surface | Total runs | Successful rows | Core rows | Episodes | Campaign success | AMV coverage | SNQI contract |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| nominal | 8 | 7 | 3/3 | 84 | true (core-anchored) | warn | warn |
| stress | 8 | 7 | 3/3 | 1008 | true (core-anchored) | warn | fail |

- Nominal success rates (best rows): `ppo=0.3333`, `prediction_planner=0.3333`,
  `goal/orca/sacadrl/socnav_sampling=0.2500`, `social_force=0.0000`
- Stress success rates (best rows): `ppo=0.2222`, `orca=0.1667`, `socnav_sampling=0.1528`,
  `prediction_planner=0.0694`, `sacadrl=0.0208`, `goal/social_force=0.0000`
- `socnav_bench` is `not_available` in both surfaces (accepted unavailable row, not successful
  evidence)

**Cross-kinematics (Issue #1354 compact, 2026-05-26):**

- 9/9 successful rows, `benchmark_success=true`, `amv_coverage_status=pass`,
  `snqi_contract_status=warn`

**Synthetic actuation smoke (Issue #1569 with Issue #1572/Issue #1582 metadata contract):**

- Verdict: `compact smoke run`, local and non-paper-facing.
- Evidence bundle:
  [evidence/issue_1569_amv_actuation_smoke_2026-05-27/summary.json](evidence/issue_1569_amv_actuation_smoke_2026-05-27/summary.json)
  plus the Issue #1556 context note.
- Campaign contract status: `campaign_execution_status=completed`, `evidence_status=valid`,
  `successful_evidence_rows=3`, `accepted_unavailable_rows=0`, `unexpected_failed_rows=0`, and
  `fallback_or_degraded_rows=0`.
- Planner episode success remained `0.0000` for all three smoke rows (`goal`, `orca`,
  `social_force`), so the smoke is executable diagnostics, not a performance result.
- Command clipping was observed (`goal=0.0407`, `orca=0.1047`, `social_force=0.2346`), while
  `yaw_rate_saturation_fraction` stayed `0.0000` for all rows.
- Issue #1572/Issue #1582 closed the metadata-contract gap after the smoke by accepting slice-local
  synthetic AMV overrides with explicit synthetic provenance and planner command-space/projection
  metadata.
  Unknown, unavailable, fallback, degraded, and failed rows remain caveats and must not count as
  actuation-envelope success evidence.

### AMV Actuation Claim Boundary

| Evidence level | Current status | Claim use |
| --- | --- | --- |
| Synthetic diagnostics | Implemented by Issue #1556 using `amv-actuation-stress-v0` and `paper_facing: false`. | May describe a software stress diagnostic and artifact/provenance contract only. |
| Platform-class proxy | Issue #2001 accepts a TRL e-scooter source for longitudinal acceleration and braking/deceleration only. | May describe field-limited adjacent-platform proxy values, not full AMV truth or unsupported yaw/latency/update-rate fields. |
| Compact smoke evidence | Issue #1569 produced a valid local 45-episode smoke; Issue #1572/Issue #1582 later repaired the metadata contract for future summaries. | May say the diagnostic slice runs and emits actuation metrics locally; must also state that task success was 0.0 and the evidence is non-paper-facing. |
| Calibrated/paper-facing evidence | Blocked on Issue #1559, issue #1585, and real command-response trace issue #2000. | No paper-facing AMV actuation claim yet; no hardware, deployment, safety, or real-AMV envelope language from synthetic-only or proxy-only evidence. |

### Known Caveats

1. **AMV coverage gap**: Both Issue #1344 primary and Issue #1353 broader nominal/stress campaigns report
   `amv_coverage_status=warn`. Coverage summaries show `Observed = -` for all required AMV
   dimensions, meaning all required dimension values are missing from source scenario metadata. This
   is not partial coverage—it is zero observed coverage against required dimensions.
2. **SNQI contract status**: Issue #1353 stress reports `snqi_contract_status=fail`, preserving
   Issue #1344's caution. SNQI should not be promoted into paper-facing claims without an explicit
   claim-scope decision.
3. **Low absolute success**: Even the best broader-baseline stress row (`ppo=0.2222`) shows modest
   success; many rows remain at or near zero.
4. **SocNav unavailable**: `socnav_bench` is `not_available` in Issue #1353 nominal/stress because
   SocNavBench control-pipeline assets are missing. This is caveated evidence, not a successful
   benchmark row.
5. **Paper-facing status**: Issue #1344 campaigns are `paper_facing=false`. Issue #1353 campaigns use
   `paper_interpretation_profile=issue-1353-broader-amv-preflight`, explicitly non-paper-facing.
6. **Synthetic actuation boundary**: Issue #1556/Issue #1569 add a useful synthetic diagnostic
   slice, not a calibrated AMV actuation claim. Issue #1572/Issue #1582 resolved the compact
   metadata contract, but that contract only makes future diagnostics clearer; it does not provide a
   real hardware envelope, controller trace, or paper-facing calibration source.
7. **Runtime hotspot**: `prediction_planner` shows the longest runtime (`59.1s` nominal, `613.4s`
   stress).

### Blocker / Next Action

**Blocker:** AMV coverage gap (all required dimensions missing from scenario metadata).

**Next actions:**

1. Decide whether the AMV coverage gap is acceptable for the manuscript scope or whether scenario
   metadata must be enriched before claiming AMV coverage.
2. Resolve the SNQI contract status boundary: either accept `warn`/`fail` with explicit caveats or
   revise the SNQI contract before paper-facing promotion.
3. Determine whether the low absolute success rates are acceptable for comparative claims or whether
   a tighter success gate is needed.
4. Keep synthetic actuation diagnostics out of paper-facing claims until Issue #1559 identifies a
   durable calibration source/profile and validates calibrated-vs-synthetic separation.

### Verdict

`blocked` — AMV coverage gap, SNQI contract cautions, and missing calibrated actuation evidence
block direct paper-facing promotion. The Issue #1569 smoke plus Issue #1572/Issue #1582 metadata
repair make a future diagnostic or thesis-method subsection plausible, but only as explicitly
synthetic diagnostics; a paper-facing AMV actuation subsection remains blocked on Issue #1559.

---

## Claim Area 2: Adversarial Stress Testing

### Candidate Claim

"Robot-SF provides a bounded adversarial comparison campaign with frozen scenario families,
search engines, budgets, replay determinism checks, and fail-closed row classification."

### Evidence Links

- Manifest freeze: [issue_1500_adversarial_manifest.md](issue_1500_adversarial_manifest.md)
- Row classification:
  [issue_1500_adversarial_manifest/row_classification_report.md](evidence/issue_1500_adversarial_manifest/row_classification_report.md)
- Manifest config: `configs/adversarial/issue_1500_adversarial_comparison_manifest.v1.yaml`
- Parent umbrella: issue #1488
- Next execution stage: issue #1501 (one-family smoke)
- Follow-up synthesis: issue #1503 (coverage/stress reporting)

### Evidence Tier

`launch_packet` (specification artifact, not execution evidence)

### Metrics Used

Not applicable (manifest stage; no execution metrics).

Future execution will use:

- Row types: `valid_behavioral_failure`, `success`, `invalid_candidate`, `simulation_error`,
  `fallback`, `degraded`, `not_available`
- Failure modes: collision, near_miss, timeout, comfort_violation
- Replay determinism checks: manifest materialization, seed determinism, search trajectory

### Key Results

Frozen manifest preserves:

- **Scenario families (2)**: `crossing_ttc` (template-based), `classic_head_on_corridor`
  (route-based)
- **Search engines (3)**: `random`, `optuna_tpe` (both for crossing/TTC),
  `guided_route_search` (for head-on corridor)
- **Planner rows (2)**: `classic_global_theta_star` (native), `orca` (adapter)
- **Total campaign budget**: 84 candidate evaluations (local), 612 (SLURM)
- **Replay checks**: manifest materialization, seed determinism, search trajectory
- **Row classification contract**: 7 row types, 4 archive-eligible
  (`valid_behavioral_failure`), 3 exclusion classes (`fallback`, `degraded`, `not_available`)

### Known Caveats

1. **Non-execution stage**: This is a specification artifact. No simulations, search engines, or
   planners have been run at this stage.
2. **Unavailable-by-design rows**: Random and TPE search on `classic_head_on_corridor` are
   `not_available` (no CandidateSpec-compatible search space exists). Guided route search on
   `crossing_ttc` is `not_available` (route override paradigm incompatible with parametric
   template).
3. **Cross-family comparison caution**: Crossing/TTC and head-on corridor use different search
   paradigms (parametric CandidateSpec vs. route-level graph optimization). Direct cross-family
   comparison of absolute failure counts is not valid.
4. **Within-family comparison boundary**: Within crossing/TTC, random vs. TPE is the intended
   comparison axis, but this manifest does not claim observed superiority.
5. **Fallback policy enforcement**: Per
   [issue_691_benchmark_fallback_policy.md](issue_691_benchmark_fallback_policy.md), fallback and
   degraded rows are explicitly not benchmark evidence.
6. **Evidence class**: Generated adversarial cases are
   `generated_cases_are_benchmark_evidence: false` and carry
   `evidence_class: development_stress_test`.

### Blocker / Next Action

**Blocker:** Execution stage not started (issue #1501 pending).

**Next actions:**

1. Run the crossing/TTC family with random and TPE search engines (issue #1501).
2. Emit `adversarial-search-manifest.v1` with explicit row classification.
3. Archive failures into `adversarial_failure_archive.v1`.
4. Report per-row counts and determinism-check results.
5. Defer coverage/stress synthesis to issue #1503 after #1501 and #1502 complete.

### Verdict

`not_ready` — specification artifact frozen, but execution evidence unavailable until issue #1501
completes.

---

## Claim Area 3: CARLA/Native Replay Parity

### Candidate Claim

"Robot-SF CARLA bridge achieves oracle replay metric parity between Robot-SF native simulation
and CARLA live replay for certified T0 export payloads."

### Evidence Links

- Live replay implementation: [issue_1169_carla_live_replay.md](issue_1169_carla_live_replay.md)
- Live parity attempt: [issue_1430_carla_live_parity.md](issue_1430_carla_live_parity.md)
- Live parity evidence:
  [issue_1430_carla_live_parity_2026-05-21/](evidence/issue_1430_carla_live_parity_2026-05-21/)
- Setup smoke: [issue_1111_carla_setup_smoke_2026-05-18/](evidence/issue_1111_carla_setup_smoke_2026-05-18/)
- Parent epic: issue #872
- Follow-up diagnosis: issue #1437

### Evidence Tier

`failed` (live CARLA runtime reached, but robot actor spawn failed before oracle replay)

### Metrics Used

**Attempted (unavailable):**

- Oracle replay status
- Metric parity (collision, near-miss, success, min-distance)

**Achieved:**

- Live CARLA server/client connectivity
- Docker + NVIDIA runtime availability
- CARLA 0.9.16 server/client version match
- Static-geometry proxy spawn (post-#1329)

### Key Results

**#1169 implementation:**

- Added opt-in Docker-backed `live-replay` path for CARLA T1 oracle replay bridge.
- Fail-closed: robot actor + scripted pedestrian actors moved by oracle transforms, with bounded
  static-geometry support (axis-aligned rectangular polygon obstacles as CARLA static-prop proxies).
- Test-first proof: `84 passed` in `tests/carla_bridge`.

**#1430 live parity attempt:**

- Host: `imech156-u` (fallback; preferred `imech036` blocked by non-interactive SSH host-key
  verification)
- Docker daemon: available
- NVIDIA GPU/container runtime: available
- CARLA image: `carlasim/carla:0.9.16`, digest
  `sha256:aaf1df22702780ece072069e23d03c4879b002ae028c79744b09c4c7ddbae953`
- Client/server: CARLA 0.9.16 connected to `Carla/Maps/Town10HD_Opt`
- Scenario: `pr_promoted_planner_smoke` from certified #1111 manifest
- **Result:** `status: failed`, `mode: failed`, `reason: CARLA failed to spawn robot`
- Parity adapter: `status: unavailable` (all metric rows unavailable because CARLA replay status is
  `failed`)

### Known Caveats

1. **No metric parity**: The run failed closed before oracle replay metrics were produced. This is
   not Robot-SF/CARLA metric parity and not simulator transfer evidence.
2. **Boundary progress**: The post-#1329 static-geometry failure was not the observed blocker; the
   run progressed past blanket static-obstacle rejection and failed at robot actor spawning.
3. **Static-geometry limitation**: Currently limited to axis-aligned rectangular polygon proxies.
   Unsupported or malformed static obstacles fail closed (not silently ignored).
4. **No sensor/perception replay**: Not yet implemented.
5. **No benchmark-strength CARLA transfer**: Live replay path does not yet provide long-running
   campaign evidence or benchmark-strength transfer claims.
6. **Setup-only smoke**: CARLA image availability, server/client connectivity, and Docker/NVIDIA
   runtime checks are setup smoke, not benchmark success.

### Blocker / Next Action

**Blocker:** CARLA failed to spawn robot actor in the certified #1111 payload.

**Next actions:**

1. Diagnose why CARLA failed to spawn the robot actor in the certified payload (issue #1437).
2. Rerun the same live replay until it either reaches `oracle-replay` or fails closed with a
   narrower actor-spawn condition.
3. Do not treat failed replay, setup-only smoke, image availability, or server/client connectivity
   as benchmark success.

### Verdict

`blocked` — robot actor spawn failure prevents oracle replay and metric parity. This is real live
CARLA runtime evidence, but not the positive metric-parity criterion required by the parent epic.

---

## Claim Area 4: Predictive Planner v2

### Candidate Claim

"Obstacle-feature predictive planner improves closed-loop navigation success and safety over
baseline predictive planner."

### Evidence Links

- Negative audit: [issue_1543_predictive_v2_negative_audit.md](issue_1543_predictive_v2_negative_audit.md)
- Obstacle pipeline: [issue_1167_predictive_obstacle_pipeline.md](issue_1167_predictive_obstacle_pipeline.md)
- Same-seed evidence:
  [issue_1427_predictive_same_seed_handoff_2026-05-21/manifest.json](evidence/issue_1427_predictive_same_seed_handoff_2026-05-21/manifest.json)
- Baseline config: `configs/training/predictive/predictive_br07_same_seed_issue_1427.yaml`
- Obstacle-feature config:
  `configs/training/predictive/predictive_obstacle_features_same_seed_issue_1427.yaml`
- Shared seed manifest:
  `configs/training/predictive/predictive_same_seed_issue_1427_base_seed_manifest.yaml`
- Related PR: #1480

### Evidence Tier

`stress` (same-seed comparison, 23 scenarios / 92 seeds, shared hard-seed manifest)

### Metrics Used

- Success rate
- Mean minimum distance
- Hard-seed success (zero for both variants)
- Best planner-grid row
- ADE/FDE (forecast metrics; qualitative improvement noted, exact final values not durably
  summarized)

### Key Results

**Final #1427 same-seed comparison:**

| Variant | Success rate | Mean min distance | Hard success | Best planner-grid row | Global mean min distance |
| --- | ---: | ---: | ---: | --- | ---: |
| Baseline predictive | 0.1304 | 2.1931 | 0.0000 | `risk_aware_adaptive` | 2.1931 |
| Obstacle-feature predictive | 0.1014 | 2.2105 | 0.0000 | `baseline_like` | 2.2081 |

**Stage-gate outcome:**

- Both runs: `evaluation_ok: false`, `campaign_ok: true`, `hard_seed_diagnostics_ok: true`
- Both runs failed the success gate (`min_success_rate: 0.3`)

**Forecast improvement:**

- The obstacle-feature model improved validation forecast loss/clearance signals.
- ADE/FDE quality gates passed.
- Exact final ADE/FDE deltas are not durably summarized in git-tracked #1427 artifacts.

### Known Caveats

1. **Negative closed-loop result**: Obstacle-feature success worsened (`0.1304` → `0.1014`) despite
   forecast improvement. This is strong evidence for prediction-to-control coupling failure, not
   "no forecast improvement."
2. **Hard-seed failure**: Hard-seed success remained zero for both variants.
3. **Planner-grid interaction**: Best planner-grid row changed (`risk_aware_adaptive` →
   `baseline_like`), but both variants still failed the same success gate, so this is a caveat, not
   the primary explanation.
4. **Per-row metrics unavailable**: Git-tracked #1427 artifacts preserve the paired 23-scenario /
   92-seed schedule, but they do not preserve exact per-row collision, near-miss, or
   low-progress/timeout values.
5. **Same-seed contract removes seed-distribution excuses**: Same scenario matrix, same hard-seed
   manifest, same base seed manifest, same training budget, same evaluation surface. Seed-schedule
   mismatch is not the primary mechanism.

### Blocker / Next Action

**Blocker:** Negative same-seed evidence (obstacle-feature variant worsened closed-loop success
despite forecast improvement).

**Next actions:**

1. Revise the planner-side coupling or planner-aligned training objective before further expansion.
2. Do not proceed with the four-way predictive-v2 expansion (issue #1490) until a bounded preflight
   shows closed-loop improvement, not only forecast improvement.
3. Treat #1427 as evidence against promoting `predictive_obstacle_features_v1` as-is, not as
   support for immediate expansion.

### Verdict

`negative` — durable same-seed evidence contradicts the claim. The most likely mechanism is
prediction-to-control coupling failure. Forecast improvement did not transfer into better planner
choices.

---

## Claim Area 5: Hard-Guarded Hybrid Learning

### Candidate Claim

"Hard-guarded hybrid learning components (learned-risk model, ORCA-residual BC, shielded PPO
repair, oracle imitation warm-start) maintain guard authority while contributing measurable
auxiliary cost, bounded residual, or repair decisions."

### Evidence Links

- Hybrid evidence matrix schema:
  [issue_1499_hybrid_evidence_matrix_schema.md](issue_1499_hybrid_evidence_matrix_schema.md)
- Learned-risk launch packet:
  [issue_1395_learned_risk_launch_packet.md](issue_1395_learned_risk_launch_packet.md)
- Learned-risk launch evidence:
  [issue_1395_learned_risk_launch_packet_2026-05-24/](evidence/issue_1395_learned_risk_launch_packet_2026-05-24/)
- Parent synthesis: issue #1489
- Component issues: Issue #1470, Issue #1472, Issue #1474, Issue #1475, Issue #1358, Issue #1496
- Fallback policy:
  [issue_691_benchmark_fallback_policy.md](issue_691_benchmark_fallback_policy.md)

### Evidence Tier

`launch_packet` (schema + launch packet only; no component campaigns have completed)

### Metrics Used

**Schema-level (required for synthesis):**

- `guard_authority`: mechanism, active, veto_rate
- `learned_component_contribution`: contribution_type, bound, active_rate
- `intervention_fallback_rates`: guard_veto_rate, fallback_rate, degraded_rate
- `outcomes`: success_rate, collision_rate, near_miss_rate, low_progress_rate, timeout_rate
- `evidence_tier`: launch_packet, smoke_only, nominal_only, stress, full_matrix, degraded,
  fallback, failed, not_available
- `verdict`: continue, revise, stop, insufficient_evidence, pending

**Launch packet (learned-risk model v1):**

- Trace contract: scenario_id, seed, candidate_id, termination_reason, metrics,
  trajectory_features, labels
- Required features: clearance, local_crowd_distance, route_progress, speed, goal_progress
- Required labels: collision, near_miss, low_progress
- Safety boundary: hard guards authoritative, learned risk auxiliary only, required diagnostics
  (learned_risk_score, hard_guard_decision, auxiliary_cost_weight)

### Key Results

**Schema (#1499):**

- Canonical evidence matrix schema frozen with 10 required fields and 8 optional diagnostic fields.
- Guard authority hard constraint: "Hard guards are always authoritative. No learned component may
  bypass, override, or silently disable a hard guard."
- Guard veto visibility required: every synthesis-accepted row must have
  `guard_authority.active = true` and a non-null `guard_authority.veto_rate`.
- Rejection rule: rows with `active=false` or zero guard interventions without proving the guard
  was active are invalid for synthesis.
- Evidence tier definitions: `full_matrix` required for comparative synthesis claims, `stress`
  supports bounded claims only, `launch_packet`/`smoke_only`/`nominal_only` excluded from
  synthesis.
- Fallback policy enforcement: `fallback`, `degraded`, `failed`, `not_available` rows are
  explicitly not benchmark evidence.
- Machine-readable validator added (issue #1515): `validate_hybrid_evidence_matrix.py` with
  required-field, enum, nullability, rate-bound, and guard-veto consistency checks.

**Launch packet (#1395, learned-risk model v1):**

- Config: `configs/training/learned_risk_model_issue_1395_launch_packet.yaml`
- Validator: `scripts/validation/validate_learned_risk_launch_packet.py`
- Frozen baseline: `hybrid_rule_v3_static_margin0_waypoint2`
- Trace contract enforced: required fields fail closed if missing or worktree-local `output/` paths
  detected.
- Safety boundary explicit: learned model auxiliary, hard guards authoritative.

### Known Caveats

1. **No component execution**: This is schema + launch packet evidence only. No component campaigns
   have run or produced evaluation metrics.
2. **Pending component evidence**: All hybrid-learning component issues (Issue #1470, Issue #1472,
   Issue #1474, Issue #1475, Issue #1358, Issue #1496) remain pending or incomplete.
3. **Schema validator limitations**: The validator proves row-level contract compliance but does not
   interpret benchmark outcomes or replace the synthesis consumer planned in #1489.
4. **Seed-schedule and scenario-manifest provenance**: Remain optional diagnostic fields; stricter
   requirements may be added after component campaigns demonstrate feasibility.
5. **Frozen baseline status**: The launch packet freezes a non-learning baseline, but the baseline
   artifact alias (`baseline_summary_stub.json`) is a preflight stub, not a durable campaign
   artifact. Future training must replace it with concrete run artifacts before claiming results.

### Blocker / Next Action

**Blocker:** Component campaigns not started or incomplete (Issue #1470, Issue #1472, Issue #1474,
Issue #1475, Issue #1358, Issue #1496).

**Next actions:**

1. Complete component campaigns and produce durable `stress` or `full_matrix` evidence for each
   component.
2. Populate the hybrid evidence matrix with synthesis-eligible rows (evidence tier `full_matrix` or
   `stress`, verdict `continue` or `revise`).
3. Use the machine-readable validator (`validate_hybrid_evidence_matrix.py`) to verify row contract
   compliance before synthesis.
4. Synthesize comparative hybrid-learning claims in issue #1489 only after component evidence is
   synthesis-eligible.

### Verdict

`not_ready` — schema and launch packet frozen, but component campaigns have not produced
synthesis-eligible evidence.

---

## Claim Area 6: Scenario Seed Sensitivity And Seed-Budget Stability

### Candidate Claim

"Robot-SF can identify seed-sensitive scenarios and hard-seed diagnostics that must caveat or gate
paper-facing comparison language until stronger seed-budget evidence exists."

### Evidence Links

- Scenario seed-sensitivity analysis:
  [issue_1608_seed_sensitivity_analysis.md](issue_1608_seed_sensitivity_analysis.md)
- Derived evidence bundle:
  [issue_1608_seed_sensitivity_2026-05-30/](evidence/issue_1608_seed_sensitivity_2026-05-30/)
- Analysis report:
  [seed_sensitivity_analysis.md](evidence/issue_1608_seed_sensitivity_2026-05-30/seed_sensitivity_analysis.md)
- Scenario table:
  [scenario_seed_sensitivity.csv](evidence/issue_1608_seed_sensitivity_2026-05-30/scenario_seed_sensitivity.csv)
- Seed difficulty table:
  [seed_difficulty_summary.csv](evidence/issue_1608_seed_sensitivity_2026-05-30/seed_difficulty_summary.csv)
- Source S10/h500 candidate bundle:
  [issue_1454_s10_h500_candidates_2026-05-23/](evidence/issue_1454_s10_h500_candidates_2026-05-23/)
- Merged analysis PR: [#1713](https://github.com/ll7/robot_sf_ll7/pull/1713)
- S20/S30 paper-facing evidence issue: [#1554](https://github.com/ll7/robot_sf_ll7/issues/1554)

### Evidence Tier

`diagnostic_s10` (derived analysis over durable compact Issue #1454 S10/h500 candidate artifacts)

### Metrics Used

- Per-scenario top-planner mean success by seed
- Across-seed mean-success range
- Hard seed count (`mean_success <= 0.5`)
- Easy seed count (`mean_success >= 0.75`)
- Hardest seed id across scenarios

### Key Results

PR #1713 classified all 48 scenarios from the issue #1454 S10/h500 candidate bundle:

| Classification | Count |
| --- | ---: |
| `seed_sensitive` | 25 |
| `not_seed_sensitive` | 23 |
| `inconclusive` | 0 |

Seed `116` was the hardest seed id across scenarios: mean top-planner success `0.7396`, with 12
hard scenario rows. The next hardest seed ids were `117`, `111`, `115`, and `112`.

### Claims Strengthened By #1713

- The repository can now identify seed-sensitive scenarios from a durable compact S10 candidate
  bundle without rerunning the full campaign.
- Seed `116` should be treated as a hard-seed diagnostic for follow-up inspection and targeted
  trace review.
- Scenario-level claim assembly should distinguish stable scenario behavior from
  seed-sensitive behavior before using single-seed or small-seed evidence in manuscript language.

### Claims That Remain S10-Only Or Blocked

- The #1713 analysis does **not** upgrade S10 evidence into paper-facing significance evidence.
- The result does **not** explain the causal mechanism behind hard seeds; issue #1609 or a
  trace-level mechanism review is still needed for causal interpretation.
- The result does **not** satisfy the #1554 claim-map gate for S20/S30 paper-facing comparisons.
  Issue #1554 remains blocked on a durable S20/S30 bundle when a paper-facing comparison requires
  stronger seed-budget evidence.
- SNQI ordering remains diagnostic only for this source surface; #1713 did not use SNQI as a
  manuscript-ready ranking metric.

### Known Caveats

1. **Derived diagnostic evidence**: The source is the exploratory issue #1454 S10/h500 candidate
   surface, not a pre-registered paper-facing S20/S30 campaign.
2. **No causal mechanism proof**: Seed sensitivity identifies where behavior changes across seeds;
   it does not prove why those seeds are hard.
3. **Top-planner subset**: Planner selection is deterministic and documented, but limited to the
   top four benchmark-success planner rows in the source bundle.
4. **Paper-facing boundary**: Per the artifact evidence vocabulary, this analysis is useful for
   prioritization and claim hygiene, not for standalone paper-facing significance claims.

### Blocker / Next Action

**Blocker:** Paper-facing seed-budget stability still requires the #1554 S20/S30 evidence gate when
the manuscript claim depends on stable planner rankings, safety deltas, or close metric comparisons.

**Next actions:**

1. Use seed `116` as a hard-seed diagnostic for targeted trace review and mechanism hypotheses.
2. Keep #1554 blocked on durable S20/S30 evidence; #1713 raises the importance of that evidence but
   does not unblock it.
3. When drafting manuscript claims, treat seed-sensitive scenarios as caveated unless supported by
   the stronger seed-budget tier required by #1554.

### Verdict

`insufficient` — #1713 strengthens diagnostic claim hygiene and hard-seed prioritization, but the
evidence remains S10-derived and does not satisfy paper-facing seed-budget stability requirements.

---

## Claim Area 7: Perturbation Mechanism Diagnostics

### Candidate Claim

"Scenario perturbation pilots can identify which local scenario mechanisms are useful diagnostic
probes, while keeping those findings separate from benchmark-strength or paper-facing planner
claims."

### Evidence Links

- Parent perturbation-criticality lane: [Issue #1610](https://github.com/ll7/robot_sf_ll7/issues/1610)
- Synthesis checkpoint:
  [issue_1965_perturbation_criticality_synthesis.md](issue_1965_perturbation_criticality_synthesis.md)
- Robot route-offset pilot:
  [issue_1904_scenario_perturbation_criticality_pilot.md](issue_1904_scenario_perturbation_criticality_pilot.md)
- Pedestrian route-offset pilot:
  [issue_1937_ped_route_offset.md](issue_1937_ped_route_offset.md)
- Corridor route trace response:
  [issue_1939_corridor_trace_response.md](issue_1939_corridor_trace_response.md)
- Single-pedestrian timing pilot:
  [issue_1941_ped_timing_phase.md](issue_1941_ped_timing_phase.md)
- Single-pedestrian speed pilot:
  [issue_1943_ped_speed_perturbation.md](issue_1943_ped_speed_perturbation.md)
- Leave-group speed trace:
  [issue_1945_orca_leave_group_speed_trace.md](issue_1945_orca_leave_group_speed_trace.md)
- Intersection timing/speed trace:
  [issue_1947_intersection_wait_timing_speed_trace.md](issue_1947_intersection_wait_timing_speed_trace.md)
- Wait-duration pilot:
  [issue_1949_ped_wait_duration_perturbation.md](issue_1949_ped_wait_duration_perturbation.md)
- Compact evidence index:
  [evidence/README.md](evidence/README.md)

### Evidence Tier

`diagnostic_local` (paired local pilots, compact tracked summaries, and trace inspections; not
benchmark-strength evidence)

### Metrics Used

- Paired no-op versus perturbed rows by planner, scenario, seed, and perturbation family
- Row status counts: completed, invalid, fallback, degraded, missing, failed
- Terminal deltas: success, collision, timeout
- Clearance deltas: mean minimum-distance changes on completed pairs only
- Trace-level mechanism notes for selected seed-local or corridor responses

### Key Results

The current Issue #1610 mechanism map is useful for diagnostic routing, not paper-facing claims:

| Mechanism / Family | Current Finding | Strength | Limitation |
| --- | --- | --- | --- |
| `robot_route_offset` | Low-criticality on the tested slice: terminal deltas stayed neutral and min-distance deltas were near zero. | observed diagnostic evidence | Limited scenarios and no terminal outcome signal. |
| `pedestrian_route_offset` | Corridor-sensitive clearance response; terminal metrics stayed neutral. | observed diagnostic evidence | Route-only support and trace-local mechanism, not broad criticality. |
| `single_pedestrian_start_delay_offset` | Strongest positive `francis2023_intersection_wait` clearance signal. | observed diagnostic evidence | Explicit single-pedestrian timing only; terminal metrics neutral. |
| `single_pedestrian_speed_offset` | Strong signed `francis2023_intersection_wait` clearance signal and one seed-local ORCA collision-to-success flip. | observed plus trace-inspected local evidence | Seed-local outcome flip is not replicated; route-pedestrian speed remains unsupported. |
| `single_pedestrian_wait_duration_offset` | Low-sensitivity on current tested magnitudes. | observed negative diagnostic evidence | One wait-bearing scenario only. |
| `pedestrian_density_offset` | Low-sensitivity in the current tiny smoke. | smoke-level evidence | Density-to-count behavior is route/runtime dependent. |
| `single_pedestrian_trajectory_waypoint_offset` | Tiny smoke completed with small positive clearance delta. | smoke-level evidence | One planner, two seeds; not enough for mechanism conclusions. |

### Claims Strengthened by Issue #1965

- The Issue #1610 lane now has a reusable mechanism vocabulary for deciding whether to stop,
  consolidate, or run a controlled follow-up slice.
- Start-delay and speed offsets in `francis2023_intersection_wait` are the clearest local
  phase/speed diagnostic signals.
- Small robot-route offsets are a low-sensitivity diagnostic on the tested slice and should not be
  rerun casually.
- Wait-duration and density pilots should be preserved as low-sensitivity or smoke-only evidence
  rather than treated as failed benchmark rows.

### Claims That Remain Diagnostic-Only or Blocked

- None of the current perturbation-family findings are benchmark-strength planner performance
  evidence.
- None of the findings are paper-facing robustness claims.
- Route-offset and timing/speed findings are mechanism hypotheses unless trace-backed for the exact
  planner/scenario/seed slice being cited.
- A broader S20/S30 or paper-facing perturbation claim remains premature until a schema-backed
  summary contract and selected controlled follow-up exist.

### Known Caveats

1. **Local diagnostic scope**: Evidence comes from small local pilots and trace inspections, not a
   pre-registered benchmark campaign.
2. **Completed-row boundary**: Invalid, fallback, degraded, missing, and failed rows remain
   limitations and must be excluded from completed-pair effect means.
3. **Terminal neutrality**: Several families show clearance shifts without success/collision/timeout
   deltas, so mechanism usefulness should not be translated into planner performance language.
4. **Seed-local effects**: The ORCA speed flip in `francis2023_leave_group` is useful for trace
   diagnosis but not replicated as a robustness result.
5. **Smoke-only families**: Trajectory waypoint and density pilots prove a narrow wiring path, not a
   stable mechanism effect.

### Blocker / Next Action

**Blocker:** This claim area is intentionally diagnostic-only. Paper-facing or benchmark-strength
language requires a controlled follow-up with schema-backed row accounting and an explicit evidence
tier upgrade.

**Next actions:**

1. Merge or otherwise land the reusable perturbation-family registry and `criticality_summary.v1`
   writer from issue #1980 before adding more one-off family summaries.
2. Prefer a controlled `francis2023_intersection_wait` speed/start-delay follow-up only if the next
   question changes exactly one axis: family, magnitude grid, planner set, scenario set, or seed
   budget.
3. Keep route-offset, wait-duration, density, and trajectory-waypoint findings in diagnostic
   routing language unless stronger paired evidence is produced.

### Verdict

`diagnostic_only` — Issue #1965 strengthens mechanism routing and claim hygiene, but it does not
create benchmark-strength or paper-facing evidence.

---

## Cross-Claim Summary

| Claim Area | Verdict | Readiness Blocker | Evidence Tier |
| --- | --- | --- | --- |
| AMV Benchmark Evidence | `blocked` | AMV coverage gap (all required dimensions missing); SNQI contract cautions | `stress` + `full_matrix` (broader) |
| Adversarial Stress Testing | `not_ready` | Execution stage not started (issue #1501 pending) | `launch_packet` |
| CARLA/Native Replay Parity | `blocked` | Robot actor spawn failure prevents oracle replay | `failed` |
| Predictive Planner v2 | `negative` | Obstacle-feature variant worsened success despite forecast improvement | `stress` |
| Hard-Guarded Hybrid Learning | `not_ready` | Component campaigns incomplete | `launch_packet` |
| Scenario Seed Sensitivity And Seed-Budget Stability | `insufficient` | S20/S30 paper-facing evidence gate remains blocked (#1554) | `diagnostic_s10` |
| Perturbation Mechanism Diagnostics | `diagnostic_only` | Needs schema-backed controlled follow-up before evidence-tier upgrade | `diagnostic_local` |

## Conservative Manuscript Readiness Assessment

**No claim area is currently manuscript-ready without explicit maintainer decisions on acceptable
evidence boundaries.**

- **AMV benchmark**: blocked by AMV coverage gap and SNQI contract status. Requires maintainer
  decision on whether `amv_coverage_status=warn` and `snqi_contract_status=fail`/`warn` are
  acceptable for paper-facing claims.
- **Adversarial stress testing**: specification frozen, but execution evidence unavailable until
  issue #1501 completes.
- **CARLA/native replay parity**: blocked by robot actor spawn failure. Requires diagnosis and
  rerun before positive metric-parity criterion is met.
- **Predictive planner v2**: negative same-seed evidence contradicts the claim. Requires
  coupling/objective revision before further expansion.
- **Hard-guarded hybrid learning**: schema and launch packet frozen, but component campaigns have
  not produced synthesis-eligible evidence.
- **Scenario seed sensitivity**: PR #1713 identifies seed-sensitive scenarios and seed `116` as a
  hard-seed diagnostic. This strengthens prioritization and caveat discipline, but it remains
  S10-derived diagnostic evidence and does not satisfy #1554's S20/S30 paper-facing evidence gate.

## Recommended Follow-Up Issues

1. **AMV coverage gap resolution**: Decide whether scenario metadata must be enriched or whether
   `amv_coverage_status=warn` is acceptable for manuscript scope.
2. **SNQI contract decision**: Accept `warn`/`fail` with explicit caveats or revise the SNQI
   contract before paper-facing promotion.
3. **CARLA robot actor spawn diagnosis**: Diagnose and fix robot actor spawn failure (issue #1437)
   before claiming metric parity.
4. **Predictive coupling revision**: Revise planner-side coupling or training objective before
   predictive-v2 expansion (block issue #1490 until preflight shows closed-loop improvement).
5. **Hybrid learning component campaigns**: Complete component campaigns (Issue #1470, Issue #1472,
   Issue #1474, Issue #1475, Issue #1358, Issue #1496) before synthesis (Issue #1489).
6. **Adversarial execution**: Complete issue #1501 (one-family smoke) before adversarial stress
   testing can support manuscript claims.
7. **Seed-budget evidence**: Keep issue #1554 blocked until a durable S20/S30 evidence bundle is
   available for any paper-facing comparison that depends on stable rankings or safety deltas.

## Validation

This note was created from durable tracked evidence sources only. Validation commands:

```bash
# Link sanity check (all repo-relative paths should resolve)
rg -F 'docs/context/' docs/context/issue_1542_manuscript_claim_evidence_map.md | \
  grep -v '^#' | grep -oP 'docs/context/[^)]+' | sort -u | \
  while read path; do [ -e "$path" ] || echo "Missing: $path"; done

# Docs proof consistency
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh

# Whitespace check
git diff --check
```

## Artifact Decision

This note is a tracked context artifact under `docs/context/`. It should be linked from
`docs/context/README.md`. No new files under `output/` were created or referenced. All evidence
links are repository-root-relative and point to durable tracked surfaces.
