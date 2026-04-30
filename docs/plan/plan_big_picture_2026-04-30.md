# Robot-SF two-horizon improvement strategy - 2026-04-30

## Core recommendation

Robot-SF should optimize for two linked horizons:

1. **Near-term paper delivery:** protect a defensible camera-ready benchmark story with clear
   provenance, SNQI contract evidence, baseline boundaries, and durable artifacts.
2. **Long-term research roadmap:** build a certified, adversarially tested, layered local-navigation
   stack after the benchmark claim is stable.

The near-term target is not a single promoted local policy. It is a falsifiable benchmark claim:

> A strong Robot-SF policy and baseline set evaluated on the maintained scenario matrix, with
> clear planner provenance, seed/bootstrap uncertainty, SNQI diagnostics, and no fallback execution
> counted as benchmark success.

The long-term target is a policy-improvement loop:

```text
scenario generator
  -> scenario certificate
  -> policy sweep
  -> failure attribution
  -> adversarial search
  -> counterexample replay set
  -> training / planner update
  -> frozen holdout evaluation
```

## Current evidence that changes the priority order

- `memory/experiments/2026-04-20_issue_791_distribution_alignment_dominates.md` shows that
  eval-aligned PPO training explains most of the observed lift to the current strong policy. Do not
  credit architecture, curriculum, or foresight as the dominant driver without new evidence.
- `memory/decisions/2026-04-20_issue_791_narrow_benchmark_claim.md` fixes the paper framing:
  benchmark-set performance across the scenario matrix, not OOD generalization or transfer.
- `docs/context/dreamerv3_program_close_out_2026_04_30.md` closes and deprioritizes the DreamerV3
  track for camera-ready scope after repeated no-eval runs, NaNs, and OOM.
- `docs/benchmark_planner_family_coverage.md` is the claim boundary for planner families. Only
  implemented-and-benchmarkable rows should support current benchmark claims.
- `docs/benchmark_camera_ready.md`, `docs/benchmark_release_protocol.md`, and
  `docs/benchmark_release_reproducibility.md` define the paper-facing benchmark, SNQI, release,
  and artifact obligations.
- `docs/context/issue_691_benchmark_fallback_policy.md` is the fallback boundary: fallback or
  degraded execution is a caveat or exclusion reason, not evidence of benchmark success.

## Horizon A - near-term paper delivery

The paper track is the first execution priority. It should produce a claim that a reviewer can
audit from committed configs, benchmark artifacts, and context notes.

### A1. Protect the claim language

Use this public framing:

- strong policy on a broad maintained scenario matrix,
- comparison against baseline-ready planners under the same benchmark contract,
- seed/bootstrap uncertainty and SNQI diagnostics reported with the result.

Avoid this framing unless a separate study exists:

- OOD generalization,
- transfer to unseen environments,
- architecture-driven lift beyond the eval-aligned PPO evidence,
- DreamerV3 as a promoted benchmark competitor.

### A2. Preserve benchmark and SNQI evidence

Paper-facing benchmark evidence must include:

- canonical benchmark command or release workflow,
- planner list and kinematics mode,
- SNQI weights, baseline assets, diagnostics, and sensitivity status,
- bootstrap or seed-variance evidence,
- artifact paths or durable artifact references,
- explicit fallback/degraded/not-available status for planners that fail their contract.

### A3. Keep planner-family claims conservative

Use `docs/benchmark_planner_family_coverage.md` as the support boundary:

- benchmarkable: current paper-facing support when provenance and dependencies are valid,
- implemented but experimental: controlled experiments only,
- conceptually adjacent: background or roadmap only,
- missing: future work only.

### A4. Treat semantic blockers as paper risks

Resolve or caveat these before using failures as policy evidence:

- route handoff or first-waypoint errors,
- invalid SVG obstacle conversion,
- SNQI or metric-contract drift,
- missing optional planner dependencies,
- fallback or degraded execution,
- worktree-local artifacts that have not been promoted to durable evidence.

### Horizon A proof obligations

| Claim type | Required proof |
| --- | --- |
| PPO benchmark result | canonical config, model artifact provenance, benchmark run, seed/bootstrap evidence, SNQI diagnostics |
| Baseline planner result | implemented benchmark entrypoint, dependency availability, native/adapter mode, non-fallback execution |
| SNQI conclusion | versioned weights, baseline normalization assets, contract diagnostics, sensitivity or ablation status |
| Release artifact | durable artifact reference or tracked manifest, plus enough metadata to recover the exact input |
| Failure attribution | scenario validity, planner mode, route/geometry sanity, and reproducible episode or counterexample bundle |

## Horizon B - long-term research roadmap

After the paper track is stable, Robot-SF should become a fast falsification and policy-development
platform. The roadmap remains a layered local-navigation stack, but every promotion step must pass
through executable Robot-SF evidence before it supports a manuscript or benchmark claim.

### B1. Scenario certification

Add a `scenario_cert.v1` concept before expanding adversarial scenario generation. The certificate
should classify scenarios as invalid, geometrically infeasible, kinodynamically infeasible,
dynamically overconstrained, knife-edge, or hard-but-solvable.

The first useful certificate should cover:

- inflated collision-free path existence,
- minimum static clearance,
- route validity for robot and pedestrians,
- kinodynamic feasibility checks for turning, acceleration, and braking,
- dynamic-agent plausibility,
- oracle or high-budget baseline success evidence,
- difficulty labels that distinguish universal hardness from planner-specific mismatch.

### B2. Adversarial falsification

Adversarial pedestrians and static scenarios should be development stress tools before they become
frozen benchmark cases.

Recommended sequence:

1. black-box parameter search over starts, goals, timing, speed, and seeds;
2. scripted adversarial families such as crossing, bottleneck blocker, mirror avoidance, group
   squeeze, late stop, cutoff, and occluded emergence;
3. counterexample replay bundles with scenario, certificate, episodes, trajectory, attribution, and
   video;
4. learned multi-agent adversaries only after plausibility constraints and replay evidence exist.

### B3. Layered policy portfolio

The long-term local policy should be a portfolio stack rather than a monolithic policy:

```text
route / topology
  -> scene graph + local free-space representation
  -> pedestrian / occupancy prediction
  -> candidate trajectory generation
  -> risk-aware trajectory scoring
  -> safety shield
  -> robot-specific controller
```

Start with a non-learning stack using route rebasing, obstacle-aware subgoals, ORCA/HRVO/DWA/MPPI
proposal generation, TTC/distance/progress scoring, braking-distance checks, and a unicycle or
e-scooter controller. Add learning in this order: prediction, risk scoring, proposal policy, then
end-to-end policy as an ablation.

### B4. CARLA transfer

CARLA should be a higher-fidelity validation layer, not the starting point for training.

Transfer readiness requires:

- stable scenario certificates,
- simulator-independent observations,
- physical command or local-trajectory outputs,
- trajectory-based metrics,
- counterexamples that are stable across seeds and not parser or route artifacts,
- validated Robot-SF performance on verified-simple, atomic, classic, adversarial-development, and
  frozen evaluation sets.

The first CARLA target should be oracle replay/parity for certified Robot-SF scenarios. Sensor-level
perception and policy training inside CARLA should come after replay parity.

## Shared proof spine

The two horizons share one rule: no workstream promotes a claim until the evidence can be inspected,
rerun, or traced to a durable artifact. This matters for paper delivery and for future research
quality.

Use these proof gates:

- Paper benchmark claim: canonical benchmark run, SNQI diagnostics, bootstrap/seed evidence,
  planner provenance, and durable artifact trail.
- Planner readiness claim: current benchmark entrypoint, dependency availability, mode
  classification, and non-fallback execution proof.
- Metric or SNQI claim: contract diagnostics and reproducible input assets.
- Research-roadmap promotion: executable proof in Robot-SF before any manuscript-facing claim.
- CARLA transfer claim: parity evidence against certified Robot-SF scenarios before perception or
  training claims.

## Sequencing

| Order | Work | Horizon | Why now |
| --- | --- | --- | --- |
| 1 | Claim-language cleanup and provenance audit | A | Prevents overclaiming before paper text and PRs reuse the result |
| 2 | Camera-ready benchmark validation and SNQI diagnostics | A | Produces reviewer-auditable evidence |
| 3 | Durable artifact and config provenance check | A | Makes the result recoverable outside this worktree |
| 4 | Route, geometry, metric, and fallback blockers | A | Prevents invalid failure attribution |
| 5 | Scenario certification v1 | B | Makes generated hard cases distinguishable from invalid cases |
| 6 | Failure attribution and adversarial replay bundles | B | Turns failures into reusable development evidence |
| 7 | Layered policy portfolio | B | Improves local navigation after the evaluation substrate is trustworthy |
| 8 | CARLA oracle replay/parity | B | Tests transfer only after simulator-independent contracts exist |

## Issue candidates

### Paper-critical or paper-risk issues

1. Audit issue-791 claim language against the narrow benchmark-set decision
   (`memory/decisions/2026-04-20_issue_791_narrow_benchmark_claim.md` and
   `memory/experiments/2026-04-20_issue_791_distribution_alignment_dominates.md`).
2. Verify camera-ready PPO provenance, benchmark command, seed/bootstrap evidence, and SNQI
   diagnostics (`docs/benchmark_camera_ready.md` and release reproducibility docs).
3. Audit baseline planner dependencies and native/adapter/fallback modes for the paper matrix using
   `docs/benchmark_planner_family_coverage.md` and
   `docs/context/issue_691_benchmark_fallback_policy.md`.
4. Verify durable artifact references for model, benchmark, SNQI, and release inputs, following the
   artifact durability rules in `AGENTS.md`.
5. Resolve or caveat route handoff (`#730`), invalid SVG geometry (`#837`), SNQI semantics (`#455`),
   metric drift, and fallback/degraded execution before using affected failures as policy evidence.

### Research follow-up issues

1. Add `scenario_cert.v1` for geometric, kinodynamic, dynamic-agent, and difficulty certification.
2. Add adversarial scenario search with plausibility constraints and counterexample replay bundles.
3. Extend adversarial pedestrians from single-agent templates to constrained multi-agent stress
   tests.
4. Add obstacle-conditioned prediction baselines and scene representations.
5. Build `policy_stack_v1` as a portfolio planner with safety shielding and controller diagnostics.
6. Add CARLA oracle replay/parity for certified Robot-SF scenarios.

## Claim-boundary notes

- PPO is currently the highest-yield paper-facing policy track, but the strongest result is
  benchmark-set performance. Do not present it as OOD generalization, transfer, or performance on
  unseen environments unless a separate held-out study is run.
- DreamerV3 is historical context and a deprioritized research track for this paper cycle. A future
  attempt needs a structurally different setup and fresh proof before it becomes a benchmark
  competitor.
- Fallback, degraded, skipped, or not-available planner execution can support diagnostics, but it
  cannot count as benchmark success.
- Worktree-local `output/` files are not durable evidence. Promote required artifacts or track a
  manifest that identifies how to recover them.
- CARLA is deferred until Robot-SF has simulator-independent scenario, observation, action, and
  metric contracts. The first CARLA milestone is oracle replay/parity, not training.

## Bottom line

The strongest near-term step is to protect the paper claim: a strong policy and baseline set on the
maintained scenario matrix, backed by provenance, SNQI diagnostics, uncertainty evidence, and
fail-closed benchmark semantics.

The strongest long-term step is to turn Robot-SF into a certified falsification loop. Scenario
certification, adversarial replay, layered policy portfolios, richer prediction, and CARLA parity
remain valuable, but they become claim-bearing only after the paper evidence spine is stable.
