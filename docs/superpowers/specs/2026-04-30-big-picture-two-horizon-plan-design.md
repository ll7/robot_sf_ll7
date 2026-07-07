# Two-Horizon Big-Picture Plan Design

Date: 2026-04-30
Target document: `docs/plan/plan_big_picture_2026-04-30.md`
Status: approved design draft for plan refinement

## Goal

Refine the current Robot-SF improvement strategy into a two-horizon plan that serves both
near-term paper delivery and long-term research roadmap quality.

The plan should keep the useful strategic ideas from the current document, but make the immediate
priority unambiguous: defensible camera-ready benchmark evidence comes before new policy-stack,
adversarial, DreamerV3, or CARLA expansion.

## Boundaries

In scope:

- Restructure the big-picture plan into near-term and long-term horizons.
- Preserve useful roadmap ideas while gating them behind evidence and sequencing.
- Incorporate current repository evidence on PPO, DreamerV3, SNQI, planner readiness, and artifact
  durability.
- Make proof obligations explicit for paper-facing claims and research-roadmap promotions.

Out of scope:

- Implementing benchmark, planner, training, scenario-certification, adversarial, or CARLA code.
- Opening or editing GitHub issues as part of this design.
- Changing benchmark semantics, configs, metrics, or promoted policy artifacts directly.

## Evidence Sources

The refined plan should be anchored in these repository surfaces:

- `memory/experiments/2026-04-20_issue_791_distribution_alignment_dominates.md`
  - Distribution alignment explains most of the PPO lift to the current strong benchmark-set
    result.
- `memory/decisions/2026-04-20_issue_791_narrow_benchmark_claim.md`
  - Paper framing is a strong policy on a broad scenario matrix, not OOD generalization.
- `docs/context/dreamerv3_program_close_out_2026_04_30.md`
  - DreamerV3 is closed and deprioritized for paper-facing scope after repeated no-eval runs,
    NaNs, and OOM.
- `docs/benchmark_planner_family_coverage.md`
  - Planner-family claims must distinguish implemented benchmarkable planners from experimental or
    conceptually adjacent families.
- `docs/benchmark_camera_ready.md`, `docs/benchmark_release_protocol.md`, and
  `docs/benchmark_release_reproducibility.md`
  - Camera-ready benchmark, release, SNQI, and reproducibility obligations.
- `docs/context/issue_691_benchmark_fallback_policy.md`
  - Fallback or degraded execution is not benchmark-strengthening evidence.
- `AGENTS.md` and `.agents/PLANS.md`
  - Plans should be operational, evidence-backed, and validation-oriented.

## Proposed Structure

### Executive Recommendation

The plan should lead with a two-horizon strategy:

- Horizon A: camera-ready paper delivery.
- Horizon B: post-paper research roadmap.
- Shared evidence spine: every claim-promoting workstream needs named evidence and proof.

This replaces the current implicit emphasis on building the "best local policy" first. The better
near-term target is a defensible benchmark story with clean provenance and clear claim boundaries.

### Current Evidence

The plan should summarize the most decision-relevant evidence:

- PPO is currently the highest-yield paper-facing policy track.
- The strongest PPO result is benchmark-set performance and must not be framed as OOD transfer.
- DreamerV3 is not part of the camera-ready scope.
- Baseline-ready planner support is narrower than the roadmap planner list.
- SNQI and camera-ready artifacts are claim-bearing surfaces, not incidental outputs.
- Worktree-local `output/` files are not durable evidence unless promoted or represented by a
  durable manifest.

### Horizon A: Paper Delivery Plan

Paper delivery should be the first execution track. It should include:

- Protect the issue-791 claim language: benchmark-set performance across the scenario matrix.
- Verify PPO and baseline provenance through canonical configs, artifacts, and benchmark commands.
- Keep SNQI contract diagnostics and bootstrap/seed evidence visible in the validation story.
- Use the planner-family coverage matrix to avoid overclaiming experimental planners.
- Treat route handoff, invalid geometry, metric drift, fallback/degraded execution, and missing
  durable artifacts as blockers or caveats before any public claim.
- Record validation results in existing benchmark/context docs or a clearly linked context note.

### Horizon B: Research Roadmap

The long-term roadmap should remain ambitious but sequenced after the paper gate:

- Scenario certification v1 for geometry, route, kinodynamics, dynamic-agent feasibility, and
  difficulty labels.
- Failure attribution and adversarial falsification loops built on certified scenarios.
- Policy-stack portfolio work that uses classical, optimization, prediction, and learned proposals
  as components rather than treating a single new policy as the immediate goal.
- Obstacle-conditioned prediction and richer scene representations after semantic blockers are
  controlled.
- CARLA transfer only after simulator-independent scenario, observation, action, and metric
  contracts exist; the first CARLA target should be oracle replay/parity, not training.

DreamerV3 may remain as historical context or future optional research only if a structurally new
setup is proposed. It should not appear as a near-term promoted baseline.

## Priority Order

The refined plan should order work as follows:

1. Paper claim protection and wording boundaries.
2. Benchmark readiness, SNQI contract, bootstrap/seed evidence, and durable artifacts.
3. PPO and baseline provenance.
4. Known semantic blockers that can corrupt attribution: route handoff, invalid SVG geometry,
   metric drift, and fallback/degraded execution.
5. Scenario certification and failure attribution.
6. Adversarial scenario generation and counterexample replay.
7. Portfolio policy stack and richer prediction/scene understanding.
8. CARLA oracle replay/parity, then later sensor/perception stress.

## Validation Model

The plan should define proof obligations by claim type:

- Paper benchmark claim: canonical benchmark run, SNQI diagnostics, bootstrap/seed evidence,
  planner provenance, and durable artifact trail.
- Planner readiness claim: current benchmark entrypoint, dependency availability, mode
  classification, and non-fallback execution proof.
- Metric or SNQI claim: contract diagnostics and reproducible input assets.
- Research-roadmap promotion: executable proof in Robot-SF before any manuscript-facing claim.
- CARLA transfer claim: parity evidence against certified Robot-SF scenarios before perception or
  training claims.

## Issue Candidate Handling

The current issue list should be revised into two groups:

- Paper-critical or paper-risk issues, such as claim protection, benchmark provenance, SNQI
  contract, route handoff, geometry validity, and artifact durability.
- Research follow-ups, such as scenario certification, adversarial search, portfolio policy stack,
  obstacle-conditioned prediction, and CARLA oracle replay.

This prevents exploratory roadmap work from appearing as immediate paper scope.

## Acceptance Criteria

The refined plan is acceptable when:

- It clearly separates paper delivery from long-term research.
- It reflects the current PPO, DreamerV3, planner-family, SNQI, and artifact evidence.
- It names proof obligations before claim promotion.
- It preserves useful long-term ideas without making them near-term blockers.
- It can guide future agents without encouraging benchmark overclaims.
