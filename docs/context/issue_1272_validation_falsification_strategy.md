# Issue #1272 Safety-Oriented Validation And Falsification Strategy

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1272>

## Status

This note is the current roadmap handoff for the strategic positioning:

> Robot SF is scenario-based validation and falsification infrastructure for low-speed autonomous
> public-space systems.

This is a repository strategy and evidence-boundary note. It does not introduce a safety
certification, homologation, or proof-of-safety claim. It also does not claim that the current
benchmark proves real-world deployment readiness.

Historical predecessor: [plan_big_picture_2026-04-30.md](../plan/plan_big_picture_2026-04-30.md).
That plan remains useful for
paper-vs-research sequencing, but this note is the concise current routing surface for the
safety-oriented validation/falsification lane.

## Why This Framing

Robot SF already has reusable pieces for more than a planner leaderboard:

- scenario contracts and scenario certification surfaces,
- camera-ready benchmark execution, aggregation, and release provenance,
- fail-closed planner fallback policy,
- adversarial search and optimizer-backed sampler pilots,
- route-clearance and semantic-blocker audits,
- artifact evidence vocabulary and durable publication rules,
- emerging failure-archive and BenchmarkClaim work.

The strategic value is to connect those pieces into an auditable loop where benchmark, adversarial,
and manual analysis outputs say exactly what they cover and what they do not cover.

## Evidence Boundaries

Use this wording discipline for future issues and PRs:

- **Observed evidence**: a tracked config, command, test, benchmark run, context note, or durable
  artifact pointer shows the behavior in this repository.
- **Development evidence**: local smoke runs, synthetic comparisons, and worktree-local
  `output/` files can guide implementation but are not paper-facing evidence by themselves.
- **Benchmark evidence**: an execution path with canonical configs, planner mode, seeds, metrics,
  SNQI/provenance, and fallback/degraded/not-available classification.
- **Falsification evidence**: a counterexample or failure bundle that records scenario inputs,
  seeds, planner mode, outcome, attribution, replayability, and artifact durability.
- **Non-claim**: none of the above is certification, legal assurance, human-subject validation, or
  real-world safety proof.

Fallback or degraded planner execution remains a caveat or exclusion reason, following
[issue_691_benchmark_fallback_policy.md](issue_691_benchmark_fallback_policy.md); it must not be
counted as successful benchmark evidence unless the task explicitly measures fallback behavior.

## Near-Term Delivery Lanes

These lanes are concrete enough for implementation issues and should stay connected to existing
backlog tickets instead of spawning broad duplicates.

| Lane | Purpose | Current issue surfaces |
| --- | --- | --- |
| Scenario intent and contracts | Record authored scenario assumptions before execution. | [scenario_contracts.md](../scenario_contracts.md), Issue #1235 |
| Scenario certification | Classify valid, infeasible, stress-only, and hard-but-solvable cases. | [scenario_certification.md](../scenario_certification.md), Issue #1057 |
| Optimizer-backed falsification | Search for hard candidates without making paper claims from pilots. | Issue #1236 |
| Failure archive curation | Preserve actionable counterexamples with minimization and caveats. | Issue #1237 |
| Scenario diversity accounting | Quantify benchmark coverage gaps before expanding claims. | Issue #1240 |
| BenchmarkClaim artifacts | Emit machine-readable claim metadata only when supporting evidence exists. | Issue #1245 |
| ODD contracts | Bound benchmark and falsification claims by operating assumptions. | Issue #1269 |
| Hazard traceability | Link scenario families to intended hazards and supporting metrics. | Issue #1270 |
| Seed-sensitivity explorer | Separate stable counterexamples from one-seed artifacts. | Issue #1271 |

Issue #1272 is the roadmap parent for those lanes. It should not absorb their implementation scope.

## Long-Term Research Directions

Keep these directions subordinate to the validation/falsification evidence loop:

- richer scenario generation with explicit certification and artifact promotion,
- prediction-aware safety shields and graded observation contracts after their prerequisite
  schemas land,
- manual-control benchmark recorder evidence as diagnostic support, not publication proof by
  anecdote,
- CARLA replay/parity as a higher-fidelity validation layer after Robot SF scenarios have stable
  contracts,
- learned or foundation-model components only when they improve a measured validation surface and
  preserve provenance.

## Non-Goals

Do not use this roadmap to imply:

- Robot SF certifies robot safety or provides a legal safety case,
- nominal scenario success proves robustness under stress or adversarial conditions,
- local-only artifacts under `output/` are durable benchmark evidence,
- exploratory samplers, shields, or learned components are paper-grade baselines before promotion,
- CARLA or real-world transfer is established before replay/parity evidence exists.

## Routing Rules For Future Work

1. Put new ideas into the closest lane above when possible.
2. If an idea changes evidence semantics, add validation and caveat text in the same PR.
3. If an issue needs maintainer choice about claim scope, mark it `decision-required` instead of
   guessing.
4. If a task produces reusable reasoning, add or update a `docs/context/` note and link it from the
   normal entry points.
5. Keep implementation PRs narrow; roadmap issues should point to children, not become umbrella
   implementation branches.

## Validation

This note is documentation/governance only. Expected validation for PR handoff:

```bash
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check origin/main...HEAD
```

Before any future PR converts this roadmap into public claim language, also review:

- [code_review.md](../code_review.md)
- [benchmark_camera_ready.md](../benchmark_camera_ready.md)
- [benchmark_artifact_publication.md](../benchmark_artifact_publication.md)
- [issue_691_benchmark_fallback_policy.md](issue_691_benchmark_fallback_policy.md)
