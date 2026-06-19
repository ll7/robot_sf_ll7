---
name: research-status
description: Current research directions, hypotheses in flight, and blocker/candidate status
metadata:
  type: project
  created: 2026-06-19
  category: research
---

# Robot SF Research Status

**Last Updated**: 2026-06-19
**Status Tracking**: GitHub Projects #5 (advisory prioritization)
**Related**: [Benchmark Platform Status](benchmark_platform_status.md)

## Active Research Areas

### 1. Social Compliance Metric (Exploratory → Candidate)

**Status**: Exploratory
**Hypothesis**: Social compliance (respecting pedestrian comfort, maintaining safe distances) is
orthogonal to success/efficiency metrics and should be measured separately.

**Evidence**: Smoke evidence
- Metric contract defined in PR #3088
- Component probes: proximity comfort, trajectory smoothness, interaction naturalness
- Next step: Integration into SNQI weighting framework

**Validation path**: Nominal benchmark evidence requires human annotation study on >100 episodes

---

### 2. Route Corridor Subgoal Recovery (Blocked)

**Status**: Blocked
**Issue**: Planners occasionally deviate from route corridor; recovery logic unclear.

**Evidence**: Diagnostic-only
- Related corner cases are tracked in issue-specific benchmark and context notes; no single
  canonical issue is assigned in this memory entry yet.
- Current fallback: Reset planner state (crude but effective)
- Candidate fix: Subgoal-based corridor re-entry

**Blockers**:
- Route corridor definition not formalized (coordinates vs. semantic zones?)
- Recovery metric not defined (how to measure "recovered" vs. "still lost"?)
- Dependent on planner architecture (SocialForce vs. learned; different recovery profiles)

**Revival condition**: Formalize route corridor contract + define recovery metric

**Tracking**: Issue #1028 (corridor-subgoal-recovery branch exists)

---

### 3. Adversarial Scenario Search (Exploratory)

**Status**: Exploratory; diagnostic-only results
**Hypothesis**: Automated search can find pedestrian configurations that expose planner failures
faster than manual scenario design.

**Evidence**: Diagnostic
- Scenario generation profile coverage is still being consolidated across issue-specific notes.
- Tool: `scripts/tools/generate_adversarial_scenario_manifests.py` (experimental)
- Limitation: No ground truth for "interesting" adversarial scenario (subjective)

**Next step**: Define adversarial criteria (high collision rate? timeout? large deviation?) and
measure repeatability across planner families

**Tracking**: Issue #1022 (route-corridor-research branch); separate branch for search variants

---

### 4. Planner Readiness Matrix (Candidate)

**Status**: Candidate; nominal benchmark evidence
**Deliverable**: Matrix showing which planners are ready for which map families / scenario types.

**Evidence**: Published
- Matrix generated in PR #3087
- Covers SocialForce, PPO, Random across standard H500 maps
- Shows success rate, SNQI by map family

**Next step**: Extend to additional planner families (graph-based, hybrid rule+learning)

**Tracking**: Issue #1051 (evidence-provenance-audit); integration into planner roadmap

---

### 5. Multi-Pedestrian Family Investigation (Smoke)

**Status**: Smoke evidence
**Hypothesis**: Varying pedestrian dynamics (SocialForce parameters, crowd size) reveals planner
robustness gaps.

**Evidence**: Smoke
- Smoke runs completed on H500 subset (Issue #1015)
- Preliminary result: PPO more sensitive to crowd size than SocialForce
- Need: Full benchmark run for nominal evidence

**Next step**: Full benchmark sweep with controlled pedestrian family variation

**Tracking**: Issue #1015 (1015-multi-ped-family-smoke branch)

---

## Blocked Work

| Issue | Title | Blocker | Revival Condition |
| --- | --- | --- | --- |
| Issue #1028 | Route corridor subgoal recovery | Route corridor contract undefined | Formalize + metric definition |
| Issue #1004 | Policy stack v1 runtime | Architecture decision on composability | Planner roadmap decision |
| Issue #1049 | H500 mechanism pilot | Benchmark route clearance certification | Complete v1 cert audit |

---

## Research Methodology

### Status Labels

Use explicit language to mark research state:

- **Exploratory**: Promising direction, unvalidated; runs may live in feature branches
- **Diagnostic**: Useful for debugging; not a benchmark claim
- **Candidate**: Plausible next implementation target; smoke evidence or strong hypothesis
- **Blocked**: Needs artifact/dependency/decision before proceeding
- **Not benchmark evidence**: Ran outside benchmark contract or under fallback mode
- **Nominal benchmark evidence**: Meets predeclared benchmark matrix; reproducible
- **Paper-grade**: Fully reproducible, suitable for manuscript-facing claims

### Evidence Grading

- **Diagnostic-only**: Reproduces edge case or surface-level behavior; limited sample
- **Smoke evidence**: Narrow execution proof on 1-2 scenarios; direction validation
- **Nominal benchmark evidence**: Predeclared config, full seed range, within test budget
- **Paper-grade**: Plus independent validation, artifact provenance link, caveats documented

### Capturing Hypotheses

For new research directions:

1. **Brief**: 2-3 sentence hypothesis statement
2. **Status**: Label as exploratory/diagnostic/blocked/candidate
3. **Evidence**: What's been tried, what result observed
4. **Next step**: Smallest proof that moves evidence grade up by one tier
5. **Tracking**: Issue number or branch name

Store hypotheses close to experiments: config files, issue comments, or feature branch README. Central
hypothesis ledger is opt-in (see `memory/research_hypotheses.md` if cross-run tracking is needed).

### Blocker Documentation

When work is blocked:

1. Name the blocker precisely (e.g., "route corridor definition", not "unclear architecture")
2. Document the exact artifact/decision/dependency needed
3. List revival condition, such as when a named ADR or roadmap decision is approved.
4. Link to related decisions or context notes when those files exist in the repository.

---

## Long-Running Tasks

### H500 Validation Campaign

**Tracking**: Issue #1049
**Status**: In progress
**Scope**: Full benchmark validation of H500 route set under nominal benchmark contract
**Timeline**: Ongoing; checkpoint results at 25/50/75/100 map completion
**Evidence**: Nominal (108 tests passing); full campaign = paper-grade

### Semantic Blocker Audit

**Tracking**: Issue #1057
**Status**: In progress
**Scope**: Audit planner implementations for unintended blocking behavior (e.g., SocialForce
collision detection edge cases)
**Output**: Issue-specific context note with reproduction configs
**Evidence**: Diagnostic + smoke evidence for identified cases

---

## Decision Tracking

### Key Decisions (ADRs)

- **Factory pattern**: see `robot_sf/spec_factory.py` and `docs/architecture/configuration.md`.
- **Planner roadmap**: see planner-facing context notes under `docs/context/` and
  `docs/ai/repo_overview.md`.
- **SNQI metric**: see benchmark metric documentation and implementation under
  `robot_sf/benchmark/` and `robot_sf/metrics/`.
- **Fallback policy**: See `docs/context/issue_691_benchmark_fallback_policy.md`

---

## Memory Maintenance

- Review this file quarterly; archive resolved blocked work
- For new research directions, add entry with status label and tracking issue
- Link decisions with ordinary Markdown links to files that exist in this repository.
- Use targeted memory edits plus `.agents/skills/context-note-maintainer/` when durable context
  notes need companion updates.

---

## Quick Links

- **Benchmark quickstart**: `specs/120-social-navigation-benchmark-plan/quickstart.md`
- **Project #5** (advisory): GitHub Projects for prioritization
- **Issues**: Filtered by label: `research`, `exploratory`, `diagnostic`, `blocked`
- **Evidence docs**: `docs/context/evidence/` (small, durable samples)
