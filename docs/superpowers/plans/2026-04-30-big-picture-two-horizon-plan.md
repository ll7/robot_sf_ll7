# Big-Picture Two-Horizon Plan Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Status:** Implemented in commit `80a8ad7a` (`docs: refine big-picture plan into two horizons`).
> This file is retained as the execution record and reference plan; unchecked boxes reflect the
> original implementation checklist, not active unexecuted work.

**Goal:** Revise `docs/plan/plan_big_picture_2026-04-30.md` into a two-horizon strategy that supports near-term paper delivery and long-term research roadmap quality.

**Architecture:** This is a documentation-only change. The revised plan should keep useful strategic content, but reorganize it around Horizon A for camera-ready paper delivery, Horizon B for post-paper research, and a shared evidence/proof spine that prevents speculative roadmap work from becoming paper-facing claims.

**Tech Stack:** Markdown documentation, repository context notes, benchmark documentation, `rg`, `sed`, `git diff`, and lightweight link/path validation.

---

## File Structure

- Modify: `docs/plan/plan_big_picture_2026-04-30.md`
  - Owns the final two-horizon strategy.
  - Should become the source a future agent can read to understand immediate paper priorities and longer-term research sequence.
- Reference only: `docs/superpowers/specs/2026-04-30-big-picture-two-horizon-plan-design.md`
  - Approved design contract for this implementation plan.
- Reference only: `memory/experiments/2026-04-20_issue_791_distribution_alignment_dominates.md`
  - Evidence that PPO distribution alignment dominates the observed lift.
- Reference only: `memory/decisions/2026-04-20_issue_791_narrow_benchmark_claim.md`
  - Evidence that paper language must be benchmark-set scoped rather than OOD/generalization scoped.
- Reference only: `docs/context/dreamerv3_program_close_out_2026_04_30.md`
  - Evidence that DreamerV3 is deprioritized for camera-ready scope.
- Reference only: `docs/benchmark_planner_family_coverage.md`
  - Evidence for planner-family claim boundaries.
- Reference only: `docs/benchmark_camera_ready.md`
  - Evidence for camera-ready benchmark, SNQI, and artifact obligations.
- Reference only: `docs/context/issue_691_benchmark_fallback_policy.md`
  - Evidence that fallback/degraded execution is a limitation, not benchmark-strengthening proof.

## Task 1: Confirm Source Context Before Editing

**Files:**
- Read: `docs/superpowers/specs/2026-04-30-big-picture-two-horizon-plan-design.md`
- Read: `docs/plan/plan_big_picture_2026-04-30.md`
- Read: `memory/experiments/2026-04-20_issue_791_distribution_alignment_dominates.md`
- Read: `memory/decisions/2026-04-20_issue_791_narrow_benchmark_claim.md`
- Read: `docs/context/dreamerv3_program_close_out_2026_04_30.md`
- Read: `docs/benchmark_planner_family_coverage.md`

- [ ] **Step 1: Read the approved design spec**

Run:

```bash
sed -n '1,260p' docs/superpowers/specs/2026-04-30-big-picture-two-horizon-plan-design.md
```

Expected: the output includes `Horizon A: Paper Delivery Plan`, `Horizon B: Research Roadmap`, `Priority Order`, and `Validation Model`.

- [ ] **Step 2: Read the current strategy document**

Run:

```bash
sed -n '1,620p' docs/plan/plan_big_picture_2026-04-30.md
```

Expected: the output shows the current broad strategy, including policy-stack, adversarial, scenario-certification, CARLA, roadmap, and references sections.

- [ ] **Step 3: Read the current PPO and claim-boundary evidence**

Run:

```bash
sed -n '1,180p' memory/experiments/2026-04-20_issue_791_distribution_alignment_dominates.md
sed -n '1,180p' memory/decisions/2026-04-20_issue_791_narrow_benchmark_claim.md
```

Expected: the first output states distribution alignment dominates the PPO lift; the second states the paper claim is a strong benchmark-set result, not OOD generalization.

- [ ] **Step 4: Read the DreamerV3 and planner-family boundaries**

Run:

```bash
sed -n '1,220p' docs/context/dreamerv3_program_close_out_2026_04_30.md
sed -n '1,150p' docs/benchmark_planner_family_coverage.md
```

Expected: the DreamerV3 note says the track is closed/deprioritized for paper scope; the planner-family matrix distinguishes benchmarkable, experimental, conceptually adjacent, and missing planner families.

- [ ] **Step 5: Inspect the working tree before edits**

Run:

```bash
git status --short
```

Expected: only this implementation plan may be uncommitted if the plan has not yet been committed. Do not revert unrelated user changes.

## Task 2: Replace the Top-Level Thesis and Evidence Summary

**Files:**
- Modify: `docs/plan/plan_big_picture_2026-04-30.md`

- [ ] **Step 1: Replace the opening with the two-horizon thesis**

Edit the beginning of `docs/plan/plan_big_picture_2026-04-30.md` so it starts with this structure:

````markdown
# Robot-SF two-horizon improvement strategy - 2026-04-30

## Core recommendation

Robot-SF should optimize for two linked horizons:

1. **Near-term paper delivery:** protect a defensible camera-ready benchmark story with clear
   provenance, SNQI contract evidence, baseline boundaries, and durable artifacts.
2. **Long-term research roadmap:** build a certified, adversarially tested, layered local-navigation
   stack after the benchmark claim is stable.

The near-term target is not "one best local policy." It is a falsifiable benchmark claim:

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
````

Expected: the document no longer opens by implying that policy-stack construction is the immediate first priority.

- [ ] **Step 2: Add a current-evidence section after the core recommendation**

Insert this section immediately after the opening thesis:

```markdown
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
```

Expected: the plan explicitly names the repository evidence that governs the rewrite.

- [ ] **Step 3: Run a focused opening-section review**

Run:

```bash
sed -n '1,120p' docs/plan/plan_big_picture_2026-04-30.md
```

Expected: the first 120 lines contain the two-horizon thesis and the current-evidence section.

## Task 3: Build Horizon A for Paper Delivery

**Files:**
- Modify: `docs/plan/plan_big_picture_2026-04-30.md`

- [ ] **Step 1: Add the Horizon A section**

Create or replace the near-term roadmap area with this section:

```markdown
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
```

Expected: the paper track appears before adversarial, policy-stack, and CARLA roadmap sections.

- [ ] **Step 2: Add Horizon A proof obligations**

Append this subsection to Horizon A:

```markdown
### Horizon A proof obligations

| Claim type | Required proof |
| --- | --- |
| PPO benchmark result | canonical config, model artifact provenance, benchmark run, seed/bootstrap evidence, SNQI diagnostics |
| Baseline planner result | implemented benchmark entrypoint, dependency availability, native/adapter mode, non-fallback execution |
| SNQI conclusion | versioned weights, baseline normalization assets, contract diagnostics, sensitivity or ablation status |
| Release artifact | durable artifact reference or tracked manifest, plus enough metadata to recover the exact input |
| Failure attribution | scenario validity, planner mode, route/geometry sanity, and reproducible episode or counterexample bundle |
```

Expected: the document defines what evidence must exist before a paper-facing claim is complete.

## Task 4: Reframe Long-Term Research as Horizon B

**Files:**
- Modify: `docs/plan/plan_big_picture_2026-04-30.md`

- [ ] **Step 1: Convert the existing policy-stack, adversarial, scenario, and CARLA content into Horizon B**

Create a `## Horizon B - long-term research roadmap` section after Horizon A. Use this opening:

```markdown
## Horizon B - long-term research roadmap

After the paper track is stable, Robot-SF should become a fast falsification and policy-development
platform. The roadmap remains a layered local-navigation stack, but every promotion step must pass
through executable Robot-SF evidence before it supports a manuscript or benchmark claim.
```

Expected: the long-term content is explicitly sequenced after the paper track.

- [ ] **Step 2: Preserve scenario certification as the first Horizon B workstream**

Include this subsection:

```markdown
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
```

Expected: adversarial generation no longer precedes scenario validity.

- [ ] **Step 3: Preserve adversarial falsification as the second Horizon B workstream**

Include this subsection:

```markdown
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
```

Expected: learned adversaries are not framed as immediate paper work.

- [ ] **Step 4: Preserve policy-stack work as the third Horizon B workstream**

Include this subsection:

````markdown
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
````

Expected: the policy stack remains in the roadmap but is no longer the first near-term priority.

- [ ] **Step 5: Preserve CARLA as a gated transfer track**

Include this subsection:

```markdown
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
perception and CARLA training should come after replay parity.
```

Expected: CARLA work is useful but explicitly deferred behind Robot-SF contracts.

## Task 5: Add Sequencing and Issue Candidate Tables

**Files:**
- Modify: `docs/plan/plan_big_picture_2026-04-30.md`

- [ ] **Step 1: Add a sequencing table**

Add this section after Horizon B:

```markdown
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
```

Expected: a future agent can identify what to do now, next, and later.

- [ ] **Step 2: Replace the issue candidate section with paper-risk and research-follow-up groups**

Use this structure:

```markdown
## Issue candidates

### Paper-critical or paper-risk issues

1. Audit issue-791 claim language against the narrow benchmark-set decision.
2. Verify camera-ready PPO provenance, benchmark command, seed/bootstrap evidence, and SNQI
   diagnostics.
3. Audit baseline planner dependencies and native/adapter/fallback modes for the paper matrix.
4. Verify durable artifact references for model, benchmark, SNQI, and release inputs.
5. Resolve or caveat route handoff, invalid geometry, metric drift, and fallback/degraded execution
   before using affected failures as policy evidence.

### Research follow-up issues

1. Add `scenario_cert.v1` for geometric, kinodynamic, dynamic-agent, and difficulty certification.
2. Add adversarial scenario search with plausibility constraints and counterexample replay bundles.
3. Extend adversarial pedestrians from single-agent templates to constrained multi-agent stress
   tests.
4. Add obstacle-conditioned prediction baselines and scene representations.
5. Build `policy_stack_v1` as a portfolio planner with safety shielding and controller diagnostics.
6. Add CARLA oracle replay/parity for certified Robot-SF scenarios.
```

Expected: the issue list no longer mixes paper-critical work and exploratory research.

## Task 6: Clean Up Conflicting or Stale Language

**Files:**
- Modify: `docs/plan/plan_big_picture_2026-04-30.md`

- [ ] **Step 1: Search for public-claim risk words**

Run:

```bash
rg -n "generaliz|transfer|unseen|OOD|DreamerV3|best local policy|CARLA training|fallback" docs/plan/plan_big_picture_2026-04-30.md
```

Expected: every match is either in a caveat, a long-term roadmap section, or a sentence that explicitly says the claim is out of near-term paper scope.

- [ ] **Step 2: Rewrite risky phrases**

Apply these replacements where the current wording makes near-term claims too broad:

```text
"best local policy" -> "defensible benchmark-set policy and long-term local-policy roadmap"
"transfer to CARLA" -> "CARLA oracle replay/parity after Robot-SF contracts exist"
"OOD generalization" -> "out-of-scope OOD generalization unless a separate study is run"
"DreamerV3 challenger" -> "DreamerV3 historical/deprioritized track"
"fallback success" -> "fallback/degraded caveat or exclusion"
```

Expected: the plan does not encourage public claims that conflict with current memory decisions.

- [ ] **Step 3: Search for stale replacement targets**

Run:

```bash
rg -n "best local policy|CARLA training|DreamerV3 challenger|fallback success" docs/plan/plan_big_picture_2026-04-30.md
```

Expected: no matches remain.

## Task 7: Validate the Revised Document

**Files:**
- Validate: `docs/plan/plan_big_picture_2026-04-30.md`

- [ ] **Step 1: Check referenced repository paths exist**

Run:

```bash
while read -r path; do test -e "$path" || echo "missing: $path"; done <<'EOF'
docs/plan/plan_big_picture_2026-04-30.md
docs/superpowers/specs/2026-04-30-big-picture-two-horizon-plan-design.md
memory/experiments/2026-04-20_issue_791_distribution_alignment_dominates.md
memory/decisions/2026-04-20_issue_791_narrow_benchmark_claim.md
docs/context/dreamerv3_program_close_out_2026_04_30.md
docs/benchmark_planner_family_coverage.md
docs/benchmark_camera_ready.md
docs/benchmark_release_protocol.md
docs/benchmark_release_reproducibility.md
docs/context/issue_691_benchmark_fallback_policy.md
EOF
```

Expected: no `missing:` lines.

- [ ] **Step 2: Search for unfinished markers in the revised strategy**

Run:

```bash
rg -n "T""BD|TO""DO|FIX""ME|place""holder|implement la""ter|fill in det""ails" docs/plan/plan_big_picture_2026-04-30.md
```

Expected: no matches.

- [ ] **Step 3: Review the final diff**

Run:

```bash
git diff -- docs/plan/plan_big_picture_2026-04-30.md
```

Expected: the diff shows a documentation-only rewrite that adds the two-horizon structure and removes or demotes stale near-term claims.

- [ ] **Step 4: Run a lightweight docs sanity check**

Run:

```bash
uv run ruff check .
```

Expected: PASS. If this command reports unrelated pre-existing failures, record the exact failures in the final handoff and do not fix unrelated files.

## Task 8: Commit the Plan Revision

**Files:**
- Commit: `docs/plan/plan_big_picture_2026-04-30.md`

- [ ] **Step 1: Inspect status**

Run:

```bash
git status --short
```

Expected: `docs/plan/plan_big_picture_2026-04-30.md` is modified. This implementation plan may also be present if it has not been committed separately.

- [ ] **Step 2: Commit only the intended documentation files**

Run:

```bash
git add docs/plan/plan_big_picture_2026-04-30.md docs/superpowers/plans/2026-04-30-big-picture-two-horizon-plan.md
git commit -m "docs: refine big-picture plan into two horizons"
```

Expected: commit succeeds. Pre-commit hooks may skip code-only checks for Markdown-only changes.

- [ ] **Step 3: Capture final status**

Run:

```bash
git status --short
```

Expected: no uncommitted changes from this implementation remain.
