# Skill Consolidation Audit Issue #3189 2026-06-20

Issue: [#3189](https://github.com/ll7/robot_sf_ll7/issues/3189)

Evidence status: `analysis-only`. This note audits skill overlap and proposes follow-up
consolidation work. It does not change skill routing behavior.

Scope boundary:

- Reviewed `.agents/skills/*/SKILL.md` frontmatter and workflow bodies on `origin/main` commit
  `c402d910c905f3e28cb24816911983e06e7f42e4`.
- Cross-checked `.agents/skills/skills.yaml` and `.agents/skills/README.md`.
- Built best-effort usage evidence from tracked references, common-Git-dir
  `.git/codex-agent-runs/notes/inbox/`, and `git log -1 -- .agents/skills/<skill>/SKILL.md`.
- No `SKILL.md` or `.agents/skills/skills.yaml` file is modified by this audit.

Usage-evidence caveat: no canonical invocation-count log was found. `refs` below is a rough
tracked-reference count from `docs/context`, `memory`, `.agents`, and `AGENTS.md`, excluding the
skill's own `SKILL.md`. It is not a usage count.

## Method

1. Parsed all direct `.agents/skills/*/SKILL.md` files with YAML frontmatter.
2. Parsed `skills.yaml` registry metadata and compared names, categories, phases, write flags,
   delegated skills, aliases, and descriptions.
3. Read workflow sections for overlap clusters surfaced by metadata and text similarity.
4. Computed a TF-IDF cosine screen over description, category, kind, phase, `delegates_to`, and
   workflow body to surface non-obvious overlaps.
5. Assigned one canonical verdict to every skill: `keep`, `merge-into <skill>`,
   `deprecate-alias`, or `retire`.

## Similarity Matrix

The matrix is intentionally compact: it lists the highest-scoring pairwise overlaps from the
53-skill corpus, plus the manual cluster used for recommendation review.

| Score | Skill A | Skill B | Cluster |
| ---: | --- | --- | --- |
| 0.682 | `oracle-imitation-campaign` | `predictive-planner-comparison` | campaign-analysis |
| 0.677 | `adversarial-search-campaign` | `predictive-planner-comparison` | campaign-analysis |
| 0.643 | `hybrid-learning-component-eval` | `predictive-planner-comparison` | campaign-analysis |
| 0.627 | `adversarial-search-campaign` | `oracle-imitation-campaign` | campaign-analysis |
| 0.625 | `carla-replay-parity` | `oracle-imitation-campaign` | campaign-analysis |
| 0.618 | `carla-replay-parity` | `predictive-planner-comparison` | campaign-analysis |
| 0.605 | `auxme-issue791-submit` | `auxme-slurm-reliable-submit` | SLURM-submit |
| 0.600 | `adversarial-search-campaign` | `hybrid-learning-component-eval` | campaign-analysis |
| 0.546 | `goal-slurm-experiment` | `slurm-campaign-submit` | SLURM-submit |
| 0.486 | `predictive-planner-comparison` | `trace-mechanism-review` | campaign-analysis |
| 0.462 | `gh-issue-autopilot` | `goal-issue-implementation` | issue-to-PR |
| 0.459 | `goal-autopilot` | `goal-issue-implementation` | autonomous-goals |
| 0.432 | `goal-issue-implementation` | `goal-pr-review` | autonomous-goals |
| 0.419 | `agent-workflow-capture` | `agent-workflow-promotion` | workflow-lessons |
| 0.412 | `gh-issue-sequencer` | `issue-audit` | issue-grooming |
| 0.404 | `evidence-synthesis` | `paper-facing-docs` | evidence-writing |
| 0.360 | `clean-up` | `pr-ready-check` | validation |
| 0.356 | `auto-improvement` | `autoresearch` | research-iteration |
| 0.349 | `issue-contract-maintainer` | `issue-splitter` | issue-grooming |
| 0.343 | `issue-audit` | `issue-contract-maintainer` | issue-grooming |
| 0.341 | `gh-issue-priority-assessor` | `gh-issue-sequencer` | issue-grooming |
| 0.334 | `gh-issue-template-auditor` | `issue-contract-maintainer` | issue-grooming |
| 0.309 | `gh-issue-autopilot` | `gh-pr-opener` | issue-to-PR |
| 0.301 | `analyze-camera-ready-benchmark` | `paper-facing-docs` | evidence-writing |
| 0.288 | `auxme-issue791-submit` | `slurm-campaign-submit` | SLURM-submit |
| 0.255 | `benchmark-overview` | `experiment-context` | context |
| 0.250 | `planner-integration` | `review-benchmark-change` | benchmark-review |

## Overlap Clusters

| Cluster | Members | Distinct capability found | Recommendation |
| --- | --- | --- | --- |
| Issue contract and grooming | `issue-contract-maintainer`, `gh-issue-clarifier`, `gh-issue-template-auditor`, `issue-audit`, `gh-issue-priority-assessor`, `gh-issue-sequencer`, `issue-splitter`, `gh-issue-creator` | Creator and splitter own distinct creation/splitting contracts. Sequencer owns Project queue normalization. Contract-maintainer already owns ambiguity/template/user-decision modes. Clarifier, template-auditor, and issue-audit mostly duplicate modes that can live in contract-maintainer. Priority-assessor overlaps sequencer's priority discussion mode. | Keep `issue-contract-maintainer`, `gh-issue-sequencer`, `gh-issue-creator`, and `issue-splitter`; merge the lower-level issue repair/audit helpers into those canonical surfaces. |
| Issue-to-PR orchestration | `goal-issue-implementation`, `gh-issue-autopilot`, `goal-autopilot` | `goal-autopilot` is continuous loop orchestration; `goal-issue-implementation` is issue queue loop; `gh-issue-autopilot` is one selected issue to PR and overlaps strongly with `goal-issue-implementation` implementation state. | Keep `goal-autopilot` and `goal-issue-implementation`; merge `gh-issue-autopilot` into `goal-issue-implementation` as selected-issue mode. |
| PR lifecycle | `goal-pr-review`, `gh-pr-comment-fixer`, `gh-pr-opener`, `gh-pr-merger` | Review loop, comment fixing, opening, and guarded merge are separate GitHub actions with different mutation boundaries. | Keep all. |
| Campaign-analysis templates | `adversarial-search-campaign`, `carla-replay-parity`, `hybrid-learning-component-eval`, `oracle-imitation-campaign`, `predictive-planner-comparison`, `trace-mechanism-review` | Body shape is intentionally shared, but each skill names different command surfaces, provenance fields, and evidence traps. | Keep all for now; follow up with a shared campaign-analysis template to reduce duplicated prose without retiring domain entry points. |
| Benchmark and evidence writing | `analyze-camera-ready-benchmark`, `analyze-latest-policy-sweep`, `benchmark-overview`, `benchmark-row-status`, `artifact-provenance`, `data-staging-provenance`, `evidence-synthesis`, `paper-facing-docs`, `planner-integration`, `review-benchmark-change` | Row status, artifact handling, staging, synthesis, and paper-facing review are distinct. `analyze-latest-policy-sweep` is narrower and substantially overlaps camera-ready campaign analysis. | Merge `analyze-latest-policy-sweep` into `analyze-camera-ready-benchmark`; keep the rest. |
| SLURM submitters | `slurm-campaign-submit`, `auxme-issue791-submit`, `auxme-slurm-reliable-submit`, `goal-slurm-experiment` | Generic campaign submission, Auxme issue-791 private wrapper routing, reliable Auxme retry/preflight, and one-active-experiment orchestration are separate levels. `auxme-issue791-submit` is narrower than the reliable Auxme skill. | Keep `slurm-campaign-submit`, `auxme-slurm-reliable-submit`, and `goal-slurm-experiment`; merge `auxme-issue791-submit` into the reliable Auxme skill. |
| Research iteration | `autoresearch`, `auto-improvement`, `agentic-eval`, `agent-workflow-capture`, `agent-workflow-promotion` | Baseline/experiment loops, small refinements, AI-output evals, private lesson capture, and durable promotion have separate stop rules. | Keep all. |
| Context and docs | `context-map`, `what-context-needed`, `context-note-maintainer`, `update-docs-on-code-change`, `experiment-context`, `skill-picker` | `context-map` maps surfaces, `what-context-needed` asks blockers, note maintainer creates durable notes, docs updater keeps implementation docs aligned, experiment-context finds runnable config evidence, and skill-picker remains useful until consolidation reduces routing load. | Keep all, but revisit `skill-picker` after the merge follow-ups land. |
| Validation and cleanup | `implementation-verification`, `pr-ready-check`, `quality-playbook`, `clean-up`, `review-and-refactor` | Claim proof, canonical readiness, risk-proportional validation planning, mutating branch cleanup, and surgical refactor workflow are distinct. | Keep all. |
| Domain utility | `svg-inspection` | Unique SVG parser/map diagnostic workflow. | Keep. |

## Inventory And Verdicts

Every direct `.agents/skills/*/SKILL.md` skill on the audited commit appears exactly once below.

| Skill | Category | Kind | Phase | Flags | Delegates | Description | Evidence | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `adversarial-search-campaign` | campaign-analysis | analysis | analysis | write, artifacts | `benchmark-row-status`, `artifact-provenance`, `evidence-synthesis` | Analyze adversarial route/search campaigns with canonical commands, expected artifacts, status boundaries, and claim limits. | refs 4; last edit 2026-06-01 | keep: domain command surfaces are distinct despite shared campaign template. |
| `agent-workflow-capture` | research-iteration | atomic | analysis | write | none | Capture private candidate lessons from agent execution into `.git/codex-agent-runs/notes/inbox/` when a repeatable workflow, routing, validation, tooling, or instruction improvement is noticed. | refs 5; last edit 2026-06-01 | keep: owns private candidate capture, not durable promotion. |
| `agent-workflow-promotion` | research-iteration | orchestrator | implementation | write | `review-and-refactor`, `update-docs-on-code-change`, `context-note-maintainer`, `gh-issue-creator` | Promote accumulated private workflow lessons into small, evidence-backed repository changes. | refs 5; last edit 2026-06-01 | keep: owns durable promotion from private notes. |
| `agentic-eval` | research-iteration | atomic | analysis | write | none | Evaluate and improve AI workflow outputs with small goldens, rubrics, and repeatable checks. | refs 3; last edit 2026-06-01 | keep: distinct AI-output evaluation loop. |
| `analyze-camera-ready-benchmark` | benchmark-evidence | atomic | context | artifacts | none | Analyze camera-ready benchmark campaigns for consistency, runtime hotspots, fallback/degraded planners, and reproducibility metadata. | refs 5; last edit 2026-06-12 | keep: broader benchmark campaign analyzer should absorb policy-sweep variant. |
| `analyze-latest-policy-sweep` | benchmark-evidence | atomic | context | artifacts | none | Analyze latest policy analysis sweep runs by comparing episodes, summaries, diagnostics, and videos. | refs 2; last edit 2026-06-01 | merge-into `analyze-camera-ready-benchmark`: narrower sweep review overlaps campaign analysis workflow. |
| `artifact-provenance` | benchmark-evidence | atomic | verification | write, artifacts | none | Classify, promote, or document generated artifacts so durable evidence is separated from local output caches. | refs 23; last edit 2026-06-01 | keep: high-reference durable artifact boundary. |
| `auto-improvement` | research-iteration | atomic | analysis | write | none | Focused measurement-aware refinement loop for Robot SF prompts, docs, and small code changes. | refs 4; last edit 2026-06-01 | keep: narrow refinement loop distinct from full experiments. |
| `autoresearch` | research-iteration | atomic | analysis | write | none | Autonomous iterative experimentation loop for measurable tasks with baseline, experiments, and keep/discard decisions. | refs 47; last edit 2026-06-01 | keep: heavily referenced measurable experiment loop. |
| `auxme-issue791-submit` | slurm | atomic | implementation | write, slurm | `slurm-campaign-submit` | Submit issue-791-specific Auxme training jobs with explicit config provenance and wrapper-safety checks. | refs 6; last edit 2026-06-09 | merge-into `auxme-slurm-reliable-submit`: issue-791 path is narrower than the reliable Auxme wrapper. |
| `auxme-slurm-reliable-submit` | slurm | atomic | implementation | write, slurm | `auxme-issue791-submit` | Submit issue-791 style Auxme SLURM jobs with live partition pressure checks and max-time-safe routing. | refs 2; last edit 2026-06-09 | keep: private Auxme reliability guard remains distinct from generic SLURM. |
| `benchmark-overview` | benchmark-evidence | atomic | context | artifacts | none | Fast benchmark-faithful orientation for scenario splits, baselines, metrics, artifacts, and reproducibility constraints. | refs 6; last edit 2026-06-01 | keep: read-only orientation surface. |
| `benchmark-row-status` | benchmark-evidence | policy | analysis | artifacts | none | Classify benchmark campaign rows under the fail-closed policy. | refs 13; last edit 2026-06-01 | keep: policy primitive used by evidence skills. |
| `carla-replay-parity` | campaign-analysis | analysis | analysis | write, artifacts | `benchmark-row-status`, `artifact-provenance`, `evidence-synthesis` | Review CARLA replay parity evidence with scenario, replay, metric, and limitation tracking. | refs 4; last edit 2026-06-01 | keep: CARLA-specific command and caveat surface. |
| `clean-up` | validation | atomic | verification | write | none | Clean up the current branch using repo-standard formatting, tests, and diff gates. | refs 7; last edit 2026-06-12 | keep: mutating cleanup workflow differs from readiness proof. |
| `context-map` | context-docs | atomic | context | none | none | Generate a focused repository context map before multi-file changes. | refs 7; last edit 2026-06-01 | keep: maps surfaces before execution. |
| `context-note-maintainer` | context-docs | atomic | context | write | none | Create or refresh linked docs/context notes so reusable knowledge stays discoverable. | refs 17; last edit 2026-06-01 | keep: canonical context-note workflow. |
| `data-staging-provenance` | benchmark-evidence | atomic | planning | write, artifacts | `artifact-provenance` | Stage external datasets and assets with checksum, license, raw-file, derived-file, and benchmark-readiness provenance. | refs 5; last edit 2026-06-01 | keep: external-data staging has distinct license/checksum concerns. |
| `evidence-synthesis` | benchmark-evidence | analysis | analysis | write, artifacts | `artifact-provenance`, `benchmark-row-status`, `paper-facing-docs` | Synthesize multiple issues, configs, seeds, metrics, and artifacts into conservative mechanism-level conclusions. | refs 13; last edit 2026-06-12 | keep: synthesis layer differs from campaign-specific analysis. |
| `experiment-context` | context-docs | atomic | context | none | none | Find canonical config-first training or evaluation paths, artifact lineage, and validation gates. | refs 7; last edit 2026-06-01 | keep: experiment-path discovery is distinct from generic context map. |
| `gh-issue-autopilot` | github-issue | orchestrator | implementation | write | `gh-issue-sequencer`, `implementation-verification`, `pr-ready-check`, `gh-pr-opener`, `artifact-provenance` | Autonomous issue-to-PR workflow from next eligible issue to ready PR. | refs 11; last edit 2026-06-12 | merge-into `goal-issue-implementation`: selected-issue implementation can be a mode of the queue runner. |
| `gh-issue-clarifier` | github-issue | atomic | context | write | none | Clarify ambiguous GitHub issues by tightening scope and acceptance criteria. | refs 9; last edit 2026-06-01 | merge-into `issue-contract-maintainer`: this is already one contract-maintenance mode. |
| `gh-issue-creator` | github-issue | atomic | context | write | none | Create structured GitHub issues from vague prompts using repo templates and conservative assumptions. | refs 11; last edit 2026-06-01 | keep: creation is a distinct mutation boundary. |
| `gh-issue-priority-assessor` | github-issue | atomic | context | write | none | Review Project #5 priority inputs, propose values with uncertainty, and route tradeoffs. | refs 7; last edit 2026-06-01 | merge-into `gh-issue-sequencer`: priority scoring belongs with queue ordering. |
| `gh-issue-sequencer` | github-issue | atomic | context | write | none | Maintain a clear next-work queue in GitHub Project #5. | refs 11; last edit 2026-06-01 | keep: owns queue normalization and priority discussion. |
| `gh-issue-template-auditor` | github-issue | atomic | context | write | none | Review existing GitHub issues against the issue-template contract and repair clear gaps. | refs 4; last edit 2026-06-01 | merge-into `issue-contract-maintainer`: template audit is a named mode there. |
| `gh-pr-comment-fixer` | github-pr | atomic | context | write | none | Fix GitHub PR review comments with branch-safe edits and validation. | refs 7; last edit 2026-06-17 | keep: focused review-comment execution. |
| `gh-pr-merger` | github-pr | atomic | verification | write | none | Guarded PR merger after label, CI, branch protection, and preflight checks. | refs 5; last edit 2026-06-11 | keep: merge mutation should remain isolated. |
| `gh-pr-opener` | github-pr | atomic | context | write | none | Open a conservative Robot SF PR with scope verification, freshness checks, and artifact discipline. | refs 8; last edit 2026-06-01 | keep: PR creation mutation boundary. |
| `goal-autopilot` | general | orchestrator | implementation | write | `goal-issue-implementation`, `gh-pr-merger`, `goal-pr-review`, `goal-issue-discovery` | Continuous implement-review-merge-discover loop with preflight validation and delegation recovery. | refs 8; last edit 2026-06-15 | keep: top-level continuous loop. |
| `goal-issue-discovery` | github-issue | orchestrator | analysis | write | `gh-issue-creator`, `gh-issue-sequencer`, `gh-issue-priority-assessor`, `agentic-eval`, `auto-improvement`, `autoresearch`, `context-map` | Autonomous issue-discovery loop that creates evidence-graded GitHub issues. | refs 7; last edit 2026-06-01 | keep: discovery is distinct from implementation. |
| `goal-issue-implementation` | github-issue | orchestrator | implementation | write | `gh-issue-sequencer`, `gh-issue-autopilot`, `implementation-verification`, `pr-ready-check`, `gh-pr-opener`, `gh-issue-creator`, `context-note-maintainer`, `issue-splitter` | Autonomous issue-to-PR loop selecting eligible issues, implementing, validating, pushing, and opening PRs. | refs 18; last edit 2026-06-17 | keep: canonical implementation loop and merge target for `gh-issue-autopilot`. |
| `goal-pr-review` | github-pr | orchestrator | verification | write | `implementation-verification`, `pr-ready-check`, `gh-pr-comment-fixer`, `review-benchmark-change`, `gh-issue-creator`, `context-note-maintainer` | Autonomous PR review loop that fixes scoped review gaps, validates proof, resolves threads, and applies merge-ready. | refs 8; last edit 2026-06-17 | keep: review loop is distinct from merge and issue implementation. |
| `goal-slurm-experiment` | research-iteration | orchestrator | implementation | write, slurm | `experiment-context`, `goal-issue-implementation`, `slurm-campaign-submit`, `artifact-provenance`, `context-note-maintainer` | Keep one skill-owned learning or training SLURM job active. | refs 7; last edit 2026-06-19 | keep: orchestration across issue gaps and job submission. |
| `hybrid-learning-component-eval` | campaign-analysis | analysis | analysis | write, artifacts | `artifact-provenance`, `benchmark-row-status`, `evidence-synthesis` | Evaluate hybrid-learning components with mechanism-level evidence tables, config provenance, and claim boundaries. | refs 3; last edit 2026-06-01 | keep: hybrid-learning component evidence has distinct fields. |
| `implementation-verification` | validation | atomic | verification | none | none | Verify branch changes against `origin/main` with claim-based evidence. | refs 12; last edit 2026-06-01 | keep: claim-to-proof mapping is distinct from test running. |
| `issue-audit` | github-issue | atomic | context | write | none | User-in-the-loop open-issue audit asking one readiness-blocking or priority-tradeoff question at a time. | refs 10; last edit 2026-06-12 | merge-into `issue-contract-maintainer`: user-decision and priority discussion can be modes of contract maintenance. |
| `issue-contract-maintainer` | github-issue | orchestrator | planning | write | `gh-issue-clarifier`, `gh-issue-template-auditor`, `issue-audit`, `issue-splitter` | Maintain GitHub issue contracts through template audits, ambiguity clarification, and user-decision application. | refs 13; last edit 2026-06-12 | keep: canonical issue-contract surface and merge target. |
| `issue-splitter` | github-issue | atomic | planning | write | `gh-issue-creator` | Split a parent, epic, decision, or research issue into the smallest independently implementable child. | refs 5; last edit 2026-06-05 | keep: child-issue extraction contract is distinct. |
| `oracle-imitation-campaign` | campaign-analysis | analysis | analysis | write, artifacts | `artifact-provenance`, `benchmark-row-status`, `evidence-synthesis` | Analyze oracle-imitation campaign outputs with lineage, dataset, checkpoint, metric, and caveat discipline. | refs 2; last edit 2026-06-01 | keep: dataset/checkpoint lineage fields are distinct. |
| `paper-facing-docs` | benchmark-evidence | atomic | context | artifacts | none | Draft or review benchmark and manuscript-support docs with provenance, reproducibility, and caveats. | refs 8; last edit 2026-06-12 | keep: public-claim wording gate. |
| `planner-integration` | benchmark-evidence | atomic | context | artifacts | none | Assess planner-family integration feasibility, adapter burden, provenance safety, and benchmark-readiness boundaries. | refs 4; last edit 2026-06-01 | keep: planner integration feasibility is distinct. |
| `pr-ready-check` | validation | atomic | verification | none | none | Run the repository PR readiness pipeline using shared scripts/dev entry points. | refs 12; last edit 2026-06-17 | keep: canonical readiness gate. |
| `predictive-planner-comparison` | campaign-analysis | analysis | analysis | write, artifacts | `benchmark-row-status`, `artifact-provenance`, `evidence-synthesis` | Compare predictive planner v2 runs against baseline planners with config, seed, artifact, and failure-mode discipline. | refs 2; last edit 2026-06-01 | keep: predictive-planner fields and commands are distinct. |
| `quality-playbook` | validation | policy | verification | none | none | Repo-wide risk-proportional validation workflow for non-trivial changes. | refs 3; last edit 2026-06-01 | keep: validation planning policy, not a command runner. |
| `review-and-refactor` | validation | atomic | verification | write | none | Surgical review-then-refactor workflow for small code or docs changes. | refs 5; last edit 2026-06-01 | keep: narrow review/refactor loop. |
| `review-benchmark-change` | benchmark-evidence | atomic | context | artifacts | none | Review benchmark-sensitive code or docs changes for semantic regressions and provenance overclaim. | refs 9; last edit 2026-06-01 | keep: benchmark-sensitive review checklist. |
| `skill-picker` | context-docs | atomic | context | none | none | Choose the most appropriate repo-local skill for an ambiguous task. | refs 3; last edit 2026-06-01 | keep: temporary router remains useful until consolidation follow-ups land. |
| `slurm-campaign-submit` | slurm | atomic | implementation | write, slurm | `artifact-provenance` | Submit generic SLURM campaigns with preflight, config provenance, job metadata, artifact expectations, and failure classification. | refs 9; last edit 2026-06-19 | keep: generic submission contract and possible future Auxme merge target. |
| `svg-inspection` | domain-utility | atomic | analysis | none | none | Inspect and debug SVG maps for parser-facing issues using reusable Robot SF helpers. | refs 2; last edit 2026-06-01 | keep: unique map/SVG diagnostic workflow. |
| `trace-mechanism-review` | campaign-analysis | analysis | analysis | write, artifacts | `artifact-provenance`, `evidence-synthesis` | Review exact planner/scenario/seed/episode traces and videos without overgeneralizing. | refs 5; last edit 2026-06-05 | keep: trace-level mechanism review has unique no-population-inference guardrail. |
| `update-docs-on-code-change` | context-docs | atomic | context | write | none | Keep docs aligned with code changes that affect workflows, contracts, or user-facing behavior. | refs 6; last edit 2026-06-01 | keep: implementation-doc drift repair differs from context-note writing. |
| `what-context-needed` | context-docs | atomic | context | none | none | Ask for the minimum repository context needed to answer or implement a task safely. | refs 3; last edit 2026-06-01 | keep: blocker-question workflow distinct from context mapping. |

## Proposed Follow-Up Merge Issues

These are proposed issue bodies, not created issues.

1. Merge selected-issue PR execution into `goal-issue-implementation`.
   - Scope: fold `gh-issue-autopilot` workflow into a selected-issue mode, preserve aliases
     `issue-to-pr` and `gh-issue-to-pr`, update README/registry/mirrors, and run full skill sync.
   - Risk: high workflow surface; needs before/after routing tests.
2. Fold issue contract repair atomics into `issue-contract-maintainer`.
   - Scope: merge `gh-issue-clarifier`, `gh-issue-template-auditor`, and `issue-audit` as modes.
     Preserve explicit aliases for compatibility.
   - Risk: moderate GitHub mutation policy risk; validate with routing cases and dry-run issue
     snapshots.
3. Fold Project priority scoring into `gh-issue-sequencer`.
   - Scope: move `gh-issue-priority-assessor` rubric and uncertainty language into sequencer
     priority discussion mode.
   - Risk: medium; Project #5 scoring remains advisory and quota-sensitive.
4. Merge latest policy-sweep analysis into camera-ready benchmark analysis.
   - Scope: make `analyze-camera-ready-benchmark` accept policy-sweep roots and retire the narrower
     `analyze-latest-policy-sweep` skill.
   - Risk: low to medium; must preserve video-artifact notes and fallback/degraded caveat order.
5. Consolidate issue-791 Auxme wrapper routing.
   - Scope: merge `auxme-issue791-submit` into `auxme-slurm-reliable-submit` while keeping generic
     `slurm-campaign-submit` separate.
   - Risk: medium; private Auxme overlay and explicit config guardrails must remain intact.
6. Extract shared campaign-analysis template without retiring domain skills.
   - Scope: factor the repeated row-status/artifact/evidence-synthesis workflow into a short shared
     reference for campaign-analysis skills.
   - Risk: low; reduces duplicated prose without changing entry points.

## Validation Plan

Required for this docs-only audit:

```bash
git diff --check
UV_NO_SYNC=1 uv run python scripts/dev/check_skills.py --preflight goal-autopilot
UV_NO_SYNC=1 uv run python scripts/tools/sync_ai_config.py --check
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
```

Broader `pr_ready_check` is not required for this issue because no runtime code, skill behavior,
registry metadata, or generated mirror content changes in this audit.
