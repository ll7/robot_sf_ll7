# Robot SF Skills

This directory contains repo-local skills for Coding Agents. Use this README as the
generated routing index; read the specific `SKILL.md` before applying a skill.

## Quick Routing

| User intent | Primary skill | Secondary skill |
| --- | --- | --- |
| Not sure which skill applies | `skill-picker` | none |
| Run continuous implement-review-merge-discover autopilot | `goal-autopilot` | `goal-issue-implementation`, `goal-pr-review`, `gh-pr-merger`, `goal-issue-discovery` |
| Take the next eligible issue to PR | `goal-issue-implementation` | `gh-issue-autopilot` |
| Execute one selected issue to ready PR | `gh-issue-autopilot` | `implementation-verification`, `gh-pr-opener` |
| Clarify or repair issue contracts | `issue-contract-maintainer` | legacy aliases only when explicitly named |
| Fix PR review comments | `gh-pr-comment-fixer` | `pr-ready-check` |
| Merge a PR carrying the merge-ready label | `gh-pr-merger` | `goal-pr-review` |
| Open a ready PR | `gh-pr-opener` | `artifact-provenance` |
| Verify branch claims | `implementation-verification` | `pr-ready-check` |
| Run the standard readiness gate | `pr-ready-check` | none |
| Set up or clean up worktrees | `skill-picker` | `gh-issue-autopilot`, `clean-up`; see `AGENTS.md` |
| Review benchmark output | `analyze-camera-ready-benchmark` | `benchmark-row-status`, `artifact-provenance` |
| Classify benchmark rows | `benchmark-row-status` | `review-benchmark-change` |
| Keep one training SLURM job active | `goal-slurm-experiment` | `goal-issue-implementation`, `slurm-campaign-submit` |
| Submit a generic SLURM campaign | `slurm-campaign-submit` | `artifact-provenance` |
| Submit issue-791 Auxme training | `auxme-issue791-submit` | `slurm-campaign-submit` |
| Stage external data | `data-staging-provenance` | `artifact-provenance` |
| Synthesize evidence across issues | `evidence-synthesis` | `paper-facing-docs` |
| Capture a private workflow lesson | `agent-workflow-capture` | none |
| Promote workflow lessons into repo changes | `agent-workflow-promotion` | `review-and-refactor`, `update-docs-on-code-change` |

## Negative Routing

- Do not use `autoresearch` for ordinary cleanup.
- Do not use `paper-facing-docs` for non-claim documentation.
- Do not use `gh-issue-autopilot` for ambiguous issues; route to `issue-contract-maintainer` first.
- Do not use `auxme-issue791-submit` for non-issue-791 campaigns.
- Do not count fallback or degraded benchmark rows as success evidence; use `benchmark-row-status`.
- Do not cite local `output/` contents as durable evidence; use `artifact-provenance`.

## Canonical Skill Stacks

| Stack | Skills |
| --- | --- |
| Continuous goal autopilot | `goal-autopilot` -> `goal-issue-implementation` -> `goal-pr-review` -> `gh-pr-merger` -> `goal-issue-discovery` |
| Issue queue to PR | `gh-issue-sequencer` -> `gh-issue-autopilot` -> `implementation-verification` -> `pr-ready-check` -> `gh-pr-opener` |
| Guarded PR merge | `goal-pr-review` -> `gh-pr-merger` |
| Issue contract repair | `issue-contract-maintainer` -> `gh-issue-sequencer` |
| PR review cleanup | `gh-pr-comment-fixer` -> `implementation-verification` -> `pr-ready-check` |
| Benchmark evidence audit | `benchmark-row-status` -> `artifact-provenance` -> `evidence-synthesis` |
| Always-on SLURM experiment | `experiment-context` -> `goal-issue-implementation` -> `slurm-campaign-submit` -> `artifact-provenance` -> `context-note-maintainer` |
| Agent workflow improvement | `agent-workflow-capture` -> `agent-workflow-promotion` -> `review-and-refactor` -> `update-docs-on-code-change` |
| SLURM campaign launch | `slurm-campaign-submit` -> `artifact-provenance` |
| External data staging | `data-staging-provenance` -> `artifact-provenance` -> `context-note-maintainer` |

## Validation Tiers

| Tier | Use for | Required proof |
| --- | --- | --- |
| 0 | docs-only, metadata-only | render, link, path, or registry check |
| 1 | local code or CLI behavior | targeted tests plus lint where relevant |
| 2 | planner, metric, benchmark, artifact behavior | targeted tests plus benchmark preflight or sample run |
| 3 | campaign or paper-facing evidence | full provenance, seeds, configs, artifacts, and interpretation note |

## GitHub And Project Policy

- `goal-issue-implementation` owns the multi-issue loop and stop condition.
- `goal-autopilot` owns the continuous implement, review, merge, and discover loop.
- `gh-issue-sequencer` owns Project #5 queue ordering, with current maintainer direction and fresh
  evidence allowed to override score order.
- `gh-issue-autopilot` owns one selected issue -> branch -> validation -> ready PR.
- `gh-pr-merger` owns guarded merge after `goal-pr-review` has established merge-ready proof.
- `gh-issue-creator` owns new issue creation.
- `issue-contract-maintainer` owns ambiguity, template, and decision repair.
- Use Project #5 `Priority Score` as an advisory queue-ordering signal; use
  `gh-issue-priority-assessor` when score inputs need review.
- Batch issue cleanup separately from Project #5 metadata writes; follow
  `docs/context/issue_713_batch_first_issue_workflow.md`.

## Maintenance And Validation

- Edit `.agents/skills/skills.yaml` when adding aliases, categories, routing metadata,
  delegated skills, write scopes, or output schemas.
- Run `uv run python scripts/dev/generate_skills_readme.py` after registry changes.
- Run `uv run python scripts/dev/check_skills.py` after adding, renaming, or removing skills.
- Keep legacy compatibility wrappers only when they reduce routing breakage.
- Keep group notes under `.agents/skills/groups/`; direct children of `.agents/skills/`
  must be real skill directories unless explicitly whitelisted by the checker.

## Generated Skill Index

### Benchmark And Experiment Evidence

| Skill | Kind | Phase | Writes | SLURM | Artifacts | Delegates | Use When |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `analyze-camera-ready-benchmark` | atomic | context | no | no | yes | none | Analyze a camera-ready benchmark campaign for consistency, runtime hotspots, fallback/degraded planners, and reproducibility metadata. |
| `analyze-latest-policy-sweep` | atomic | context | no | no | yes | none | Analyze latest policy analysis sweep runs (*_policy_analysis_*) by comparing episodes/summary metrics, diagnostics, and video artifacts; generate a concise markdown report and optional frame snapshots. |
| `artifact-provenance` | atomic | verification | yes | no | yes | none | Classify, promote, or document generated artifacts so durable evidence is separated from local output caches. |
| `benchmark-overview` | atomic | context | no | no | yes | none | Fast benchmark-faithful orientation for scenario splits, baselines, metrics, artifacts, and reproducibility constraints in robot_sf_ll7. |
| `benchmark-row-status` | policy | analysis | no | no | yes | none | Classify benchmark campaign rows under the fail-closed policy so fallback or degraded execution never counts as successful evidence. |
| `data-staging-provenance` | atomic | planning | yes | no | yes | `artifact-provenance` | Stage external datasets and assets with checksum, license, raw-file, derived-file, and benchmark-readiness provenance. |
| `evidence-synthesis` | analysis | analysis | yes | no | yes | `artifact-provenance`, `benchmark-row-status`, `paper-facing-docs` | Synthesize multiple issues, configs, seeds, metrics, and artifacts into conservative mechanism-level conclusions with caveats. |
| `paper-facing-docs` | atomic | context | no | no | yes | none | Draft or review benchmark and manuscript-support docs conservatively, with explicit provenance, reproducibility, and caveat handling. |
| `planner-integration` | atomic | context | no | no | yes | none | Assess planner-family integration feasibility, adapter burden, provenance safety, and benchmark-readiness boundaries in robot_sf_ll7. |
| `review-benchmark-change` | atomic | context | no | no | yes | none | Review benchmark-sensitive code or docs changes for semantic regressions, normalization drift, reproducibility gaps, and provenance overclaim. |

### Campaign Analysis

| Skill | Kind | Phase | Writes | SLURM | Artifacts | Delegates | Use When |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `adversarial-search-campaign` | analysis | analysis | yes | no | yes | `benchmark-row-status`, `artifact-provenance`, `evidence-synthesis` | Analyze adversarial route/search campaigns with canonical commands, expected artifacts, status boundaries, and claim limits. |
| `carla-replay-parity` | analysis | analysis | yes | no | yes | `benchmark-row-status`, `artifact-provenance`, `evidence-synthesis` | Review CARLA replay parity evidence with scenario, replay, metric, and limitation tracking. |
| `hybrid-learning-component-eval` | analysis | analysis | yes | no | yes | `artifact-provenance`, `benchmark-row-status`, `evidence-synthesis` | Evaluate hybrid-learning components with mechanism-level evidence tables, config provenance, and claim boundaries. |
| `oracle-imitation-campaign` | analysis | analysis | yes | no | yes | `artifact-provenance`, `benchmark-row-status`, `evidence-synthesis` | Analyze oracle-imitation campaign outputs with lineage, dataset, checkpoint, metric, and caveat discipline. |
| `predictive-planner-comparison` | analysis | analysis | yes | no | yes | `benchmark-row-status`, `artifact-provenance`, `evidence-synthesis` | Compare predictive planner v2 runs against baseline planners with config, seed, artifact, and failure-mode discipline. |
| `trace-mechanism-review` | analysis | analysis | yes | no | yes | `artifact-provenance`, `evidence-synthesis` | Review exact planner/scenario/seed/episode traces and videos without overgeneralizing from qualitative samples. |

### Context And Docs

| Skill | Kind | Phase | Writes | SLURM | Artifacts | Delegates | Use When |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `context-map` | atomic | context | no | no | no | none | Generate a focused repository context map before multi-file changes; use when you need to identify the relevant files, docs, commands, and risks. |
| `context-note-maintainer` | atomic | context | yes | no | no | none | Create or refresh linked docs/context notes so reusable agent knowledge stays discoverable, current, and easy to hand off. |
| `experiment-context` | atomic | context | no | no | no | none | Find the canonical config-first training or evaluation path, artifact lineage, and validation gates for a concrete experiment task in robot_sf_ll7. |
| `skill-picker` | atomic | context | no | no | no | none | Choose the most appropriate repo-local skill for an ambiguous task by consulting .agents/skills/README.md. |
| `update-docs-on-code-change` | atomic | context | yes | no | no | none | Keep docs aligned with code changes that affect workflows, contracts, or user-facing behavior. |
| `what-context-needed` | atomic | context | no | no | no | none | Ask for the minimum repository context needed to answer or implement a task safely. |

### Domain-Specific Utilities

| Skill | Kind | Phase | Writes | SLURM | Artifacts | Delegates | Use When |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `svg-inspection` | atomic | analysis | no | no | no | none | Inspect and debug SVG maps for parser-facing issues using reusable Robot SF helpers. |

### General

| Skill | Kind | Phase | Writes | SLURM | Artifacts | Delegates | Use When |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `goal-autopilot` | orchestrator | implementation | yes | no | no | `goal-issue-implementation`, `gh-pr-merger`, `goal-pr-review`, `goal-issue-discovery` | Continuous goal autopilot; orchestrates implement, review, merge, and discover cycles with preflight validation and delegation failure recovery. |

### Issue Lifecycle

| Skill | Kind | Phase | Writes | SLURM | Artifacts | Delegates | Use When |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `gh-issue-autopilot` | orchestrator | implementation | yes | no | no | `gh-issue-sequencer`, `implementation-verification`, `pr-ready-check`, `gh-pr-opener`, `artifact-provenance` | Autonomous issue-to-PR workflow from next eligible issue to ready PR with consistent metadata handling. |
| `gh-issue-clarifier` | atomic | context | yes | no | no | none | Clarify ambiguous GitHub issues by tightening scope and acceptance criteria, proposing solution options with pros/cons, and marking decision-required issues when maintainer input is needed. |
| `gh-issue-creator` | atomic | context | yes | no | no | none | Create structured GitHub issues from vague prompts using repo templates, conservative assumptions, and Project #5 metadata. |
| `gh-issue-priority-assessor` | atomic | context | yes | no | no | none | LLM-backed review workflow for Project #5 priority inputs; assess plausibility, propose values with uncertainty, route maintainer-value tradeoffs to issue-audit, and optionally apply explicit opt-in updates. |
| `gh-issue-sequencer` | atomic | context | yes | no | no | none | Maintain a clear next-work queue in GitHub Project #5 by normalizing issue status, priority, and execution order; route genuine priority tradeoffs to issue-audit. |
| `gh-issue-template-auditor` | atomic | context | yes | no | no | none | Review existing GitHub issues against the repo's issue-template contract and repair underspecified issues when the fix is clear. |
| `goal-issue-discovery` | orchestrator | analysis | yes | no | no | `gh-issue-creator`, `gh-issue-sequencer`, `gh-issue-priority-assessor`, `agentic-eval`, `auto-improvement`, `autoresearch`, `context-map` | Use for an autonomous Robot SF issue-discovery loop that finds bounded improvement opportunities and creates evidence-graded GitHub issues; not for implementation. |
| `goal-issue-implementation` | orchestrator | implementation | yes | no | no | `gh-issue-sequencer`, `gh-issue-autopilot`, `implementation-verification`, `pr-ready-check`, `gh-pr-opener`, `gh-issue-creator`, `context-note-maintainer`, `issue-splitter` | Use for an autonomous Robot SF issue-to-PR loop that selects eligible GitHub issues, implements one scoped issue at a time, validates, pushes, and opens PRs. |
| `issue-audit` | atomic | context | yes | no | no | none | User-in-the-loop open-issue audit that asks one readiness-blocking question at a time or one priority-tradeoff question at a time and updates issues as decisions are made. |
| `issue-contract-maintainer` | orchestrator | planning | yes | no | no | `gh-issue-clarifier`, `gh-issue-template-auditor`, `issue-audit`, `issue-splitter` | Maintain GitHub issue contracts through template audits, ambiguity clarification, and user-decision application. |
| `issue-splitter` | atomic | planning | yes | no | no | `gh-issue-creator` | Split a parent, epic, decision, or research issue into the smallest independently implementable child issue with duplicate checks and conservative parent linking. |

### PR Lifecycle

| Skill | Kind | Phase | Writes | SLURM | Artifacts | Delegates | Use When |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `gh-pr-comment-fixer` | atomic | context | yes | no | no | none | Fix GitHub PR review comments with branch-safe edits, validation, and explicit thread resolution. |
| `gh-pr-merger` | atomic | verification | yes | no | no | none | Guarded PR merger; merges merge-ready PRs after verifying label, CI status, branch protection, and preflight checks. |
| `gh-pr-opener` | atomic | context | yes | no | no | none | Open a conservative Robot SF PR with scope verification, freshness checks, and artifact discipline. |
| `goal-pr-review` | orchestrator | verification | yes | no | no | `implementation-verification`, `pr-ready-check`, `gh-pr-comment-fixer`, `review-benchmark-change`, `gh-issue-creator`, `context-note-maintainer` | Use for an autonomous Robot SF PR review loop that fixes scoped review gaps, validates proof, resolves review threads, and applies merge-ready; not for merging. |
| `pr-hindsight-review` | analysis | analysis | no | no | no | none | Review merged PRs after the fact to decide whether autonomous routing produced useful progress, partial coverage, duplicate coverage, or a successor slice. |

### Research Iteration

| Skill | Kind | Phase | Writes | SLURM | Artifacts | Delegates | Use When |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `agent-workflow-capture` | atomic | analysis | yes | no | no | none | Capture private candidate lessons from agent execution into `.git/codex-agent-runs/notes/inbox/` when a repeatable workflow, routing, validation, tooling, or instruction improvement is noticed. |
| `agent-workflow-promotion` | orchestrator | implementation | yes | no | no | `review-and-refactor`, `update-docs-on-code-change`, `context-note-maintainer`, `gh-issue-creator` | Promote accumulated private `.git/codex-agent-runs/notes/inbox/` workflow lessons into small, evidence-backed repository instruction, skill, docs, or tooling changes with validation. |
| `agentic-eval` | atomic | analysis | yes | no | no | none | Evaluate and improve AI workflow outputs with small goldens, rubrics, and repeatable checks; use when tuning skills, prompts, instructions, or agent behavior. |
| `auto-improvement` | atomic | analysis | yes | no | no | none | Focused measurement-aware refinement loop for Robot SF prompts, docs, and small code changes; use when a task benefits from trying a few simple improvements. |
| `autoresearch` | atomic | analysis | yes | no | no | none | Autonomous iterative experimentation loop for measurable Robot SF tasks; use when the user wants an improvement loop with baseline, experiments, and keep/discard decisions. |
| `goal-slurm-experiment` | orchestrator | implementation | yes | yes | no | `experiment-context`, `goal-issue-implementation`, `slurm-campaign-submit`, `artifact-provenance`, `context-note-maintainer` | Keep one skill-owned Robot SF learning or training SLURM job active by selecting the best current experiment candidate, closing implementation gaps through an issue-to-PR workflow, and submitting the validated job from its owning worktree. |

### SLURM And Campaign Submission

| Skill | Kind | Phase | Writes | SLURM | Artifacts | Delegates | Use When |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `auxme-issue791-submit` | atomic | implementation | yes | yes | no | `slurm-campaign-submit` | Submit issue-791-specific Auxme training jobs with explicit config provenance and wrapper-safety checks. |
| `auxme-slurm-reliable-submit` | atomic | implementation | yes | yes | no | `auxme-issue791-submit` | Submit issue-791 style Auxme SLURM jobs with explicit config, live partition pressure checks, and max-time-safe wrapper routing. |
| `slurm-campaign-submit` | atomic | implementation | yes | yes | no | `artifact-provenance` | Submit generic SLURM campaigns with preflight, config provenance, job metadata, artifact expectations, and failure classification. |

### Validation And Cleanup

| Skill | Kind | Phase | Writes | SLURM | Artifacts | Delegates | Use When |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `clean-up` | atomic | verification | yes | no | no | none | Clean up the current branch in the Robot SF repo by following docs/dev_guide.md and reusable scripts/dev commands; use when asked to tidy a branch, run Ruff format/fix, or run parallel pytest before sharing changes. |
| `implementation-verification` | atomic | verification | no | no | no | none | Verify branch changes against origin/main with claim-based evidence, not only test status. |
| `pr-ready-check` | atomic | verification | no | no | no | none | Run the repository PR readiness pipeline using shared scripts/dev entry points (ruff fix/format, parallel tests, coverage, and docstring checks). |
| `quality-playbook` | policy | verification | no | no | no | none | Repo-wide risk-proportional validation workflow for non-trivial changes with context, risk, validation, and follow-through. |
| `review-and-refactor` | atomic | verification | yes | no | no | none | Surgical review-then-refactor workflow for small code or docs changes; use when a task needs inspection before a narrow improvement. |

## Aliases

| Alias | Canonical skill |
| --- | --- |
| `agent-improvement-capture` | `agent-workflow-capture` |
| `agent-improvement-promotion` | `agent-workflow-promotion` |
| `auxme-issue791-reliable-submit` | `auxme-slurm-reliable-submit` |
| `context-unblocker` | `what-context-needed` |
| `continuous-autopilot` | `goal-autopilot` |
| `gh-issue-to-pr` | `gh-issue-autopilot` |
| `guarded-pr-merge` | `gh-pr-merger` |
| `implement-review-merge-discover` | `goal-autopilot` |
| `issue-clarification` | `issue-contract-maintainer` |
| `issue-contract-audit` | `issue-contract-maintainer` |
| `issue-discovery` | `goal-issue-discovery` |
| `issue-queue-runner` | `goal-issue-implementation` |
| `issue-to-pr` | `gh-issue-autopilot` |
| `parent-to-child-issue` | `issue-splitter` |
| `pr-merger` | `gh-pr-merger` |
| `pr-retrospective` | `pr-hindsight-review` |
| `pr-review-runner` | `goal-pr-review` |
| `proof-policy` | `quality-playbook` |
| `quality-strategy` | `quality-playbook` |

## Notes

- Prefer the most specific skill that matches the task.
- Combine skills only when they cover different phases.
- This README is generated from `.agents/skills/skills.yaml`; do not hand-edit the index.
