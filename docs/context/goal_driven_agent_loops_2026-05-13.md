# Goal-Driven Agent Loop Skills

**Date**: 2026-05-13
**Status**: Adopted as repo-local skill-first workflow guidance

## Purpose

This note records the shared contract for four goal-driven agent loop skills:

- `.agents/skills/goal-issue-discovery/SKILL.md`
- `.agents/skills/issue-audit/SKILL.md`
- `.agents/skills/goal-issue-implementation/SKILL.md`
- `.agents/skills/goal-pr-review/SKILL.md`

The loops are intentionally skill-first. No slash-command wrappers, prompt wrappers, or formal
`docs/superpowers/specs/` design document are part of the initial implementation.

## Shared Contract

Each autonomous goal loop starts with a short preflight that states the active loop, scope, write
mode, stop condition, and exclusions. After that preflight, autonomous GitHub writes are allowed by
default. This includes issue creation and edits, issue consolidation or splitting, Project #5
updates, branch/PR work, follow-up issue creation, and `merge-ready` labeling when the written
criteria pass.

The default stop condition is queue exhaustion. A user-provided time budget is also valid; when a
time budget stops the loop, the agent must leave a concise handoff that names the last completed
item, current blocker, changed GitHub objects, and next recommended action.

Goal-loop skills are orchestration layers, not replacements for narrower workflow skills. They
should delegate to existing repo-local skills for concrete work:

- issue creation: `gh-issue-creator`
- issue clarification and template repair: `gh-issue-clarifier`, `gh-issue-template-auditor`
- queue routing: `gh-issue-sequencer`, `gh-issue-priority-assessor`
- issue implementation: `gh-issue-autopilot`, `gh-pr-opener`
- PR repair during review: `gh-pr-comment-fixer`, `gh-issue-creator`
- verification: `implementation-verification`, `pr-ready-check`, `review-benchmark-change`
- durable notes: `context-note-maintainer`

Spark subagents may be used only for bounded side tasks such as locating files, summarizing a small
area, inspecting command output, or performing narrow mechanical edits with clear file ownership.
The main agent remains responsible for planning, integration, validation, GitHub writes, and final
judgment. For the issue implementation phase (e.g., under goal-issue-implementation), the main agent
may spawn a full sub-agent inside a fresh linked worktree to execute the end-to-end implementation
and validation tasks, while the parent remains responsible for selection, queue audits, artifact
verification, PR opening, and teardown.

## Research-Result Mode

When a user asks the loops to prioritize research results or research directions, issue selection
should favor work that changes what the repository can honestly claim. Good candidates close or
revise a hypothesis, move a claim boundary, record a useful negative result, synthesize accumulated
diagnostics, or unblock a durable experiment. Support, cleanup, and simulator-speed work can still
win when they remove a concrete blocker, but they should not crowd out research-result issues by
default.

Research-result issues and PRs must state the compact evidence contract before treating the work as
complete:

- target hypothesis, claim, blocker, or research question;
- baseline or comparator;
- evidence tier and claim boundary;
- decision or stop rule;
- compact durable evidence and provenance plan;
- result classification, such as positive, negative, diagnostic-only, blocked, or inconclusive;
- parent issue, dashboard, registry, claim map, context note, or synthesis surface to update.

Track hypotheses near the experiment by default. A config, launch packet, issue comment, private
ops ledger note, or issue-specific context note is enough while the work is still exploratory or
the next decision is simply what to run next. Record the lightweight fields that make the run
auditable: hypothesis, variant or config, expected signal, result classification, artifact pointer
or snapshot, and next decision.

Create a central hypothesis ledger only when a research family needs cross-run belief management:
many related runs, confusing or contradictory outcomes, repeated negative results, duplicate
variant risk, claim-boundary movement, dissertation or paper synthesis, or the question has shifted
from "what should we run next?" to "what do we believe now?". Do not make a central ledger a
required gate for every exploratory run. Local or private hypotheses are acceptable until they
affect public decisions, claims, or follow-up work.

Use this compact template when drafting the issue body, PR body, or synthesis note:

```md
- Target: <hypothesis, claim, blocker, or research question>
- Baseline/Comparator: <baseline or comparator>
- Evidence Tier And Claim Boundary: <tier and boundary>
- Decision/Stop Rule: <decision or stop rule>
- Evidence And Provenance Plan: <compact durable evidence and provenance plan>
- Result Classification: <positive, negative, diagnostic-only, blocked, or inconclusive>
- Update Surface: <parent issue, dashboard, registry, claim map, context note, or synthesis surface>
```

Use this compact per-experiment note shape when a full issue or synthesis note would be overhead:

```md
- Hypothesis:
- Variant/Config:
- Expected Signal:
- Result Classification:
- Artifact Pointer Or Snapshot:
- Next Decision:
```

After several diagnostic children accumulate under one research parent, prefer a synthesis pass
before adding another exploratory child. The synthesis should name what was learned, what stayed
inconclusive, which lanes are redundant or negative, and the next smallest experiment or follow-up
that would change the conclusion. This is guidance for research-mode routing, not a universal gate
on low-risk maintenance.

## Loop Boundaries

`goal-issue-discovery` may create broad proposal issues, not only observed bug reports. Every new
issue must carry an evidence grade: `observed`, `inferred`, or `proposal`. The grade prevents
speculative ideas from being presented as measured defects or benchmark regressions.

`issue-audit` is not a default `/goal` loop. It is a user-in-the-loop refinement workflow that
orders open issues by readiness blockers and asks exactly one user-facing question at a time. It
updates issue bodies, labels, comments, Project #5 metadata, and follow-up issues as decisions are
made.

`goal-issue-implementation` processes issues sequentially. It selects one eligible issue, creates
one branch, implements, validates, checks generated artifacts, commits, pushes, and opens one PR
before continuing. It stops instead of guessing when an issue is blocked, ambiguous, already covered
by an open PR, or missing proof requirements.

`goal-pr-review` reviews open PRs against linked issue contracts, not just CI state. When the full
proof bar fails, the loop should fix safe actionable gaps on writable PR branches before leaving a
passive "not merge-ready" comment. Each gap should be classified as fixable now, deferred
follow-up, or handoff-only blocker. Fixes must stay inside the issue/PR contract, be committed and
pushed, and be validated before reassessing readiness. It applies a dedicated `merge-ready` label
only after the full proof bar passes: issue contract resolved, diff matches scope, checks or
readiness proof are adequate, review threads are handled, generated artifacts are classified, and
deferred work is captured as follow-up issues.

## Deferred Scope

The initial implementation deliberately avoids:

- slash-command or prompt-wrapper aliases,
- a central router skill that guesses which loop to run,
- persistent daemon-style execution,
- parallel issue implementation as the default,
- a new database, retrieval layer, or queue service.

Those can be reconsidered after the skill-first loops prove useful and the failure modes are clear.

## Validation Surface

Changes to these skills should be validated with:

```bash
uv run python scripts/dev/check_skills.py
uv run python scripts/tools/sync_ai_config.py --check
```

For docs/skill-only changes, also check discoverability by grepping for the skill names in
`.agents/skills/README.md`, `docs/context/README.md`, and any touched AI entry points. Before PR
handoff, run the repository readiness gate unless there is a documented reason to use a narrower
proof:

```bash
BASE_REF=origin/main scripts/dev/pr_ready_check.sh
```

## Related

- `docs/context/issue_713_batch_first_issue_workflow.md`
- `docs/context/issue_728_coding_agents_compatibility.md`
- `docs/ai/awesome_copilot_adaptation.md`
- `.agents/skills/README.md`
