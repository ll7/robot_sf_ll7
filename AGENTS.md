# Repository Guidelines

This file is the compact boot contract and the entry point into repository instructions. It owns
the Instruction Precedence contract below; other surfaces link to it instead of restating their own
authority order. Topic owners for long-form workflow guidance are indexed at
`docs/dev/agents/relocated-agents-guidance.md`.
Prefer reusable shell entry points under `scripts/dev/`. Keep agent prompts and handoff notes
token-efficient; clarity wins on human-facing surfaces (expand terms or link `docs/glossary.md`).

## Instruction Precedence

<!-- instruction-precedence:start -->
Repository-internal sources, highest first:

1. **Repository invariants** — safety, evidence integrity, honest validation reporting, and
   recoverability of local work; benchmark, metric, schema, model-provenance, and paper-facing
   contracts are invariant-class.
2. **Current maintainer direction** in the active issue, PR, or thread; it overrides stale workflow
   prose but cannot waive an invariant or authorize an unproven claim.
3. **Nearest scoped guidance** (`SLURM/AGENTS.md`, task-owned procedure docs); it specializes root
   guidance and cannot silently weaken root invariants.
4. **`AGENTS.md`** (this file): boot router, invariant list, and owner of this precedence contract.
5. **Task-owned procedures and skills** selected through `docs/ai/agent_workflow_entrypoints.md`.
6. **`docs/maintainer_values.md`**: rationale and tie-breakers, not a second rulebook.
7. **`docs/dev_guide.md`, context notes, historical reports, and provider adapters**: detail and
   provider mechanics.

Resolution rules:

- Platform and system instructions outrank repository documents. An applicable user task request
  sets task intent within those higher-level constraints; it cannot waive a repository invariant or
  authorize an unproven claim. This contract governs repository-owned documents only.
- A task plan, issue comment, or historical record never outranks current code and evidence; stale
  plans are re-validated before use, and scoped guidance cannot weaken a root invariant.
- Provider adapters configure tools and link to canonical policy; they never create repository
  policy, even under a renamed heading.
- `.specify/memory/constitution.md` states WHAT the platform delivers and its stable contracts; it
  does not define a competing runtime precedence.
<!-- instruction-precedence:end -->

When this contract resolves a recurring conflict, update the active issue or PR, patch the stale
instruction, or open a bounded follow-up issue.

## Task-Scoped Context Entrypoints

The machine-readable profile and route mapping is `.agents/task_scope_manifest.yaml`; the human route
table is owned by `docs/ai/agent_workflow_entrypoints.md`. Select one profile from the changed
surfaces and risk, load only its required context, and escalate when inspection reveals more risk:
Observe (inspect, explain, triage; no environment, branch, plan, worktree, or PR ceremony), Local
(bounded edit; scoped guidance plus targeted validation), Coordinated (multi-module, API, migration,
or ambiguous; plan, isolated worktree when collision risk exists, integration validation),
Evidence-critical (benchmark, research, release, security, publication; full evidence, custody,
reproducibility, exact-head, and release gates).

Always-required core context:
- `docs/maintainer_values.md`: maintainer principles and tie-breakers.
- `AGENTS.md`: invariants, precedence, and this router.
- `docs/ai/agent_workflow_entrypoints.md`: route table, command entrypoints, handoff format, and large-file navigation.

The route table is the single owner of task-to-guidance routing; references are required by default
and optional only when marked optional/illustrative, generated, or explicitly background. For the
token-efficient thread profile, phase audits, meta-workflow gate, SLURM lanes, knowledge graph,
cross-agent compatibility, and context-note policy, use the topic index at
`docs/dev/agents/relocated-agents-guidance.md`.

## Local Machine Context

If `local.machine.md` or `local.machine.<name>.md` exists at the repository root, read it before
expensive commands and follow its concurrency, location, and hardware limits. Never store secrets
there.

## Mutation And Delivery Triggers

| Situation | Required response |
| --- | --- |
| No mutation and no durable artifact (inspect, explain, triage, review) | No branch, worktree, PR, or landing procedure; report findings only |
| Bounded mutation without collision or custody risk | Work in the active task checkout according to caller/tool ownership; run targeted validation; report the diff |
| Concurrent, multi-step, or artifact-bearing mutation | Use an isolated linked worktree and follow `docs/dev/worktree_lifecycle.md` |
| Requested delivery, review, or merge | Follow `docs/code_review.md` and `docs/dev_guide.md` for PR metadata, exact-head review, readiness, and landing |

Every durable implementation ends with a delivered PR or commit, a requested direct change, or an
explicit handoff. Destructive Git operations, losing dirty worktrees, publishing ignored-but-durable
artifacts, and reporting validation that was not performed remain forbidden regardless of profile.
Worktree creation, bootstrap, branch synchronization, teardown, artifact custody, and stash safety
are owned by `docs/dev/worktree_lifecycle.md`.

## Validation And Evidence

Match proof to risk: docs and instruction changes use the cheap path (inspect the diff, verify
links); runtime changes need focused tests plus lint and format; benchmark and planner changes need
benchmark, policy-analysis, or equivalent executable evidence; metric and schema changes need
targeted assertions with a reproducible sample; paper-facing claims need reproducible evidence at
the claim boundary. Claim strength overrides the nominal class. Fallback or degraded benchmark
execution is never success evidence. If proof fails or cannot be gathered, close as `blocked`,
`diagnostic`, or `not benchmark evidence` and record the next smallest step. The full validation
matrix is owned by `docs/code_review.md`; benchmark governance by `docs/benchmark_governance.md`.

## Delivery And Communication

Use conventional commits. A PR states intent, linked issues, validation commands, artifact
disposition, and downstream propagation; because merges squash, reconcile the final title and body
with `uv run python scripts/dev/gh_pr_body_rest.py` and pass Markdown-heavy comments through
`scripts/dev/gh_comment.sh` or a body file. Prefer concise-but-explanatory reporting: what changed,
why it matters, remaining risk, and uncertainty separated from observed evidence. When a persistent
plan is required (Coordinated and Evidence-critical profiles), follow `.agents/PLANS.md`; delivery
and context skills are indexed in `.agents/skills/README.md`.

## Friction And Follow-Up

Findings observed while working get a proportional response:

| Finding | Response |
| --- | --- |
| Threatens correctness, security, evidence integrity, or blocks the task | Address it, or explicitly block or escalate before delivery |
| Small, clearly in scope, and cheaper to fix than to work around | Fix inline |
| Repeated and materially valuable | Search for an existing issue first; track once only if none exists |
| Incidental, speculative, cosmetic, or unrelated | Do not widen the task; mention it briefly in the handoff if useful |

Examples: a missing validation script that blocks a gate is addressed or escalated; a repeated
manual release step with demonstrated value is tracked once and may become a checked-in helper; a
harmless one-off inconvenience and an unrelated code smell get no action; stale documentation found
outside the task gets a handoff note. Automation is justified by demonstrated repetition, error
reduction, custody or reproducibility value, or clear net maintenance benefit. Friction issues use
the existing `friction:` prefix and `technical-debt` or `documentation` labels.

## Donts

- Never change code in `.venv`; manage dependencies through `pyproject.toml` and `uv sync`.
- Never present fallback/degraded benchmark execution as success evidence.
- Never rely on worktree-local `output/` as durable artifact storage.
- Never remove dirty worktrees or ignored-but-important outputs without preservation or a clear handoff.
- Never pass multiline Markdown-heavy GitHub comments through inline shell strings.
