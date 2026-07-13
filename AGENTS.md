# Repository Guidelines

Use `docs/maintainer_values.md` as the highest-level current maintainer guidance, then `AGENTS.md`,
`.specify/memory/constitution.md`, and `docs/dev_guide.md` for repository execution details.
This file is the compact boot contract. Long-form situational guidance moved to
`docs/dev/agents/relocated-agents-guidance.md`; read the linked section before that workflow.
Prefer reusable shell entry points under `scripts/dev/` for automation and AI skills.
Use `.vscode/tasks.json` as thin wrappers around those scripts.
Keep agent prompts, internal instructions, and handoff notes token-efficient while preserving meaning.
For human-facing surfaces (README, `docs/`, feature names, `CHANGELOG.md`, public docstrings, PR/issue titles),
clarity wins: define acronyms/project terms on first use or link `docs/glossary.md`, and lead with a plain-language summary.

## Maintainer Value Hierarchy

`docs/maintainer_values.md` is the compact source of truth for current maintainer values. Optimize
first for concrete research progress on social-navigation simulation, benchmarking, and planner
exploration. The hard rule is to be honest, transparent, and reproducible. Apply proof,
documentation, and process in proportion to risk:

- benchmark, metric, schema, model-provenance, and paper-facing claims still require strong,
  reproducible evidence before they are treated as established;
- exploratory planner or research work may move faster when its status is clearly labeled as
  exploratory, diagnostic, blocked, or not yet benchmark evidence;
- low-risk docs, metadata, and instruction changes use the cheaper validation path by default:
  inspect the diff and verify changed links or referenced paths where practical;
- substantive claims, recommendations, benchmark conclusions, and prioritization judgments below
  roughly 95 percent confidence should include a numeric uncertainty estimate, caveat, or condition
  that would change the conclusion;
- current maintainer direction overrides stale workflow prose. When an instruction appears to
  conflict with the user's current priority, follow the current priority, call out the conflict, and
  propose or make the smallest doc update needed to remove the drift.

When instruction surfaces conflict, use this precedence order:

1. Current maintainer direction in the active issue, PR, or thread.
2. `docs/maintainer_values.md`.
3. `AGENTS.md`.
4. `.agents/README.md`, `.agents/PLANS.md`, and repo-local skill docs under `.agents/skills/`.
5. `docs/dev_guide.md`, context notes, and tool-specific compatibility pointers.

If this order resolves a recurring workflow conflict, make the conflict visible where practical:
update the active issue/PR, patch the stale instruction, or open a bounded follow-up issue. Routine
workflow cleanup should proceed autonomously when the scope is bounded; label assumptions,
uncertainty, and evidence grade instead of pausing for confirmation. Treat Project #5 ordering and
scores as advisory when they conflict with fresh maintainer direction or newly observed evidence;
record the override and update Project metadata later when quota and API limits allow.

Use validation proportional to the file/change type, with claim strength as an escalation override:

| Change class | Minimum proof | Full `pr_ready_check` required when |
| --- | --- | --- |
| Docs-only or instruction-only | Inspect diff; verify changed links or paths where practical; run available lightweight markdown, index, or sync checks. On human-facing surfaces, also run a clarity pass: expand acronyms/project terms or link `docs/glossary.md`, and lead with a plain-language summary. | The text changes generated indexes, compatibility surfaces, or makes evidence-sensitive claims. |
| Workflow/tooling docs or skills | Cheap docs proof plus relevant skill/schema/sync checks such as `uv run python scripts/dev/check_skills.py --preflight <skill>` or `uv run python scripts/tools/sync_ai_config.py --check`. | Scripts, schemas, generated indexes, routing behavior, or automation behavior changes. |
| Runtime code | Focused tests for changed behavior plus lint/format gates. | The change is user-facing, cross-module, release-facing, or affects shared execution paths. |
| Benchmark, metric, schema, model-provenance | Executable proof on the intended contract with provenance and fallback/degraded exclusions. | Almost always; skip only for explicit diagnostic-only docs with no semantic change. |
| Paper-facing or public claims | Reproducible evidence matching the claim boundary, caveats, uncertainty, and artifact provenance. | Always before treating the claim as established. |

Claim strength overrides the nominal row: a docs or workflow edit that asserts a benchmark, metric,
schema, model-provenance, or paper-facing result must use the stronger proof tier for that claim.

## Always-Load Context

Read only the surfaces relevant to the task. Prefer repo-local files over ad-hoc summaries in issue comments.

- `docs/maintainer_values.md`: compact current values and hard contracts.
- `AGENTS.md`: top-level execution rules, repo structure, and workflow defaults.
- `memory/MEMORY.md`: concise repo-local memory index for stable facts and links to topic files.
- `docs/code_review.md`: benchmark-facing review criteria, provenance checks, and regression traps.
- `docs/context/INDEX.md`: retrieval-first catalog for current context-note entry points.
- `.agents/PLANS.md`: plan-writing convention for non-trivial work.
- `.agents/skills/`: canonical skill tree for execution workflows and context packs.
- `.agents/prompts/`, `.agents/commands/`, `.agents/agents/`: canonical agent workflow sources.
- `docs/ai/`: AI-facing overview documents for structure, planner-zoo state, and context packing.
- `.understand-anything/knowledge-graph.json`: shared codebase graph; see `docs/ai/understand_anything.md` before reading or updating it.

For the token-efficient active thread profile, phase audits, meta-workflow PR gate, SLURM lane rules,
shared knowledge graph, cross-agent compatibility, and detailed context-note policy, read
`docs/dev/agents/relocated-agents-guidance.md`.

## Local Machine Context

If `local.machine.md` or `local.machine.<name>.md` exists at the repository root, read it before
running expensive commands. Follow local limits for concurrency, execution location, GPU/SLURM
requirements, and machine-specific constraints. If no local context exists, use conservative
repository-safe commands. Never store secrets in local machine context files.

## Fresh Worktree Bootstrap

When working in a linked Git worktree, detect bootstrap state before running expensive commands.
Honor explicit user/native-tool worktree locations first. Otherwise prefer a sibling container such
as `../robot_sf_ll7.worktrees/<branch-or-issue-slug>`; use issue number plus a short slug when possible.
Treat a checkout as linked when `.git` is a file pointing into `.git/worktrees/...`, or when
`git rev-parse --git-common-dir` differs from `git rev-parse --git-dir`.

A linked worktree is fresh only when it lacks both `local.machine.md` and `.venv` plus any
team-specific initialized marker. In a fresh worktree, run
`scripts/dev/bootstrap_worktree.sh` (preferred) or manually: derive the main checkout from the
common Git dir, symlink the main `local.machine.md` when present, then run
`uv venv .venv && uv sync --all-extras`. The `uv venv .venv` step is required: `uv sync --all-extras`
alone may silently detect and reuse the main checkout's `.venv` without creating one locally,
leaving `.venv/bin/activate` missing. After sync, verify `.venv/bin/python` exists before
continuing; if absent, the environment is not usable and the caller must fail closed. Then
`source .venv/bin/activate` before Python tooling. For quick targeted validation that intentionally
reuses the main checkout virtualenv, prefer `scripts/dev/run_worktree_shared_venv.sh -- <uv-run-command>`
so `PYTHONPATH` points at the active worktree. Use a full local `.venv` and final PR readiness for
merge proof. Do not include CARLA in routine `--all-extras`; opt into `--group carla` only for
CARLA-capable worktrees and prove runtime with `scripts/dev/check_carla_runtime.sh` when needed.

If the current branch is not `main`, fetch latest `origin/main` and merge it early:
`git fetch origin main && git merge origin/main`. Do not create divergent per-worktree machine
contexts unless the worktree truly needs machine-specific behavior.

## Worktree Teardown And Artifacts

After PR review, issue implementation, publishing, or abandoned exploration, either remove no-longer-needed
worktrees safely or record why they are preserved. Before removal, enumerate worktrees and inspect
status. Preserve relevant tracked, untracked, and ignored-but-important local change by committing,
stashing, saving a patch, promoting a durable artifact, or recording an explicit handoff. Do not
remove dirty worktrees or worktrees with unpushed commits unless the preservation record says what
was kept or why nothing needed preservation. Inspect large ignored directories such as `output/`
before removal; classify them as disposable, ignored cache, tracked manifest/evidence,
durable-required, or handoff-needed.

Treat `output/` as temporary, worktree-local, and generally untracked. Do not rely on it for durable
dependencies unless the launcher hydrates from a canonical source. Promote artifacts required by
future runs, benchmarks, reports, or release workflows to durable storage and track lightweight
manifests or registry entries. Keep raw episode JSONL, videos, large SLURM logs, coverage HTML,
model caches, and checkpoints out of git unless there is a narrow fixture reason. Do not use Git LFS
as the default answer for generated benchmark artifacts.

For compact worktree hygiene and the full teardown procedure, read `docs/dev/agents/relocated-agents-guidance.md`.

## Project Structure And Ownership

Core simulation code lives in `robot_sf/` with `gym_env`, `sim`, `nav`, and `render` subpackages.
Training and evaluation entry points sit in `scripts/`; demos and notebooks live under `examples/`.
Tests are split between `tests/`, `tests/pygame/`, and `fast-pysf/`. Assets and checkpoints are
versioned under `maps/svg_maps/` and `model/`.

Before creating a new module, script, config family, or asset registration, find the canonical owner
and extend it instead of duplicating it. Search first with `rg`, read relevant matches, and reuse
shared primitives in `robot_sf/benchmark/`, `robot_sf/research/`, `scripts/tools/`, or
`scripts/validation/` when they already own the concern. Per-issue scripts are for genuinely new
orchestration, not re-deriving existing capability. If a new owner is unavoidable, state what it
supersedes in the PR.

## Build, Test, And Style

Set up dependencies with `uv sync --all-extras` and hooks with `uv run pre-commit install`. Format
and lint with `uv run ruff check .` followed by `uv run ruff format .`. Run the main suite with
`uv run pytest tests`; add `-m "not slow"` for long benches. Headless GUI checks use
`DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy uv run pytest tests/pygame`. Validate SocialForce
with `uv run python -m pytest fast-pysf/tests -v`.

Prefer shared development entry points:

- `scripts/dev/ruff_fix_format.sh`
- `uv run python scripts/dev/run_compact_validation.py -- <command>`
- `scripts/dev/run_tests_parallel.sh`
- `BASE_REF=origin/main scripts/dev/pr_ready_check.sh`
- `scripts/dev/gh_comment.sh` for Markdown-heavy GitHub comments

Use config-first workflows for reproducibility: stable experiments belong under `configs/` with a
canonical `--config <path>` command. CLI flags are for short-lived local overrides. Ruff enforces
4-space indent, 100-character lines, and double-quoted strings. Prefer type-annotated interfaces,
factory functions as public entry points, `snake_case` modules, `PascalCase` classes/dataclasses,
`test_<feature>.py` tests, fixtures in `conftest.py`, and structured logging instead of ad-hoc prints.

## Research And Benchmark Discipline

Research progress is primary, but claim strength controls proof. Benchmark-facing, metric-facing,
schema-facing, skill, and test changes require concrete evidence appropriate to risk. New local
planners must be proven with an actual benchmark or targeted execution path in this repository.
Metric changes need proof that values compute as intended or fix the regression. New skills must be
checked against their real invocation path and referenced files. New tests should fail for the right
reason before the fix when practical, or include direct evidence for the intended contract.

Fallback/degraded benchmark execution is not success evidence unless the task explicitly measures
that mode. If a planner, environment, or dependency cannot satisfy an accurate benchmark contract,
fail closed with a clear error and `not available` or `failed` status. If intended proof fails or
cannot be gathered, close as `blocked`, `diagnostic`, or `not benchmark evidence`; record the next
smallest proof step. Benchmark reports and issue follow-ups must identify `native`, `adapter`,
`fallback`, or `degraded` mode and treat fallback/degraded as caveats.

Match proof to risk: benchmark/planner changes need benchmark, policy-analysis, or equivalent
executable evidence; training/config workflow changes need canonical command or smoke path;
metrics/schema changes need targeted assertions and a reproducible sample; docs/skill/instruction
changes need referenced path, command, and discoverability checks.

## Commit And Pull Request Workflow

Use conventional commit style from history. Each PR should summarize intent, reference related
issues, and list validation commands. Include screenshots or GIFs when UI playback changes, and note
new assets under `maps/` or `model/`. Before opening a PR, fetch latest `origin/main`, merge or
rebase into the feature branch, and validate against the fresh base. Docs-only low-risk instruction
branches may use the cheaper official path: inspect diff, verify referenced paths where practical,
and state when full readiness was not run.

Before PR creation, inspect newly created `output/*` with a compact count-first view and classify
artifact handling. Fill downstream propagation for evidence-producing PRs; for low-risk or
not-applicable changes, state why. Research, benchmark, metric, paper-facing, or other evidence-producing
PRs must state target claim/hypothesis/blocker, comparator/baseline, evidence tier, result
classification, decision/stop rule, and synthesis/update target. Support/tooling/docs-only PRs with
no research claim may use `NA` and explain why.

Use repository-root-relative paths in PRs, issue comments, docs, and agent responses. For GitHub
comments with Markdown-heavy bodies, do not pass body inline through the shell; use
`scripts/dev/gh_comment.sh`, `gh issue/pr comment --body-file`, or REST JSON input.

For batching, GraphQL quota, Project #5 metadata, Spark sidecar routing, and autonomous stop guard
details, read `docs/dev/agents/relocated-agents-guidance.md`.

## Planning And Communication

For non-trivial work, follow `.agents/PLANS.md`: restate boundaries, list evidence sources before
implementation, keep validation commands explicit, and say what proof will demonstrate. Prefer
concise-but-explanatory responses over terse status-only updates. For benchmark or planner findings,
include what changed, why it matters, and what risk or limitation remains. Separate observed
evidence from hypothesis when uncertainty remains.

## Key Codex Skills

Use repo-local skills under `.agents/skills/` when task primarily involves issue delivery, PR review,
benchmark/planner context, validation, or GitHub workflow automation. Start with `.agents/skills/README.md`
when choosing among them. Common delivery skills include `goal-issue-implementation`,
`gh-issue-autopilot`, `implementation-verification`, `pr-ready-check`, `gh-pr-opener`,
`goal-pr-review`, and `gh-pr-merger`.

## Friction And Automation

Standing policy (2026-07-13): any friction observed during work — a flaky command, a
misleading readout, a manual step that should be automatic, or a repeated failure class — is
either fixed inline when the change is small, or tracked durably as an issue before moving on.
Never just navigate around it: un-tracked friction gets re-paid on every future encounter, and
self-improvement is part of this system. Track friction with the existing convention: an issue
titled with a `friction:` prefix and the `technical-debt` label (or `documentation` for a docs
gap), naming the file/command and the concrete suggested change (see #5468, #5475). Do not
invent new labels or priorities.

Standing policy (2026-07-13): anything that can be improved or automated without an LLM must
become a reusable script or automation — never a repeated manual step or an LLM prompt. If a
deterministic sequence runs twice, promote it to a checked-in helper under `scripts/dev/`
rather than re-typing or re-prompting it.

## Donts

- Never change code in `.venv`; manage dependencies through `pyproject.toml` and `uv sync`.
- Never present fallback/degraded benchmark execution as success evidence.
- Never rely on worktree-local `output/` as durable artifact storage.
- Never remove dirty worktrees or ignored-but-important outputs without preservation or a clear handoff.
- Never pass multiline Markdown-heavy GitHub comments through inline shell strings.
