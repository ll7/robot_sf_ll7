# Repository Guidelines

Use `docs/maintainer_values.md` as the highest-level current maintainer guidance, then `AGENTS.md`,
`.specify/memory/constitution.md`, and `docs/dev_guide.md` for repository execution details.
This document covers briefly the repository structure, coding style, testing workflow, and contributor conventions.
Prefer reusable shell entry points under `scripts/dev/` for automation and AI skills.
Use `.vscode/tasks.json` as thin wrappers around those scripts.
Keep agent prompts, internal instructions, and handoff notes as streamlined and token-efficient as possible while preserving the same meaning and constraints.
This token-efficiency goal applies to agent-internal surfaces only. For human-facing surfaces (README, `docs/`, feature names, `CHANGELOG.md`, public docstrings, and PR/issue titles), prioritize clarity per the `## Clarity` rule in [`maintainer_values.md`](docs/maintainer_values.md#clarity): define every acronym or project term on first use or link it to [`glossary.md`](docs/glossary.md), and lead with a plain-language summary before dense terminology. When the two goals conflict on a human-facing surface, clarity wins.

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
| Docs-only or instruction-only | Inspect diff; verify changed links or paths where practical; run available lightweight markdown, index, or sync checks. On human-facing surfaces, also run a clarity pass: expand acronyms/project terms or link [`glossary.md`](docs/glossary.md), and lead with a plain-language summary (see the `## Clarity` rule in [`maintainer_values.md`](docs/maintainer_values.md#clarity)). | The text changes generated indexes, compatibility surfaces, or makes evidence-sensitive claims. |
| Workflow/tooling docs or skills | Cheap docs proof plus relevant skill/schema/sync checks such as `uv run python scripts/dev/check_skills.py --preflight <skill>` or `uv run python scripts/tools/sync_ai_config.py --check`. | Scripts, schemas, generated indexes, routing behavior, or automation behavior changes. |
| Runtime code | Focused tests for changed behavior plus lint/format gates. | The change is user-facing, cross-module, release-facing, or affects shared execution paths. |
| Benchmark, metric, schema, model-provenance | Executable proof on the intended contract with provenance and fallback/degraded exclusions. | Almost always; skip only for explicit diagnostic-only docs with no semantic change. |
| Paper-facing or public claims | Reproducible evidence matching the claim boundary, caveats, uncertainty, and artifact provenance. | Always before treating the claim as established. |

Claim strength overrides the nominal row: a docs or workflow edit that asserts a benchmark, metric,
schema, model-provenance, or paper-facing result must use the stronger proof tier for that claim.

## Agent Context Stack
Treat the following files as the repository-native context stack for Agent-style agents:

- `docs/maintainer_values.md`: compact current values and hard contracts.
- `AGENTS.md`: top-level execution rules, repo structure, and workflow defaults.
- `memory/MEMORY.md`: concise repo-local memory index for stable cross-session facts and links to
  detailed topic files under `memory/`.
- `docs/code_review.md`: benchmark-facing review criteria, provenance checks, and regression traps.
- `docs/context/INDEX.md`: retrieval-first catalog for current context-note entry points, status
  rules, optional context tools, and curated context-pack scopes.
- `.agents/PLANS.md`: plan-writing convention for non-trivial work so intent, scope, and validation stay explicit.
- `.agents/skills/`: canonical skill tree for execution workflows and repo-local context packs,
  mirrored at `.codex/skills/` and `.opencode/skills/` for compatibility.
- `.agents/prompts/`, `.agents/commands/`, and `.agents/agents/`: canonical prompt, command,
  and GitHub agent sources, mirrored into tool-specific compatibility paths when possible.
- `docs/ai/`: AI-facing overview documents for repo structure, planner-zoo state, context packing, and deferred retrieval decisions.
- `.understand-anything/knowledge-graph.json`: shared codebase knowledge graph for interactive
  navigation through the Understand-Anything dashboard/chat tools; see
  `docs/ai/understand_anything.md` before reading the raw large JSON, regenerating it, or updating
  it.

Read only the surfaces relevant to the task. Prefer these repo-local files over ad-hoc summaries in issue comments, and avoid loading broad context-note indexes unless the task actually needs them.

## Token-Efficient Active Thread Profile

For long autonomous goals, delegated batches, or explicitly token-saving work, start or resume with
the copyable profile in `docs/templates/token_efficient_thread_profile.md`. The header records the
goal, `task_class`, `validation_tier`, scope, out-of-scope work, worktree, context budget,
delegation artifacts, output budget, stop guard, validation plan, and handoff target.

Use `task_class` to name the dominant risk surface and `validation_tier` to point back to the
proportional readiness matrix above; do not redefine evidence rules inside the active thread. Use
compact snapshots and validation wrappers before broad reads, and keep raw logs in common Git-dir
agent-run artifacts with only artifact paths and short excerpts in the parent thread. Delegated
worker outputs are route evidence only: parent Codex still reviews the diff, checks the artifact
bundle, and runs the selected validation before accepting, rerouting, or rejecting the work.
At each autonomous phase boundary, run the profile's `Phase Audit` checklist before opening another
batch: reuse recorded usage and route-cache facts, check issue/PR freshness before editing,
prefer filtered worktree/status snapshots, bound CI polling output, and hand off instead of starting
new work when usage is close to the stop guard. The audit must also identify the largest
parent-thread outputs since the previous phase, including broad searches, full skill or doc rereads,
raw validation logs, verbose delegate notifications, failed command families, repeated monitor
noise, and unclear instructions that caused retries. Convert repeated leaks into a compact ledger
field, route-cache entry, worker prompt constraint, or docs patch before the next batch.
For Slurm-backed phases, also record the private-queue freshness check, the submit-host worktree
proof, and any model-route quota failures so resumes do not rediscover the same blockers.

When a long thread resumes after compaction, interruption, or a new user message, run a one-screen
current-request guard before continuing the previous ledger action. Confirm the newest user request,
active goal or PR/issue, current worktree, active delegates, dirty worktrees, and next remote-visible
mutation. If the newest request changes the objective, park the previous batch with a compact handoff
instead of silently continuing stale goal work. Close or explicitly preserve no-longer-needed
subagents, and record the cleanup or preservation decision in the common Git-dir ledger.

For meta-workflow requests such as "review this thread", "improve instructions", or "make this
workflow more token efficient", treat the request as a new bounded docs-or-workflow task. Do not keep
executing the prior goal ledger unless the user explicitly asks to resume it. Start from a fresh
worktree on `origin/main`, summarize the parked batch by worktree, PR/issue, dirty status, active
delegates, and next safe command, then scope the PR to reusable instruction changes. Review the
last thread only for decision-level evidence: repeated broad reads, command failures, unclear
instruction loops, stale-state drift, delegate lifecycle leaks, and missed route choices.

When the user elevates SLURM work as a current priority, make it a first-class lane in the next
phase audit instead of treating it as optional background work. Use `goal-slurm-experiment` for the
single prioritized training lane and `slurm-campaign-submit` for explicit capacity-aware batches.
Before any submission, read the private submit guidance from `~/git/robot_sf_ll7-private-ops` on the
submit path, prove the owning worktree exists and is clean on the submit host, refresh live queue
state, and record duplicate checks plus immediate health-check expectations. If those checks are
not available, classify the candidate as blocked or analysis-only rather than burning Codex turns on
speculative job setup.

## Shared Knowledge Graph

This repository tracks an Understand-Anything graph under `.understand-anything/` as a shared
orientation artifact for agents and contributors. Use it to explore architecture, layers, files,
functions, classes, imports, calls, inheritance, and broad test/config relationships before doing
large discovery reads.

- Install or refresh the Codex skills from upstream with
  `curl -fsSL https://raw.githubusercontent.com/Lum1104/Understand-Anything/main/install.sh | bash -s codex`,
  then restart Codex.
- Launch the dashboard with `/understand-dashboard` from the repository root.
- Ask graph-grounded questions with `/understand-chat` when available, but verify benchmark,
  metric, schema, and paper-facing conclusions against source files and repo validation commands.
- Keep `.understand-anything/knowledge-graph.json` and `.understand-anything/fingerprints.json`
  in Git LFS. Do not commit `intermediate/`, `tmp/`, or `diff-overlay.json`.
- Prefer post-commit or explicit refreshes over pre-commit graph generation; graph updates are too
  slow and too mutating for the commit preparation path.
- For detailed setup, update, and artifact policy, read `docs/ai/understand_anything.md`.

## Local Machine Context (Gitignored)

To support multi-machine workflows, agents may consume machine-specific guidance from local-only
files at the repository root.

- Committed template: `docs/templates/local.machine.example.md`
- Local overrides (gitignored): `local.machine.md` or `local.machine.<name>.md`

Execution rules:

- If a local machine context file exists, read it before running expensive commands.
- Follow local limits for concurrency and execution location (for example CPU worker caps, tmux
  requirements, GPU-only jobs, or SLURM submission constraints).
- If no local machine context exists, use conservative defaults and repository-safe commands.
- Never store secrets in local machine context files.

## Fresh Worktree Bootstrap

When working in a linked Git worktree, detect bootstrap state before running expensive commands.

- For manually created worktrees, prefer a sibling container next to the main checkout instead of a
  directory inside the repository. Use `<repo-name>.worktrees/<branch-or-issue-slug>`; for this
  checkout, that means paths like
  `../robot_sf_ll7.worktrees/issue-123-short-description`.
- Honor an explicit user or native-tool worktree location first. If no preference is given, prefer
  an existing sibling `../robot_sf_ll7.worktrees/` container, then an existing project-local
  `.worktrees/` or `worktrees/` container only when it is already established and ignored. Default
  new manual worktree containers to the sibling `../robot_sf_ll7.worktrees/` path.
- Name branch/worktree directories with the issue number and a short feature slug when possible,
  for example `issue-123-short-description`.
- Treat the checkout as a linked worktree when `.git` is a file pointing into `.git/worktrees/...`,
  or when `git rev-parse --git-common-dir` resolves to a different path than
  `git rev-parse --git-dir`.
- Treat the worktree as fresh only when that linked-worktree signal is present and the root is
  missing both `local.machine.md` and `.venv` (plus any other team-specific initialized marker).
  If either already exists, assume bootstrap has already happened and reuse the existing setup.
- In a fresh worktree, find the main repository root from the common Git dir before bootstrapping:
  `MAIN_REPO_ROOT="$(cd "$(git rev-parse --git-common-dir)/.." && pwd)"`.
- If `local.machine.md` is absent in the worktree and present in the main repository root, create a
  symlink instead of copying the file. Example:
  `ln -s "$MAIN_REPO_ROOT/local.machine.md" .`
- After the machine-context symlink is in place, run `uv sync --all-extras`, then
  `source .venv/bin/activate` before using Python tooling.
- For quick targeted validation from a sibling worktree that intentionally reuses the main
  checkout virtualenv, prefer `scripts/dev/run_worktree_shared_venv.sh -- <uv-run-command>` so
  `PYTHONPATH` is pinned to the active worktree while `UV_PROJECT_ENVIRONMENT` points at the shared
  `.venv`. Use a full local `.venv` and final PR readiness for merge proof.
- Do not add CARLA to the routine `--all-extras` bootstrap. For CARLA-capable worktrees, opt into
  the host-side Python client with `uv sync --all-extras --group carla`, then prove the local
  runtime with `scripts/dev/check_carla_runtime.sh` or `scripts/dev/check_carla_runtime.sh --smoke`
  when it is acceptable to start the simulator container.
- If the current branch is not `main`, fetch the latest `origin/main` and merge it into the current
  branch early in the work cycle so the branch benefits from repository-wide fixes and workflow
  improvements before local changes diverge. Typical command sequence:
  `git fetch origin main && git merge origin/main`.
- Do not create divergent per-worktree machine context files unless the worktree really needs
  machine-specific behavior that should not be inherited from the main checkout.

## Worktree Teardown And Preservation

Treat worktree cleanup as a normal closeout step after PR review, issue implementation, publishing,
or abandoned exploration. Once a worktree is no longer needed for active validation, CI follow-up,
artifact recovery, or handoff, either remove it safely or record why it must be preserved.

Before removing or pruning worktrees, enumerate them with `git worktree list --porcelain` from the
main checkout. For each candidate, inspect `git -C <path> status --short --branch`; when generated
outputs may matter, also inspect ignored output paths such as
`[ -d "<path>/output" ] && git -C <path> status --ignored --short -uall output`.
For token-sensitive autonomous cleanup or repositories with many linked worktrees, start with the
compact hygiene helper instead of printing the full worktree inventory in the parent thread:
`uv run python scripts/dev/worktree_hygiene_snapshot.py --repo-status --json`. Add `--filter
<branch-or-path-substring>` for a single branch cleanup, and read raw `git worktree list
--porcelain` only when the compact payload is insufficient or reports a stale administrative entry
that needs manual inspection.

- Preserve every relevant tracked, untracked, and ignored-but-important local change before removal
  by committing it, stashing it, saving a patch, promoting a durable artifact, or recording an
  explicit handoff.
- Do not remove a dirty worktree or a worktree with unpushed commits unless the preservation record
  says exactly what was kept or why nothing needed preservation.
- Inspect large ignored directories such as `output/` before removal. Classify them as disposable,
  ignored cache, tracked manifest/evidence, durable-required, or handoff-needed; never treat
  worktree-local `output/` contents as durable artifact storage.
- Prefer `git worktree remove <path>` for clean worktrees and `git worktree prune` only after
  verifying stale administrative entries no longer point at useful local state.

## Knowledge Capture & Context Notes

Treat `docs/context/` as the repository's Markdown knowledge base for issue execution history and
agent handoff, not as a dump of incidental scratch notes. This tree should stay aggressively
indexed, pruned, and refactored; stale or superseded notes should be marked or removed when touched.
The broader retrieval-architecture redesign is tracked in GitHub issue #1714.

Treat `memory/` as the complementary repo-local memory layer for stable cross-session facts:
architecture summaries, durable decisions, reusable experiment outcomes, known failure modes, and
benchmark memory boundaries. Keep `memory/MEMORY.md` concise and update linked topic files for
detail instead of turning the index into another full instruction document.

- For relevant non-trivial work, persist reusable insights, decisions, reasoning, validation notes, and
  handoff context in Markdown when that context would otherwise be trapped in chat, PR text, or
  issue comments.
- Use `docs/context/` for issue-specific execution history and validation detail; use `memory/`
  for knowledge that should be reused across multiple future sessions.
- Prefer updating an existing canonical note when it already covers the same topic. Create a new
  note only when the subject is distinct enough that merging would hide the decision trail.
- Link notes to the related issue/PR, relevant canonical docs, proof artifacts, and any predecessor
  or successor note when a document is superseded.
- If a touched note contains outdated or superseded statements, update them, remove them, or mark
  them clearly with a pointer to the current source of truth.
- Keep note names and links discoverable from normal contributor entry points. Start broad reads
  from `docs/context/INDEX.md`; use `docs/context/README.md` and
  `.agents/skills/context-note-maintainer/SKILL.md` when creating or refreshing context notes.

## Cross-Agent Compatibility

For the repository's stance on adapting external agent-workflow sources (coding-agents, Awesome
Copilot, etc.) and the retrieval → planning → execution → verification discipline, see:

- `docs/context/issue_728_coding_agents_compatibility.md` — canonical accept/reject record for
  `vultuk/coding-agents` concepts and the four-phase agent loop mapped to Robot SF skills.
- `docs/ai/awesome_copilot_adaptation.md` — decisions for the Awesome Copilot List adaptation.

## Project Structure & Module Organization
Core simulation code lives in `robot_sf/` with key subpackages: `gym_env` for Gymnasium bindings, `sim` for physics glue, `nav` for path planning, and `render` for playback tooling. Training and evaluation entry points sit in `scripts/`, while curated demos and notebooks live under `examples/`. Tests are split between `tests/` (unit and integration), `tests/pygame/` (GUI regressions), and the `fast-pysf/` subtree. Assets and checkpoints are versioned under `maps/svg_maps/` and `model/`; the canonical (git-ignored) artifact root for generated outputs is `output/` (legacy `results/` has been migrated there).

## Durable Artifacts & Worktree Output

- Treat `output/` as temporary, worktree-local, and generally untracked. Do not assume files under `output/` exist in another checkout, worktree, or machine.
- Do not rely on worktree-local `output/` contents for durable dependencies unless the launcher can hydrate them from a canonical source.
- Any artifact required by future runs, benchmarks, reports, or release workflows must be promoted to a durable location before downstream steps depend on it.
- Prefer durable publication through W&B artifacts/model uploads or another explicit persistent store with enough metadata to recover the exact artifact later.
- Keep lightweight tracked manifests or registry entries when they describe how to retrieve, validate, or identify a durable artifact.
- Use `docs/context/evidence/` for small, reviewable copies of generated evidence that supports a
  durable context note or benchmark decision, such as compact `summary.json` files, Markdown
  reports, CSV/JSON tables, and checksums. Do not mirror `output/` wholesale.
- Keep raw episode JSONL, videos, large Slurm logs, coverage HTML, model caches, and checkpoints out
  of git unless there is a narrow, reviewable fixture reason. If those artifacts are reproducible
  from tracked configs, seeds, commits, and commands, they may be left ignored or deleted locally.
  If they are expensive and non-regenerable, archive them in W&B, release storage, or another
  durable store and track only the manifest/pointer.
- Do not add Git LFS as the default answer for generated benchmark artifacts. Consider Git LFS only
  for a deliberately versioned, non-regenerable binary fixture after an explicit maintainer decision.
- Worktree launchers that need `output/model_cache/...` inputs should either hydrate them from a canonical cache or artifact reference, or fail with an actionable error that names the missing durable source.
- Avoid committing bulky generated outputs directly unless the file is intentionally small, reviewable, and part of the source contract.

## Build, Test & Development Commands
Set up dependencies with `uv sync --all-extras` and install hooks via `uv run pre-commit install`. Format and lint using `uv run ruff check .` followed by `uv run ruff format .`. Run the main suite with `uv run pytest tests`; add `-m "not slow"` to skip long benches. Headless GUI checks use `DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy uv run pytest tests/pygame`. Validate the SocialForce backend with `uv run python -m pytest fast-pysf/tests -v`. Typical training workflows call `uv run python scripts/training/train_ppo.py --config configs/training/ppo/expert_ppo_issue_576_br06_v3_15m_all_maps_randomized.yaml`.

For shared local + VS Code + Codex workflows, prefer:
- `scripts/dev/ruff_fix_format.sh`
- `uv run python scripts/dev/run_compact_validation.py -- <command>` for bounded parent-thread validation output
  with full logs and summary JSON under the common Git dir
- `scripts/dev/run_tests_parallel.sh` (uses `pytest -n auto -x --failed-first` by default; supports wrapper flags `--new-first`, `--no-ordering`, `--no-fast-fail`)
- `BASE_REF=origin/main scripts/dev/pr_ready_check.sh`
- `scripts/dev/gh_comment.sh` for multiline `gh` PR/issue comments (stdin or `--body-file`, avoids literal `\n` formatting issues)

For GitHub comments with Markdown-heavy bodies, do not pass the body as an inline shell string.
Use `scripts/dev/gh_comment.sh`, `gh issue/pr comment --body-file`, or REST JSON input. This is
mandatory when the body contains backticks, YAML, shell commands, multiline Markdown, or anything
the shell could expand before `gh` receives it.

## Config-First Strategy
Prefer a config-first workflow for reproducibility and reviewability. Add or update YAML files under `configs/` for stable experiments and document the canonical command using `--config <path>`. Use CLI flags only for short-lived overrides while iterating locally.

## Coding Style & Naming Conventions
The project enforces Ruff with a 4-space indent, 100-character lines, and double-quoted strings (`pyproject.toml`). Prefer type-annotated interfaces and keep factory functions (`environment_factory.make_*`) as the public entry point. Modules and files use `snake_case`; classes and dataclasses follow `PascalCase`. Name tests `test_<feature>.py` and keep fixtures under `conftest.py`. Avoid ad-hoc prints in library code—use the existing structured logging. Prefer to use more docstrings (for private methods also) and inline comments for clarity, especially in complex algorithms or data flows.

## Testing Guidelines
Target the full `tests/` suite before pushing changes and rerun targeted slow markers when behavior or performance may shift. GUI and physics suites are mandatory for changes touching rendering, SocialForce integration, or pedestrian dynamics. Record notable validation runs in `docs/context/` or other tracked notes when benchmarks change, and only keep small explicit manifests or reviewable artifacts under version control. Do not treat worktree-local `output/` as the durable record. Update or add smoke tests under `scripts/validation/` when introducing new critical workflows. Agents may remove low-value tests without maintainer approval when the reason is unambiguous and documented; do not assume flaky tests are common without evidence.

## Research-Progress-First Validation

Research progress is the primary goal. Verification should be strong where claims are strong and
cheap where risk is low. Any benchmark-facing, metric-facing, schema-facing, skill, or test change
must still be backed by concrete evidence appropriate to the risk.

For research-producing work, use the following as adapted guidance rather than a hard gate:
research issues and PRs should prefer work that moves a claim boundary, closes or revises a
hypothesis, records a useful negative result, synthesizes accumulated diagnostics, or unblocks a
durable experiment. If a result remains diagnostic-only, label it that way and avoid treating it as
claim movement.

When practical, research issue contracts should name:

- target claim or hypothesis;
- comparator or baseline;
- minimum valid evidence;
- decision or stop rule;
- expected artifacts and provenance plan;
- parent issue, claim map, registry, context note, or synthesis surface to update.

After several diagnostic child PRs under one research parent, prefer a synthesis pass before adding
more exploratory families. The synthesis should classify what was learned, what stayed
inconclusive, which families are redundant or negative, and what follow-up experiment, if any,
should run next.

- New local planners must be proven with an actual benchmark or targeted execution path that shows
  they run correctly in this repository.
- Metric changes must include a clear proof that the updated metric now computes the intended values
  or fixes the intended regression.
- New skills must be checked against their real invocation path, referenced files, and repository
  workflow fit.
- New tests must be shown to fail for the right reason before the fix when practical, or otherwise
  justified with direct evidence that they cover the intended contract.

Benchmark-specific policy:

- Use the canonical fail-closed benchmark fallback note:
  `docs/context/issue_691_benchmark_fallback_policy.md`
- Fallback behavior is **not** acceptable as a successful benchmark outcome unless the task
  explicitly exists to measure that fallback mode.
- If a planner, environment, or dependency cannot satisfy the contract needed for an accurate
  benchmark run, the run for that planner must fail closed with a clear error and an explicit
  `not available` or `failed` status.
- If the intended proof fails or cannot be gathered, close the work as `blocked`, `diagnostic`, or
  `not benchmark evidence` as appropriate, and record the next smallest proof step instead of
  presenting the claim as complete.
- Do not classify fallback execution as benchmark-strengthening evidence; report it as a limitation
  or exclusion reason with the exact condition that triggered it.
- Benchmark reports and issue follow-ups should clearly identify whether a planner ran in
  `native`, `adapter`, `fallback`, or `degraded` mode, and fallback/degraded should be treated as a
  caveat, not a success condition.

Prefer proof that matches the risk:

- benchmark or planner changes: benchmark run, policy-analysis run, or other executable evidence,
- training/config workflow changes: canonical command run or smoke path,
- metrics/schema changes: targeted assertions plus a reproducible sample or fixture,
- docs/skill/instruction changes: verify referenced paths, commands, and discoverability surfaces.

Do not present benchmark, metric, schema, model-provenance, or paper-facing changes as complete
until the proof is recorded in the validation notes, PR text, or issue follow-up. For docs-only or
low-risk instruction changes, use the cheap validation path by default and name any skipped
expensive gates in the PR or handoff.

## Commit & Pull Request Workflow
Adopt the conventional commit style seen in history (e.g., `refactor: adjust observation scaling`).
Each PR should summarize intent, reference related issues, and list the commands you ran. Include
screenshots or short GIFs when UI or playback output changes, and note any new assets placed under
`maps/` or `model/`. Before opening a PR, fetch the latest `origin/main`, merge or rebase it into
the feature branch, and then run `BASE_REF=origin/main scripts/dev/pr_ready_check.sh`; readiness
proof from before that sync is stale and must not be used for PR creation. Ensure CI stays green by
resolving lint or test failures locally before requesting review.
- Docs-only and low-risk instruction branches use a cheaper official path by default: inspect the
  diff, verify referenced paths where practical, and state when the full readiness gate was not run.
- Do not wait until PR creation to pick up `main` branch improvements on long-lived branches.
  Merge the latest `origin/main` into the current branch at the start of active work, and repeat
  that sync before PR creation so validation covers the newest shared baseline.
- Before creating a PR, inspect newly created `output/*` files, including ignored paths via
  `[ -d output ] && git status --ignored --short -uall output`. Decide whether each output is
  disposable, should remain ignored, should be represented by a tracked manifest or registry entry,
  or must be uploaded to a durable artifact store before the branch is handed off.
- Fill the PR template's `Downstream Propagation` section for evidence-producing PRs. Check whether
  the parent issue, claim map or benchmark report, leaderboard or artifact catalog, registry or
  config index, context index or memory note, and deferred follow-up issue need updates. For
  low-risk or not-applicable changes, state the reason explicitly instead of deleting the section.
  PR #2044 is a recent small example: it promoted a compact trace-viewer screenshot and updated the
  context index/catalog so the evidence remained discoverable after worktree cleanup.
- For research-labelled, benchmark-labelled, metric-facing, paper-facing, or other
  evidence-producing PRs, fill the `Research Result Guidance` section with the target
  claim/hypothesis/blocker, comparator/baseline, evidence tier, result classification,
  decision/stop rule, and synthesis surface/update target.
  For support/tooling/docs-only PRs with no research claim, use `NA` and state why.
- When a PR adds a new research, benchmark, metric, or paper-facing analysis tool, either use it
  once on durable/versioned input in the same PR, or link a concrete follow-up issue that names the
  decision, claim boundary, benchmark report, registry, context note, or synthesis surface the tool
  will update. Disposable local `output/` files do not count as durable proof unless represented by
  a tracked manifest, registry entry, context note, or external artifact pointer. Small support
  helpers that are not intended to support research interpretation may state `NA - support helper`
  with that reason. Trace-panel generators, topology-score instrumentation, seed-sufficiency
  analysis, and why-report generation are research-facing examples that need first use or a
  concrete follow-up.
- For delegated-worker routing, prefer a compact preflight ledger before implementation or review
  dispatch: `gh auth status`, `scripts/dev/snapshot_pr_queue.py --prs <number> --expected-head-sha <sha> --json`,
  and `scripts/dev/check_pr_ci_status.py --expected-head-sha <sha>`.
- Default claimable issue snapshots should keep blocked external-data issues out of agent routing;
  use `scripts/dev/snapshot_issue_batch.py --blocked-external-report` for the parked human-action
  report, or `--include-blocked-external` only when explicitly auditing that queue.
- Use `scripts/dev/snapshot_issue_batch.py --active-portfolio` when reviewing the broader open
  issue portfolio for executable, human-decision, blocked-external, diagnostic-only, stale
  synthesis, and paper-critical routing recommendations without applying labels automatically.
- Treat `preflight.status == "stale"` lanes as invalid until refreshed.
- If review/comment data is needed, start from compact `review_snapshot`/`comment_snapshot`/`checks`
  output rather than raw `gh` payloads. For PR review threads, use
  `scripts/dev/snapshot_pr_queue.py --prs <number> --review-threads --json`; it emits bounded
  comment excerpts, label names, and omits raw `diff_hunk` payloads. Fetch full review-comment
  bodies or hunks only with an explicit artifact path such as
  `--raw-review-comments-artifact "$(git rev-parse --path-format=absolute --git-common-dir)/codex-agent-runs/.../raw-review-comments.json"`.
- Before finalization, cover format/check, focused tests, changed-file coverage when practical, and a
  clean worktree check (`git status --short`) as part of the readied proof bundle.
- For failing validation commands, prefer `uv run python scripts/dev/run_compact_validation.py -- <command>` before
  pasting raw test, lint, type-check, or docs output into an agent parent thread. The wrapper stores
  the full log and `summary.json` under the common Git dir and prints only command, exit code,
  elapsed time, artifact paths, failing pytest node ids when present, and bounded excerpts.
- Before broad status or inventory output, use
  `uv run python scripts/dev/autopilot_state_snapshot.py --include-worktrees` or another compact
  helper so generated `.venv`, `.opencode`, `node_modules`, and `output` trees are summarized
  instead of dumped into agent context.
Prefer GitHub MCP / GitHub app tools for interactive repository interactions such as viewing,
commenting on, and triaging issues and PRs. Keep the GitHub CLI (`gh`) for scripted batch
operations, auth debugging, and fallback when MCP coverage is insufficient.
When review feedback or PR scope identifies deferred follow-up work, always create a dedicated GitHub issue with `gh` before closing out the task.
When referencing files in PRs, issue comments, docs, and agent responses, use repository-root-relative paths (for example, `robot_sf/nav/svg_map_parser.py`) instead of absolute local filesystem paths like `/Users/...`.

## Communication Depth

- Prefer concise-but-explanatory responses over terse status-only updates.
- For benchmark and planner findings, include:
  - what changed,
  - why it matters,
  - what risk/limitation remains.
- When citing metric values, add one line interpreting the implication (for example, safety improved but success unchanged).
- If uncertainty remains, clearly separate observed evidence from hypothesis.

## Planning Convention

For non-trivial work, follow `.agents/PLANS.md`:
- restate the goal and boundaries first,
- list evidence sources before implementation,
- keep validation commands explicit,
- state what proof will demonstrate the change actually works here,
- record follow-up risks separately from completed scope.

Do not treat plans as throwaway scratch text when they influence benchmark semantics, model provenance, or public docs.

## GitHub Workflow Batching

When working issue batches or Project #5 updates:

- treat Project #5 ordering and score fields as advisory planning inputs, not as hard authority over
  current maintainer direction or fresh evidence,
- use REST (`gh api repos/...`) for ordinary issue/PR/label/comment operations when GraphQL quota
  is low or when the operation does not need Projects v2,
- prefer REST endpoints for simple label/comment publication writes even before quota is low when
  `gh` routes through brittle GraphQL surfaces; PR #2520 hit a classic Projects deprecation error
  via `gh pr edit --add-label merge-ready` while the REST issue-label endpoint worked immediately,
- for labels, use `gh api repos/:owner/:repo/issues/<number>/labels -f labels[]=<label>` to add
  and `gh api -X DELETE repos/:owner/:repo/issues/<number>/labels/<label>` to remove,
- for comment creation or patching, use body files or JSON payloads such as
  `gh api repos/:owner/:repo/issues/<number>/comments -F body=@body.md` and
  `gh api -X PATCH repos/:owner/:repo/issues/comments/<comment-id> -F body=@body.md`,
- use local `git` for branch, diff, merge-base, and commit state instead of asking GitHub,
- clean up issue text and labels first,
- route Project #5 metadata in a separate pass,
- run derived score sync once at the end of the batch,
- cache project and field IDs once per shell session or in the local gitignored
  `.github/cache/project5.json` cache for long-running/multi-agent work,
- reserve GraphQL for Projects v2, review threads, and nested reads that are genuinely cheaper,
- check `gh api rate_limit` before large batches and degrade to REST/local state when GraphQL
  remaining is low.

Canonical note:

- `docs/context/issue_713_batch_first_issue_workflow.md`

## Autonomous Usage Stop Guard

When a goal, background task, or autopilot loop has a Codex usage stop threshold, a usage check
below that threshold is a hard pause state for the loop. Persist the pause in the common Git dir
resolved with `git rev-parse --path-format=absolute --git-common-dir`, for example
`$(git rev-parse --path-format=absolute --git-common-dir)/codex-agent-runs/active/`. Do not write
to a literal worktree-relative `.git/codex-agent-runs/active/` path, because linked worktrees may
store `.git` as a file. After recording the pause, short-circuit repeated automatic continue
prompts without repo, GitHub, validation, delegation, or broad context-loading work.

- Do not re-run usage checks on automatic continue prompts unless a recorded cooldown has elapsed.
- A direct user request for current usage may run one fresh usage check.
- A direct user override may resume work, but record that the stop guard was overridden.
- Use a compact repeated-pause response such as `Paused: weekly remaining 13% < 28%. No actions.`
- Do not mark the active goal complete while paused unless a real completion audit already proved
  every requirement before the pause fired.

## Key Codex Skills

For issue management and delivery, use these local skills:

- `.agents/skills/goal-issue-discovery/SKILL.md`
  - Autonomous goal loop for finding evidence-graded improvement opportunities and opening detailed issues.
- `.agents/skills/issue-audit/SKILL.md`
  - User-in-the-loop issue audit that asks one readiness-blocking question at a time and updates issues as decisions crystallize.
- `.agents/skills/goal-issue-implementation/SKILL.md`
  - Sequential goal loop for implementing eligible open issues through branch, validation, push, and PR creation.
- `.agents/skills/goal-pr-review/SKILL.md`
  - Autonomous PR review loop that fixes scoped writable gaps before applying `merge-ready` after the full proof bar passes.
- `.agents/skills/gh-issue-autopilot/SKILL.md`
  - Autonomous issue-to-PR workflow: select next best issue, branch, implement, validate, push, and open a ready PR.
- `.agents/skills/gh-issue-clarifier/SKILL.md`
  - Tightens ambiguous issues with pros/cons/recommendation and applies `decision-required` when maintainer input is needed.
- `.agents/skills/gh-issue-priority-assessor/SKILL.md`
  - Reviews Project #5 priority inputs against the static rubric, explains plausibility, and keeps field writeback opt-in.
- `.agents/skills/analyze-camera-ready-benchmark/SKILL.md`
  - Runs consistency diagnostics for camera-ready benchmark campaigns and summarizes runtime/quality/fallback signals.

Use the repo-local context skills under `.agents/skills/` when the task is primarily about
understanding or reviewing benchmark/planner context rather than executing GitHub workflow
automation:

- `.agents/skills/benchmark-overview/SKILL.md`
- `.agents/skills/context-note-maintainer/SKILL.md`
- `.agents/skills/experiment-context/SKILL.md`
- `.agents/skills/planner-integration/SKILL.md`
- `.agents/skills/paper-facing-docs/SKILL.md`
- `.agents/skills/review-benchmark-change/SKILL.md`

## Spark Sidecar Routing

Spark (`gpt-5.3-codex-spark`, or the configured Spark sidecar model) is eligible for small, low-risk
read-only task classes:

- **tiny lookup** — file location, name resolution, short grep.
- **read-only review** — narrow diff inspection, single-file summary.
- **docs cross-check** — link validation, path reference checks.
- **issue/file surface mapping** — issue-to-file coverage, surface enumeration.

Spark prompts must require compact output: files inspected, exact evidence, uncertainty, and
recommended next prompt.
For long autonomous runs, cache Spark usage-limit failures in the active ledger with the reset time.
Before spawning another Spark sidecar in the same run, check that ledger entry and route directly to
the next eligible cheap worker if Spark is still unavailable.
If a Spark spawn still fails for quota, close any allocated subagent handle immediately and treat
the failure as route evidence only.

Spark is explicitly excluded from:

- final benchmark interpretation and paper claims,
- merge readiness and publication decisions,
- GitHub mutation (labels, comments, PR creation, merge, close),
- long CI polling unless a bounded monitor helper exists,
- shell-executable fallback unless a real headless wrapper is available.

Do not configure Spark as a shell-executable fallback; this is routing guidance only.

## Donts

- Never change code in `.venv`. To manage dependencies, edit `pyproject.toml` and run `uv sync` to update the virtual environment.
