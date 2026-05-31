# Repository Guidelines

Use `AGENTS.md`, `.specify/memory/constitution.md`, and `docs/dev_guide.md` to guide AI assistants.
This document covers briefly the repository structure, coding style, testing workflow, and contributor conventions.
Prefer reusable shell entry points under `scripts/dev/` for automation and AI skills.
Use `.vscode/tasks.json` as thin wrappers around those scripts.
Keep prompts, instructions, and handoff notes as streamlined and token-efficient as possible while preserving the same meaning and constraints.

## Maintainer Value Hierarchy

Optimize first for concrete research progress on social-navigation simulation, benchmarking, and
planner exploration. Apply proof, documentation, and process in proportion to risk:

- benchmark, metric, schema, model-provenance, and paper-facing claims still require strong,
  reproducible evidence before they are treated as established;
- exploratory planner or research work may move faster when its status is clearly labeled as
  exploratory, diagnostic, blocked, or not yet benchmark evidence;
- low-risk docs, metadata, and instruction changes may use a cheaper validation path when the user
  explicitly asks for it, as long as the skipped checks are named in the PR or handoff;
- current maintainer direction overrides stale workflow prose. When an instruction appears to
  conflict with the user's current priority, follow the current priority, call out the conflict, and
  propose or make the smallest doc update needed to remove the drift.

## Agent Context Stack
Treat the following files as the repository-native context stack for Agent-style agents:

- `AGENTS.md`: top-level execution rules, repo structure, and workflow defaults.
- `memory/MEMORY.md`: concise repo-local memory index for stable cross-session facts and links to
  detailed topic files under `memory/`.
- `docs/code_review.md`: benchmark-facing review criteria, provenance checks, and regression traps.
- `docs/context/INDEX.md`: retrieval-first catalog for current context-note entry points, status
  rules, optional context tools, and curated context-pack scopes.
- `.agent/PLANS.md`: plan-writing convention for non-trivial work so intent, scope, and validation stay explicit.
- `.agents/skills/`: canonical skill tree for execution workflows and repo-local context packs,
  mirrored at `.codex/skills/` and `.opencode/skills/` for compatibility.
- `.agents/prompts/`, `.agents/commands/`, and `.agents/agents/`: canonical prompt, command,
  and GitHub agent sources, mirrored into tool-specific compatibility paths when possible.
- `docs/ai/`: AI-facing overview documents for repo structure, planner-zoo state, context packing, and deferred retrieval decisions.

Read only the surfaces relevant to the task. Prefer these repo-local files over ad-hoc summaries in issue comments, and avoid loading broad context-note indexes unless the task actually needs them.

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
- If the current branch is not `main`, fetch the latest `origin/main` and merge it into the current
  branch early in the work cycle so the branch benefits from repository-wide fixes and workflow
  improvements before local changes diverge. Typical command sequence:
  `git fetch origin main && git merge origin/main`.
- Do not create divergent per-worktree machine context files unless the worktree really needs
  machine-specific behavior that should not be inherited from the main checkout.

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
Core simulation code lives in `robot_sf/` with key subpackages: `gym_env` for Gymnasium bindings, `sim` for physics glue, `nav` for path planning, and `render` for playback tooling. Training and evaluation entry points sit in `scripts/`, while curated demos and notebooks live under `examples/`. Tests are split between `tests/` (unit and integration), `test_pygame/` (GUI regressions), and the `fast-pysf/` subtree. Assets and checkpoints are versioned under `maps/svg_maps/` and `model/`; the canonical (git-ignored) artifact root for generated outputs is `output/` (legacy `results/` has been migrated there).

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
Set up dependencies with `uv sync --all-extras` and install hooks via `uv run pre-commit install`. Format and lint using `uv run ruff check .` followed by `uv run ruff format .`. Run the main suite with `uv run pytest tests`; add `-m "not slow"` to skip long benches. Headless GUI checks use `DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy uv run pytest test_pygame`. Validate the SocialForce backend with `uv run python -m pytest fast-pysf/tests -v`. Typical training workflows call `uv run python scripts/training/train_ppo.py --config configs/training/ppo/expert_ppo_issue_576_br06_v3_15m_all_maps_randomized.yaml`.

For shared local + VS Code + Codex workflows, prefer:
- `scripts/dev/ruff_fix_format.sh`
- `scripts/dev/run_tests_parallel.sh` (uses `pytest -n auto -x --failed-first` by default; supports wrapper flags `--new-first`, `--no-ordering`, `--no-fast-fail`)
- `BASE_REF=origin/main scripts/dev/pr_ready_check.sh`
- `scripts/dev/gh_comment.sh` for multiline `gh` PR/issue comments (stdin or `--body-file`, avoids literal `\n` formatting issues)

## Config-First Strategy
Prefer a config-first workflow for reproducibility and reviewability. Add or update YAML files under `configs/` for stable experiments and document the canonical command using `--config <path>`. Use CLI flags only for short-lived overrides while iterating locally.

## Coding Style & Naming Conventions
The project enforces Ruff with a 4-space indent, 100-character lines, and double-quoted strings (`pyproject.toml`). Prefer type-annotated interfaces and keep factory functions (`environment_factory.make_*`) as the public entry point. Modules and files use `snake_case`; classes and dataclasses follow `PascalCase`. Name tests `test_<feature>.py` and keep fixtures under `conftest.py`. Avoid ad-hoc prints in library code—use the existing structured logging. Prefer to use more docstrings (for private methods also) and inline comments for clarity, especially in complex algorithms or data flows.

## Testing Guidelines
Target the full `tests/` suite before pushing changes and rerun targeted slow markers when behavior or performance may shift. GUI and physics suites are mandatory for changes touching rendering, SocialForce integration, or pedestrian dynamics. Record notable validation runs in `docs/context/` or other tracked notes when benchmarks change, and only keep small explicit manifests or reviewable artifacts under version control. Do not treat worktree-local `output/` as the durable record. Update or add smoke tests under `scripts/validation/` when introducing new critical workflows.

## Research-Progress-First Validation

Research progress is the primary goal. Verification should be strong where claims are strong and
cheap where risk is low. Any benchmark-facing, metric-facing, schema-facing, skill, or test change
must still be backed by concrete evidence appropriate to the risk.

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
low-risk instruction changes, it is acceptable to skip expensive gates when the skipped checks are
explicitly named.

## Commit & Pull Request Workflow
Adopt the conventional commit style seen in history (e.g., `refactor: adjust observation scaling`).
Each PR should summarize intent, reference related issues, and list the commands you ran. Include
screenshots or short GIFs when UI or playback output changes, and note any new assets placed under
`maps/` or `model/`. Before opening a PR, fetch the latest `origin/main`, merge or rebase it into
the feature branch, and then run `BASE_REF=origin/main scripts/dev/pr_ready_check.sh`; readiness
proof from before that sync is stale and must not be used for PR creation. Ensure CI stays green by
resolving lint or test failures locally before requesting review.
- Docs-only and low-risk instruction branches may use a cheaper official path when the user asks for
  it: inspect the diff, verify referenced paths where practical, and state that the full readiness
  gate was intentionally not run.
- Do not wait until PR creation to pick up `main` branch improvements on long-lived branches.
  Merge the latest `origin/main` into the current branch at the start of active work, and repeat
  that sync before PR creation so validation covers the newest shared baseline.
- Before creating a PR, inspect newly created `output/*` files, including ignored paths via
  `git status --ignored --short -uall output`. Decide whether each output is disposable, should remain
  ignored, should be represented by a tracked manifest or registry entry, or must be uploaded to a
  durable artifact store before the branch is handed off.
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

For non-trivial work, follow `.agent/PLANS.md`:
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
  - Autonomous issue-to-PR workflow: select next best issue, branch, implement, validate, push, and open draft PR.
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

## Donts

- Never change code in `.venv`. To manage dependencies, edit `pyproject.toml` and run `uv sync` to update the virtual environment.
