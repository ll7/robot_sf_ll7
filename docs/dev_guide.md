# Robot SF – Development Guide

[← Back to Documentation Index](./README.md)

Welcome to the Robot SF Development Guide! This document serves as the central reference for contributors working on the Robot SF codebase. It covers setup instructions, architectural overviews, coding standards, and best practices to ensure a smooth development experience.
<!--
This document should be kept as short as possible to maintain clarity and ease of navigation.
Whenever possible, link out to more detailed documents or external resources.
Refactor this document regularly to be as concise as possible.
LLM Constitution and guides can be found here:
- `docs/maintainer_values.md`
- `.specify/memory/constitution.md`
- `.github/copilot-instructions.md`
- `AGENTS.md`
-->

## Setup

### Installation and setup

```bash
# Check host tools that live outside uv.
scripts/dev/check_runtime_requirements.sh

# One‑time setup with all extras and pre-commit
uv sync --all-extras
source .venv/bin/activate
uv run pre-commit install

# Quick import check
uv run python -c "from robot_sf.gym_env.environment_factory import make_robot_env; print('Import successful')"
```

Host tools and optional machine capabilities that are not installed by `uv` are tracked in
[`docs/dev_runtime_requirements.md`](dev_runtime_requirements.md).

### Instruction precedence and proportional readiness

Use the maintainer hierarchy and readiness matrix in `AGENTS.md` before older workflow prose or
tool-specific compatibility pointers. In short: active maintainer direction wins over stale
instructions, `docs/maintainer_values.md` defines the hard contracts, and Project #5 scores are
advisory when fresh evidence or maintainer direction conflicts with them.

Routine workflow cleanup can proceed without extra confirmation when it is bounded and the PR or
handoff clearly labels assumptions, uncertainty, evidence grade, and any deferred follow-up issue.
Use a detached checkout at latest `origin/main` only for read-only discovery, duplicate checks, and
issue creation or update work. Create or switch to a branch/worktree before editing docs or code,
running validation for a PR, pushing, or publishing.

For long autonomous goals, delegated batches, and token-saving threads, seed the active prompt or
resume summary with `docs/templates/token_efficient_thread_profile.md`. The profile keeps
`task_class`, `validation_tier`, context budget, delegation artifact requirements, and output budget
explicit without duplicating the maintainer hierarchy or readiness matrix.

Docs-only and instruction-only changes normally use the cheap validation path: inspect the diff,
verify changed links or paths where practical, and run available lightweight checks. Skill or AI
workflow edits should also run the relevant skill and sync checks, for example:

```bash
uv run python scripts/dev/check_skills.py --preflight <skill-name>
uv run python scripts/tools/sync_ai_config.py --check
```

Escalate to `BASE_REF=origin/main scripts/dev/pr_ready_check.sh` when the change touches scripts,
schemas, generated indexes, routing behavior, automation, runtime behavior, benchmark/metric/schema
semantics, model provenance, or paper-facing claims.

For benchmark scenario, metric, model-profile, and release-evidence changes, first apply the
[Benchmark Scenario And Model Governance](benchmark_governance.md) review contract. It defines the
versioning, comparability, reproduction, and deprecation details that PRs must make explicit before
benchmark or paper-facing claims are treated as established.

Use `BASE_REF=origin/main scripts/dev/pr_ready_check.sh` for the dependency-minimal core
readiness lane. If a change touches predictive or other optional-extra paths and you need the
optional proof lane directly, run `ROBOT_SF_TEST_LANE=optional scripts/dev/run_tests_parallel.sh
--lane optional`.
This lane split was introduced for issue #3301 and PR #3314; the executable source of truth is
`scripts/dev/pr_ready_check.sh` dispatching to `scripts/dev/run_tests_parallel.sh`.

### Claim-map validation

The fast-results claim map is an executable issue queue, not only a context note. Before changing
`docs/context/issue_2943_fast_results_claim_map_v0.md`, run:

```bash
uv run python scripts/dev/check_fast_results_claim_map.py --json
```

This check is also part of `scripts/dev/pr_ready_check.sh`. It verifies that each priority row has
a status, p0 rows have exactly one owner issue and one next command or artifact, and completed rows
point at durable evidence instead of worktree-local `output/`.

### Fresh linked-worktree bootstrap

When creating a new linked worktree, prefer a sibling container next to the main checkout rather
than a directory inside the repository. For this checkout, use
`../robot_sf_ll7.worktrees/<branch-or-issue-slug>` unless a user or native tool
chooses another location. Keep issue work readable with names such as
`issue-123-short-description`.

Example manual creation from the main checkout:

```bash
MAIN_REPO_ROOT="$(git rev-parse --show-toplevel)"
WORKTREE_PARENT="$(dirname "$MAIN_REPO_ROOT")/$(basename "$MAIN_REPO_ROOT").worktrees"
mkdir -p "$WORKTREE_PARENT"
git fetch origin main
git worktree add -b issue-123-short-description \
  "$WORKTREE_PARENT/issue-123-short-description" \
  origin/main
cd "$WORKTREE_PARENT/issue-123-short-description"
```

Bootstrap the local machine context before using Python tools. You can detect a linked worktree
because `.git` is a file that points into
`<main checkout>/.git/worktrees/<worktree-name>`, and `git rev-parse --git-common-dir` resolves to
the main checkout's `.git` directory instead of the worktree-local Git dir.

Treat the worktree as fresh only if both `local.machine.md` and `.venv` are absent. If either
already exists, assume the worktree has already been bootstrapped and reuse the existing setup.

A cheap fresh-worktree check is:

```bash
[ "$(git rev-parse --git-common-dir)" != "$(git rev-parse --git-dir)" ] \
  && [ ! -e local.machine.md ] \
  && [ ! -d .venv ]
```

Use this order for a fresh worktree:

```bash
MAIN_REPO_ROOT="$(cd "$(git rev-parse --git-common-dir)/.." && pwd)"
ln -s "$MAIN_REPO_ROOT/local.machine.md" .
uv sync --all-extras
source .venv/bin/activate
```

Notes:

- The symlink target should point at the main checkout's local machine context, not a copied
  per-worktree file.
- If the worktree path differs, derive the correct source from `$MAIN_REPO_ROOT/local.machine.md`.
- Reuse the symlinked `local.machine.md` instead of copying it so machine-specific limits stay in
  sync across worktrees.
- CARLA is intentionally not part of `uv sync --all-extras`. For a CARLA-capable worktree, add the
  host-side Python client explicitly with `uv sync --all-extras --group carla`, then run
  `scripts/dev/check_carla_runtime.sh` for preflight or `scripts/dev/check_carla_runtime.sh --smoke`
  for the bounded Docker connectivity proof.
- If you are starting work on a feature branch, merge the latest `origin/main` into the current
  branch early so you inherit repository-wide fixes and workflow improvements before your local
  changes diverge. Typical command sequence:

```bash
git fetch origin main
git merge origin/main
```

### Worktree teardown and preservation

Make worktree cleanup part of normal closeout after PR review, issue implementation, publishing, or
abandoned exploration. If a worktree is no longer needed for active validation, CI follow-up,
artifact recovery, or handoff, remove it safely or record why it is intentionally preserved.

Before deleting old worktrees, run `git worktree list --porcelain` from the main checkout and inspect
each candidate with `git -C <path> status --short --branch`. If the worktree may contain generated
evidence or local experiment outputs, also inspect relevant ignored paths, for example
`[ -d "<path>/output" ] && git -C <path> status --ignored --short -uall output`.

Only remove a worktree after preserving relevant tracked, untracked, and ignored-but-important
changes through a commit, stash, patch, durable artifact promotion, or explicit handoff note. Do not
delete dirty or unpushed worktrees unless the cleanup record states what was preserved or why nothing
needed preservation. Classify large ignored directories such as `output/` before removal as
disposable, ignored cache, tracked manifest/evidence, durable-required, or handoff-needed; do not let
worktree-local `output/` become durable artifact storage. Use `git worktree remove <path>` for clean
worktrees; reserve `git worktree prune` for stale administrative entries after local state is
checked.

### Targeted shared-venv worktree validation

For quick, targeted checks in a sibling worktree, you can reuse the main checkout virtualenv while
pinning imports to the current worktree:

```bash
scripts/dev/run_worktree_shared_venv.sh -- pytest tests/test_ci_script_contract.py -q
scripts/dev/run_worktree_shared_venv.sh --venv ../robot_sf_ll7/.venv -- ruff check scripts/dev
scripts/dev/run_worktree_shared_venv.sh --standalone -- \
  python scripts/dev/check_docs_evidence_integrity.py --files docs/dev_guide.md
```

The helper runs from `git rev-parse --show-toplevel`, sets `UV_PROJECT_ENVIRONMENT` to the shared
`.venv`, and sets `UV_NO_SYNC=1`. By default it also prepends the worktree root to `PYTHONPATH`.
This is intended for fast local feedback when dependencies are already current. It should fail if
the shared virtualenv is missing instead of silently installing into the wrong checkout.

Use `--standalone` for a dependency-light command whose tests verify that it does not import
`robot_sf` or other project packages. This mode still reuses third-party dependencies from the
shared environment, but it skips the project-source freshness check and does not add the worktree
root to `PYTHONPATH`. For example, `check_docs_evidence_integrity.py` has a minimal-environment
import guard in `tests/tooling/test_docs_evidence_import_boundary.py`, so it remains safe to run when
an unrelated installed `pysocialforce` copy is stale. Do not use this mode for tests or commands
that import project code; refresh the owning checkout environment for those commands instead.

If a fresh linked worktree fails to collect a focused test because an optional dependency such as
`torch` is not installed in that worktree, rerun the same focused command through
`scripts/dev/run_worktree_shared_venv.sh -- uv run pytest <test-node>`. A pass through the wrapper
classifies the direct failure as setup or optional-dependency friction for that worktree, not as a
code regression; record both commands in the PR or handoff.

### Agent-run artifact paths in linked worktrees

In a linked worktree, `.git` is a file (not a directory), so writing to a literal
`.git/codex-agent-runs/active/...` path fails.  Use the shared helpers to resolve the
correct absolute path via `git rev-parse --git-common-dir`.

**Shell** (for scripts that source `scripts/dev/common_setup.sh`):

```bash
source scripts/dev/common_setup.sh
artifact_dir="$(resolve_agent_artifact_dir my-subdir)"
mkdir -p "$artifact_dir"
echo "data" > "$artifact_dir/result.json"
```

**Python** (for scripts under `scripts/dev/`):

```python
from scripts.dev.git_common import resolve_agent_artifact_dir

artifact_dir = resolve_agent_artifact_dir("my-subdir")
# artifact_dir is an absolute Path; mkdir is done automatically
```

**One-liner** (for ad-hoc shell use or agent instructions):

```bash
mkdir -p "$(git rev-parse --path-format=absolute --git-common-dir)/codex-agent-runs/active/my-subdir"
```

Never hard-code a literal `.git/codex-agent-runs/...` path in scripts, agent instructions, or
task artifact wording.  Always resolve through `git rev-parse --git-common-dir` or use the
helpers above.

When validating the SNQI (Social Navigation Quality Index) contract or camera-ready exit handling,
pass the relevant files explicitly. `-k` filters only after pytest has collected files, so starting
from `pytest tests -k ...` can import unrelated optional stacks first. This command collects only
the files that own the checks:

```bash
DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy scripts/dev/run_focused_tests.sh \
  tests/unit/benchmark/test_snqi_campaign_contract.py \
  tests/benchmark/test_camera_ready_campaign.py \
  tests/tools/test_run_camera_ready_benchmark.py \
  -k "snqi_contract or exit or camera_ready_summary" -q
```

If a change adds another focused contract test, append its file path to this command rather than
falling back to the whole `tests` tree. This is a collection boundary, not a replacement for the
optional readiness lane when the changed tests actually require optional dependencies.

Use a normal worktree-local `uv sync --all-extras` and
`PR_READY_MODE=final BASE_REF=origin/main scripts/dev/pr_ready_check.sh` for final PR proof,
dependency changes, generated lockfile validation, or any run where environment isolation matters.

### Critical dependencies and setup: Fast-pysf integration

The `fast-pysf/` directory contains the optimized SocialForce physics engine and is now integrated as a **git subtree** (previously a submodule). After cloning the repository, the fast-pysf code is automatically available—no additional initialization steps required.

**Note**: If you're working with an older branch that still uses submodules, see the [Subtree Migration Guide](./SUBTREE_MIGRATION.md) for migration instructions and workflow differences.

### Quick Start Commands

```bash
# source .venv
source .venv/bin/activate
# Lint+format
uv run ruff check --fix . && uv run ruff format .
# Tests
uv run pytest -n auto tests
```

### Examples Quickstart Walkthrough

The `examples/README.md` file now captures a curated onboarding path. New contributors
can get a full tour in roughly five minutes by running the quickstart trio in order:

```bash
uv run python examples/quickstart/01_basic_robot.py
uv run python examples/quickstart/02_trained_model.py
uv run python examples/quickstart/03_custom_map.py
```

- `01_basic_robot.py` introduces the environment factory pattern and headless rollouts.
- `02_trained_model.py` replays the bundled PPO baseline and writes JSONL metrics to
  `output/results/episodes_demo_ppo.jsonl`.
- `03_custom_map.py` shows how to load `maps/svg_maps/debug_06.svg` via
  `RobotSimulationConfig.map_pool` for custom layouts.

See `examples/README.md` for the decision tree, prerequisites, and links to additional
tiers (advanced features, benchmarks, plotting, and archived scripts).

### Advanced Feature Demos

Developers exploring specific capabilities should jump to the curated scripts in
`examples/advanced/`. Each file follows the numbered naming scheme surfaced in
`examples/README.md` and comes with a manifest-backed docstring describing how to
run it. Highlights include:

- **Backends & factory ergonomics**: `01_backend_selection.py` and
  `02_factory_options.py` demonstrate switching simulators and recording options via
  unified configs.
- **Observation & training workflows**: `03_image_observations.py` and
  `04_feature_extractors.py` showcase image sensors and feature extractor presets
  (run with `uv sync --extra training`, or `uv sync --all-extras` for full local parity).
- **Pedestrian & policy scenarios**: Scripts `06`–`11` cover factory-based
  pedestrian environments, single/multi pedestrian setups, and PPO rollouts using
  the maintained checkpoints under `model/`.
- **Tooling, validation, and visualization**: `12_social_force_planner_demo.py`
  through `15_view_recording.py` provide the Social Force planner showcase, SVG
  map validation helper, trajectory visualization, and recording playback flows.

Check the Advanced table in `examples/README.md` for prerequisites, tags, and whether
a script is enabled for CI smoke execution.

### Model registry

Trained policies are tracked in a local registry to make reuse and automation
easier:

- Human-readable notes: `model/registry.md`
- Machine-readable registry: `model/registry.yaml`
- Helper API: `robot_sf.models.resolve_model_path(...)` for on-demand loading
  (auto-downloads from W&B when metadata is present).

Use `robot_sf.models.upsert_registry_entry(...)` to auto-populate or update the
registry from training pipelines.
Benchmark-promoted learned checkpoints must also include `benchmark_promotion`
observation-track metadata; see `model/registry.md` and
`docs/context/issue_1612_observation_track_architecture.md`.

### One‑liner quality gates (CLI):

```bash
uv run ruff check --fix . && uv run ruff format . && uvx ty check . --exit-zero && uv run pytest -n auto tests
```

`ty` currently runs in advisory mode with `--exit-zero`: it reports findings, but the canonical
typecheck phase is not a PR-readiness merge blocker by itself.

### Merge-race prevention (ADR — issue #5389)

**Problem.** Three main-red incidents in 36 hours (2026-07-11/12) had the same shape: two PRs, each
green on its own merge-ref, broke main when both landed in a 3-second merge race. The red-main merge
hold (#5385) stops breakage *stacking* once main is red, but nothing prevented the race itself: a
PR's CI ran against a main that moved before the merge landed.

**Decision: gate-side staleness check.** We adopt option 2 from the issue — a staleness rule that
prevents merging a PR whose CI ran against a stale main:

- **Script**: `scripts/dev/check_pr_merge_staleness.py <pr-number>`.
- **Integration**: the `gh-pr-merger` skill runs this check as preflight step 6 before any merge.
- **Behavior**: when the check detects that main has moved since the PR's CI ran, it returns exit
  code 1 and the merger skips the PR with a staleness report. The precise path reads the completed
  workflow run's recorded `pull_requests[].base.sha`; when that provenance is unavailable, the
  checker falls back to the PR base-vs-main comparison. The author must `gh pr update-branch` and
  re-run CI before the PR becomes mergeable again.

**Why not GitHub merge queue?** The native merge queue is the ideal solution — it re-validates each
PR against the up-to-date prospective main before merging automatically. We chose the gate-side rule
because:

1. It works immediately without enabling a repository-level feature that requires maintainer approval
   to toggle branch-protection settings.
2. It provides the same merge-race guarantee at the cost of slightly more manual branch updates.
3. It is easy to roll back — remove the preflight step from the skill and the script can be deleted.

**When to revisit.** If the native merge queue becomes available and is enabled, the gate-side
staleness check can be replaced by the queue's built-in re-validation, which is strictly stronger.
The gate-side rule remains useful as a safety net for non-GitHub CI providers.

**Rollback path.** Remove step 6 from `.agents/skills/gh-pr-merger/SKILL.md` and
`.opencode/skills/gh-pr-merger/SKILL.md`. The script `scripts/dev/check_pr_merge_staleness.py`
and its tests can be deleted at that point.

### Reusable dev scripts

Prefer calling shared scripts from `scripts/dev/` so VS Code tasks, local shells, and Codex
skills use the same commands:

```bash
scripts/dev/ruff_fix_format.sh
scripts/dev/run_tests_parallel.sh
scripts/dev/run_worktree_shared_venv.sh -- pytest tests/test_ci_script_contract.py -q
scripts/dev/run_ci_local.sh
scripts/dev/local_signoff.sh --no-setup lint test
scripts/dev/check_docs_proof_consistency_diff.sh
scripts/dev/sbatch_use_max_time.sh --partition <partition> --qos <qos> --sbatch-arg --partition=<partition> --sbatch-arg --qos=<qos> SLURM/templates/gpu_training.sl
BASE_REF=origin/main scripts/dev/pr_ready_check.sh
PR_READY_MODE=final BASE_REF=origin/main scripts/dev/pr_ready_check.sh
uv run python scripts/dev/complexity_runtime_baseline.py --top 10 robot_sf scripts tests
uv run python scripts/dev/ci_timing_summary.py --run-id <github-actions-run-id> --top 10
scripts/dev/gh_comment.sh pr --current <<'EOF'
Summary line
- bullet 1
- bullet 2
EOF
```

`scripts/dev/run_ci_local.sh` is the local CI-equivalent entrypoint for the shared
validation phases. By default it runs `uv sync --all-extras --frozen`, migrates legacy artifacts,
then delegates to `scripts/dev/ci_driver.sh` so local runs and `.github/workflows/ci.yml`
share the same phase definitions (`lint`, `typecheck`, `test`, `examples-smoke`, `smoke`, and
`artifact-policy`). Pass explicit phases to scope a run, for example
`scripts/dev/run_ci_local.sh lint test`. After dependencies are already current, use
`scripts/dev/run_ci_local.sh --no-setup lint test` for faster repeat local feedback.

`scripts/dev/local_signoff.sh` is the optional local-CI attestation wrapper. It runs selected
`run_ci_local.sh` phases, auto-installs the `basecamp/gh-signoff` GitHub CLI extension if missing,
and posts advisory `signoff/local-*` statuses only after the worktree is clean and `HEAD` is already
pushed to its push remote. It never calls `gh signoff install` and never changes branch protection.
Use `scripts/dev/local_signoff.sh --no-setup lint test` for fast repeat local proof, or
`scripts/dev/local_signoff.sh --full` before a higher-confidence handoff.

Before opening a PR, fetch the latest `origin/main`, integrate it into the feature branch with
either merge or rebase, and only then run
`PR_READY_MODE=final BASE_REF=origin/main scripts/dev/pr_ready_check.sh`.
Final mode refuses to write readiness evidence unless the non-ignored worktree is clean, so the
stamp represents committed `HEAD` rather than an interim dirty-tree check. Plain
`BASE_REF=origin/main scripts/dev/pr_ready_check.sh` remains useful for local feedback while edits
are in progress; if it records a dirty-tree stamp, treat that stamp as interim and rerun final mode
after committing. The `BASE_REF` value tells the readiness gate what to compare against; it does
not update the feature branch by itself, so validation from before the latest-main sync is stale for
PR creation.
Do not wait until PR creation to pick up `main` branch improvements on long-lived feature branches;
merge latest `origin/main` into the current branch when active work starts, then sync again before
opening the PR.

Use `uv run python scripts/dev/complexity_runtime_baseline.py --top 10 robot_sf scripts tests`
before/after substantial refactor PRs when you need a quick, repeatable snapshot of largest modules,
longest functions, and optional pytest duration rows from a captured `--pytest-log`.
Use `uv run python scripts/dev/ci_timing_summary.py --run-id <github-actions-run-id> --top 10`
when GitHub CI wall time drifts from local readiness and you need queue, job, and slowest-step
timings from `gh run view` data.

For routine autopilot CI waits, prefer the compact monitor helper instead of leaving the parent
thread idle on raw GitHub output. In a fresh linked worktree, run it through the shared-venv
wrapper so uv reuses the owning checkout's environment and does not create or prompt for a local
`.venv`:

```bash
scripts/dev/run_worktree_shared_venv.sh -- python scripts/dev/check_pr_ci_status.py \
  <pr-number> \
  --expected-head-sha <head-sha> \
  --poll-attempts 40 \
  --poll-interval 30 \
  --max-wall-seconds 1200 \
  --json
```

The wrapper sets `UV_PROJECT_ENVIRONMENT` to the owning checkout's `.venv` and `UV_NO_SYNC=1`,
so the command works from a worktree that has not run `uv sync`. The `--help` output of
`scripts/dev/check_pr_ci_status.py` also prints this invocation for quick agent copy/paste.
Use `--max-wall-seconds` to give long-running monitors a clean local stop path before patching or
pushing a branch; exit code 2 means checks were still pending when the local cap expired, not that
remote GitHub checks were cancelled or failed.

Each JSON payload includes `monitor` metadata for the active delegation ledger: expected head SHA,
SHA-match result, poll attempt, wait budget, optional wall-clock cap, deadline, and
`route_evidence_only: true`. When the local wall cap expires while checks are still pending, the
payload also includes `monitor.local_stop_reason: "max_wall_seconds"`. Monitor success is route
evidence only; reassess the current PR head SHA and normal readiness proof before labeling or
merging.

When CI polling repeats similar JSON in the parent thread, switch to status-change summaries and
write the full monitor payload to the active ledger or a common-Git-dir artifact. For in-progress
run debugging, inspect job metadata first; fetch direct job-log excerpts only for the failing or
suspect step, and avoid `gh run view --log` dumps until the run is complete enough for that command
to return useful output.

When a completed job's normal log is absent (for example, a runner infrastructure failure omitted
the job from the log archive), recover its retained check-run annotations with:

```bash
uv run python scripts/dev/diagnose_actions_job.py <job-id>
```

The helper prints normal logs when they are available and otherwise prints the annotations linked
from the job metadata. It exits nonzero if neither source provides diagnostics.

For routine goal-autopilot orientation, prefer the compact state snapshot helper before broad parent
thread reads:

```bash
uv run python scripts/dev/autopilot_state_snapshot.py \
  --include-worktrees \
  --claim-issue <issue-number> \
  --issue-search "is:issue is:open <queue-filter>" \
  --pr <pr-number>
```

The JSON output includes source commands, branch/head SHA, `origin/main` SHA, linked worktrees,
claim refs, issue queue rows, explicit PR headline state, compact tracked status, generated-path
presence, a `controller_checkpoint`, and freshness metadata. Use the checkpoint as the first
resume artifact after compaction or automatic continuation: it should name the active branch/PR,
known generated paths, stale claims, check state, and next action without reopening raw logs,
issue queues, worktree inventories, or skill files. Compact status omits generated untracked trees
such as `.venv`, `.opencode`, `node_modules`, and `output`, reporting only the generated
roots that are present. Run fresh focused `gh`/`git` checks before
claim, push, PR, label, merge, or publication decisions. Raw logs and broad CLI output are
appropriate when the snapshot reports `ok: false`, stale claims, missing state, or insufficient
fields.
Worktree rows are capped by default; use `worktree_count` and `worktrees_truncated` to decide
whether a larger `--worktree-limit` is worth the parent-thread context cost.
For remote cleanup and branch-drift triage, use the read-only hygiene snapshot before broad
`git worktree` output or stale-worktree cleanup:

```bash
uv run python scripts/dev/worktree_hygiene_snapshot.py --repo-status --json
```

The payload reports total and included worktree counts, dirty worktrees, missing upstreams,
ahead/behind drift, detached heads, and truncation status. Use `--filter <branch-or-path-substring>`
or `--worktree-limit <n>` when remote hosts have many linked worktrees.

For delegation routing and PR-review polling, treat `snapshot_pr_queue` as the entry point:

- Preflight lanes with `--expected-head-sha <sha>` before dispatch.
- Reuse `preflight.status` (`healthy` | `stale` | `blocked`) and `next_action` to avoid stale or noisy routes.
- Invalidate stale-lane routes (refresh snapshot) before reassigning or reviewing.
- Start review loops from compact `review_snapshot`, `comment_snapshot`, and `checks` output, not raw
  full-comment payloads.

```bash
uv run python scripts/dev/snapshot_pr_queue.py --prs 2677 --json \
  --expected-head-sha "$PR_HEAD_SHA"
```

The resulting JSON keeps review/comment/CI payloads compact; review noise is reduced to counts,
latest author-attributed samples, and bounded body excerpts.

Use `BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh` before PR handoff when a
branch adds or edits context notes, evidence bundles, or other proof-heavy docs surfaces. The
checker is intentionally conservative: it only flags high-confidence issues such as missing
`docs/context/README.md` links for new top-level context notes, tracked evidence files that still
contain absolute local paths, and tracked evidence that links to ignored `output/` artifacts.
It also validates the curated machine-readable context catalog at `docs/context/catalog.yaml` so
indexed context entry points keep explicit status and freshness metadata.
Run `uv run python scripts/validation/check_docs_proof_consistency.py --check-evidence-catalog`
for the explicit full evidence-catalog hygiene pass; it scans tracked
`docs/context/evidence/` bundles and reports bundles that have no catalog entry.
When issue or PR text needs to classify proof strength, use the
[artifact evidence vocabulary](context/artifact_evidence_vocabulary.md) so local `output/` paths are
not promoted into durable benchmark or paper-facing claims.

#### Issue-reading fallback

`gh issue view <number> --comments` can fail on some GitHub CLI versions with a
`repository.issue.projectCards` GraphQL deprecation error. Use the targeted REST
fallback (see issues #5186 and #5188):

```bash
# Drop-in shell wrapper (tries native CLI first, falls back to REST on projectCards):
bash scripts/dev/gh_issue_view.sh <number> --repo ll7/robot_sf_ll7

# Direct REST helper (same fallback logic, more options):
uv run python scripts/dev/gh_issue_rest.py thread <number> --repo ll7/robot_sf_ll7

# Explicit REST read with normalized fields (stable JSON output shape):
uv run python scripts/dev/gh_issue_rest.py view <number> --repo ll7/robot_sf_ll7 --comments
uv run python scripts/dev/gh_issue_rest.py view <number> --json number title state url labels comments
```

All issue-delivery skills (`gh-issue-autopilot`, `gh-issue-clarifier`,
`goal-issue-implementation`, etc.) already route to `gh_issue_rest.py thread` when
`gh issue view --comments` fails; see
`docs/context/issue_713_batch_first_issue_workflow.md` for the full command reference.

For GitHub issue batches and Project #5 updates, follow the batch-first workflow note:

- `docs/context/issue_713_batch_first_issue_workflow.md`
- Use REST-backed `gh api repos/...` calls for ordinary issue, label, PR, branch, commit, and
  workflow-run operations when possible.
- Reserve GraphQL for Projects v2 operations, review-thread operations, and nested reads that are
  genuinely cheaper.
- Use local `git` for branch, diff, merge-base, and commit state instead of asking GitHub.
- Prefer GitHub MCP / GitHub app tools for interactive issue, PR, and project work when available,
  but switch to REST for issue cleanup when GraphQL quota is low.
- Keep `gh` for scripted batch operations, derived score sync, auth debugging, REST fallback, and
  one-off deterministic commands.
- Clean up issues first, then route Project #5 metadata, then run derived score sync once at the end.
- Cache project and field IDs once per shell session instead of rediscovering them for every issue;
  for long-running or multi-agent work, use a local gitignored `.github/cache/project5.json` cache
  following `docs/templates/github.project5-cache.example.json`.
- Check `gh api rate_limit` before large batches and leave Project #5 writes pending when GraphQL
  is exhausted instead of retry-looping.
- For low-GraphQL or long autonomous publication runs, keep a REST-first command ledger covering
  PR creation, commit check-run polling, issue comments, labels, merge, and branch cleanup. Treat
  remote branch deletion reporting a missing ref after merge as a cleanup caveat to record, not as
  evidence that the merge failed; verify the merged PR or base branch SHA instead.

### REST-first publication snippets for low-GraphQL autopilot

For token-efficient autopilot runs, collect compact local snapshots before broad
GitHub or repository reads:

```bash
uv run python scripts/dev/snapshot_issue_batch.py 2665 2675 --json
uv run python scripts/dev/snapshot_issue_batch.py 2665 2675 --json \
  --capsule-dir "$(git rev-parse --path-format=absolute --git-common-dir)/codex-agent-runs/active"
uv run python scripts/dev/snapshot_issue_batch.py --claimable --json
uv run python scripts/dev/snapshot_issue_batch.py --blocked-external-report \
  --report-path "$(git rev-parse --path-format=absolute --git-common-dir)/codex-agent-runs/active/blocked-external-assets.md"
uv run python scripts/dev/snapshot_issue_batch.py --active-portfolio \
  --report-path "$(git rev-parse --path-format=absolute --git-common-dir)/codex-agent-runs/active/active-issue-portfolio.md" \
  --json
uv run python scripts/dev/snapshot_pr_queue.py --prs 2677 2678 2679 --json \
  --expected-head-sha "$PR_HEAD_SHA"
uv run python scripts/dev/pr_babysitter_snapshot.py 2679 --expected-head-sha "$SHA" --json
uv run python scripts/dev/watch_pr_ci_status.py 2679 --expected-head-sha "$SHA" --json --once
```

Default `--claimable` issue snapshots omit issues classified as blocked on external data, assets,
licenses, or human staging input. Use `--include-blocked-external` only when deliberately auditing
that parked queue, or `--blocked-external-report` to generate a compact human-action report with
monthly review dates. Use `--active-portfolio` for a compact non-mutating open-issue portfolio
that classifies executable, human-decision, blocked-external, diagnostic-only, stale synthesis, and
paper-critical rows with owner types and label-change recommendations.

Use the snapshot JSON to seed worker prompts and active ledgers. Redirect broad
search output or raw GitHub bodies to private agent-run artifacts; return only
the compact snapshot, context capsule path, validation command, exit status, and
short evidence excerpt to the parent Codex thread.

For routine repository discovery, exclude the dense tracked evidence archive unless the archive is
the explicit search target. Start with a focused command such as:

```bash
rg -n "<concept>" AGENTS.md docs scripts tests robot_sf .agents \
  --glob '!docs/context/evidence/**' --glob '!output/**'
```

Omit the evidence exclusion only when the task is to inspect or validate evidence artifacts. This
keeps ordinary code and workflow matches visible without treating the archive as unimportant.

For implementation-thread validation, prefer `scripts/dev/run_focused_tests.sh`
for focused pytest targets. It stores the full pytest log under the common
Git-dir agent-run artifacts and prints only a bounded pass/fail summary by
default. Use `FOCUSED_TEST_FULL_OUTPUT=1` only when the raw pytest stream itself
is the thing being debugged.
For non-pytest gates or commands that may produce large failure logs, use
`uv run python scripts/dev/run_compact_validation.py -- <command>`. It stores the full log and
summary JSON under the common Git-dir agent-run artifacts and prints only the
command, exit code, elapsed time, artifact paths, failing pytest node ids when
present, and a bounded failure excerpt.

After delegated worker runs, summarize route efficiency from one or more
`scripts/dev/routed_worker_manifest.py` outputs without reading raw worker logs:

```bash
uv run python scripts/dev/route_efficiency_report.py output/issue-2764/worker/routing_manifest.json \
  --format markdown
```

The report counts delegated attempts, complete artifact sets, reroutes,
validation presence, and optional final acceptance metadata. It also emits a
`routing_recommendations` array with deterministic classes (`prefer_provider`,
`avoid_provider`, `investigate_failure_class`, `reroute_threshold_met`,
`no_recommendation`). Each entry includes `class`, `action`, `evidence`, and
`caveat` keys, and the markdown report includes a compact "Routing
recommendations" section. Route success and complete artifact presence are
**route evidence only**; they are not task acceptance. The orchestrator must
still inspect the diff and run the required local validation.

PR-loop dry-run policy can consume the same routed-worker manifests directly:

```bash
uv run python scripts/dev/pr_loop_policy.py --snapshot output/pr_queue.json \
  --manifest 1234=output/issue-2764/worker/routing_manifest.json --json
```

Manifest-driven decisions remain dry-run and mutation-free. Complete artifacts can unblock
`ready_to_merge` classification only when the compact PR snapshot is otherwise ready; missing
artifacts, failed validation text, stale expected heads, risky manifest paths, and draft PRs stay
reroute/stop signals instead of task acceptance.

For phase-end token audits, record one compact route-efficiency row before
starting another delegated batch:

- latest Codex usage snapshot and whether it is close to the user-defined stop
  guard;
- largest parent-thread outputs since the previous audit, such as broad
  multi-directory `rg` results, full skill rereads, raw validation output, raw
  GitHub JSON, or verbose delegate final messages;
- failed command patterns, confusing helper contracts, repeated monitor noise,
  and unclear instructions that caused retries or semantic drift;
- cache hits and misses for loaded skills, issue or PR snapshots, route quota
  facts, CI monitor commands, and validation artifacts;
- accepted, rejected, rerouted, and skipped delegates, with the reason the route
  was or was not cheaper than direct Codex work;
- next route change, such as using `rg --files | rg <pattern>` before content
  search, requiring smaller app-agent final messages, or reusing a recorded
  quota reset instead of retrying the same blocked route.

Keep the row in the active ledger or a common-Git-dir self-review note. Promote
it into durable docs only when the same leak repeats, the leak was expensive, or
the user explicitly asks for workflow improvements.

For implementation-thread reviews over a recent time window, do not reopen the full transcript in
the parent thread. Generate a bounded session-summary artifact that records the time window, record
counts, largest output records, broad commands, failed commands, and unclear-instruction themes; use
that artifact plus existing self-review notes as the evidence base for any workflow patch.

For historical route audits, add `--dashboard` and pass multiple routed-worker
manifest files. Dashboard mode emits `route_efficiency_dashboard.v1` JSON or
Markdown with overall metrics, per-manifest breakdowns, provider trends,
incomplete-provider and failure-class totals, common missing artifact counts,
and the same route-evidence-only recommendations and warning.

### PR Review: Route Efficiency

When reviewing PRs with route-efficiency changes, ensure:

- [ ] **Route completeness vs task success**: Complete artifacts or a zero exit code are
  not evidence that the task was accepted, correct, or merged.
- [ ] **Validation presence vs validation success**: Check `validation_presence.present`
  separately from `validation_presence.success_inferable`; a validation artifact can exist
  while the validation result failed or stayed ambiguous.
- [ ] **Reroute count interpretation**: Treat high reroute counts (2 or more) and
  `reroute_threshold_met` warnings as routing-friction evidence, not as proof that the
  final diff is wrong.
- [ ] **Raw-log avoidance**: Start from `scripts/dev/route_efficiency_report.py` outputs and
  compact worker artifacts; read raw worker logs only when compact evidence is missing,
  inconsistent, failed, or suspicious.
- [ ] **Visible evidence warning**: Confirm the report still displays the route-evidence-only
  warning and does not let route metrics replace manual diff inspection or local validation.

### Spark Sidecar Routing

Spark (`gpt-5.3-codex-spark`, or the configured Spark sidecar model) is a first-class route for
small, low-risk read-only task classes. Route Spark when the task fits one of:

- **tiny lookup** — file location, name resolution, short grep.
- **read-only review** — narrow diff inspection, single-file summary.
- **docs cross-check** — link validation, path reference checks.
- **issue/file surface mapping** — issue-to-file coverage, surface enumeration.
- **inspect small command output** — bounded stdout/stderr review.

Spark prompts must require compact output: files inspected, exact evidence, uncertainty, and
recommended next prompt.

Do not route Spark to:

- final benchmark interpretation and paper claims,
- merge readiness and publication decisions,
- GitHub mutation (labels, comments, PR creation, merge, close),
- long CI polling unless a bounded monitor helper exists,
- shell-executable fallback unless a real headless wrapper is available.

This is routing guidance only; do not configure Spark as a shell-executable fallback.

Assume `OWNER`, `REPO`, `ISSUE`, `BRANCH`, and `BASE` are set:

```bash
gh api repos/$OWNER/$REPO/pulls -X POST \
  -f title="Issue #$ISSUE: short summary" \
  -f head=$BRANCH -f base=$BASE \
  -f body="Automated publication update from Issue #$ISSUE"
```

```bash
PR=$(gh api repos/$OWNER/$REPO/pulls --method GET \
  -f state=open -f head="$OWNER:$BRANCH" \
  --jq '.[0].number')
uv run python scripts/dev/gh_pr_body_rest.py "$PR" --repo "$OWNER/$REPO" \
  --body-file /path/to/updated-pr-body.md
```

```bash
RUN_ID=$(gh api repos/$OWNER/$REPO/actions/runs \
  --method GET \
  -f branch=$BRANCH \
  -q '.workflow_runs | sort_by(.created_at) | reverse | .[0].id')
while :; do
  STATUS=$(gh api repos/$OWNER/$REPO/actions/runs/$RUN_ID --jq '.status')
  echo "run=$RUN_ID status=$STATUS"
  [ "$STATUS" = "completed" ] && break
  sleep 20
done
CONCLUSION=$(gh api repos/$OWNER/$REPO/actions/runs/$RUN_ID --jq '.conclusion')
PR_SHA=$(gh api repos/$OWNER/$REPO/pulls/$PR --jq '.head.sha')
gh api repos/$OWNER/$REPO/commits/$PR_SHA/check-runs \
  --jq '.check_runs[] | [.name, .status, .conclusion] | @tsv'
```

```bash
gh api repos/$OWNER/$REPO/issues/$ISSUE/comments -f body="CI=$CONCLUSION (run=$RUN_ID)"
gh api repos/$OWNER/$REPO/issues/$ISSUE/labels -X POST -f labels[]="autopilot-reviewed"
gh api repos/$OWNER/$REPO/issues/$ISSUE/labels/needs-publication -X DELETE --silent
```

```bash
gh api repos/$OWNER/$REPO/pulls/$PR/merge -X PUT \
  -f merge_method="squash" -f commit_title="Merge PR #$PR" -f sha="$PR_SHA"
```

```bash
gh api repos/$OWNER/$REPO/git/refs/heads/$BRANCH -X DELETE \
  || gh api repos/$OWNER/$REPO/git/refs/heads/$BRANCH --silent >/dev/null \
  || echo "branch ref already absent; confirm merge by checking issue/merge commit before closing out"
```

- GraphQL is still appropriate for: Projects v2 objects, review-thread resolution, and nested
  permission-dependent reads where REST needs multiple expanded requests and GraphQL is materially
  cheaper.

### Context note workflow

For non-trivial work, persist reusable insights, decisions, reasoning, validation notes, and
handoff context in Markdown instead of leaving them trapped in chat or PR history.

- Use `docs/context/README.md` as the canonical workflow and naming guide.
- Prefer updating an existing canonical note before creating a new one.
- If a touched note is outdated or superseded, update it, remove it, or mark it clearly with a
  pointer to the current source.
- Link notes to the related issue/PR, canonical docs, validation commands, and replacement notes.
- Use `.agents/skills/context-note-maintainer/SKILL.md` when the task includes creating or
  refreshing context notes.
- For docs/context-only changes, use the documented default gate: inspect the diff, verify changed
  README/INDEX/catalog links, and run `BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh`.
  The wrapper auto-detects context-only PRs and includes README/INDEX/catalog in that gate. State
  explicitly when benchmark, simulator, or full PR readiness gates were not run because the branch only
  changes discoverability or workflow text.

### Agent memory conventions

The repository now keeps a repo-local Markdown memory layer under `memory/` for stable cross-session
agent context.

- Start with `memory/MEMORY.md`, which acts as the concise index.
- Store reusable memory in typed subdirectories such as `memory/architecture/`,
  `memory/decisions/`, `memory/experiments/`, `memory/failures/`, and `memory/benchmarks/`.
- Use the experiment naming pattern `memory/experiments/YYYY-MM-DD_<topic>.md`.
- Keep `memory/MEMORY.md` short and push detail into linked topic files so it stays compatible with
  startup loading in agent runtimes that read project files.
- Use `docs/context/` for issue execution history and validation detail; use `memory/` only for
  knowledge worth reusing across future sessions.
- Optional MCP integration should expose the Markdown files directly; do not add a retrieval
  database or vector store unless the repository's retrieval-deferral policy changes.

### Question-first experiment registry

Use `experiments/registry.yaml` for planned or active exploratory ML/search/manual-control runs that
need a reviewable question, hypothesis, command, artifact expectation, evidence grade, and paper
relevance before execution. This registry complements GitHub issues, W&B artifacts, local telemetry
under `output/run-tracker/`, and publication bundles; it does not make local `output/` files durable.

Validate the registry with:

```bash
uv run python scripts/tools/validate_experiment_registry.py experiments/registry.yaml
```

On macOS, `scripts/dev/run_tests_parallel.sh` uses a bounded fixed xdist worker count by
default instead of `-n auto`, because the unbounded auto worker selection can leave local
validation wrappers hanging after child processes should have exited. Override with
`PYTEST_NUM_WORKERS=<int>` or `PYTEST_NUM_WORKERS=auto` when needed.

`scripts/dev/run_tests_parallel.sh` also accepts `PYTEST_XDIST_DIST=<mode>` to select the
pytest-xdist scheduler. The default remains `load` for compatibility. Use alternate schedulers
such as `PYTEST_XDIST_DIST=worksteal` only for targeted local experiments until hosted evidence
shows they improve CI timing without exposing new order dependencies.

For new SLURM batch jobs, prefer `scripts/dev/sbatch_use_max_time.sh` so the submitted
wall time tracks the live partition and QoS maximum instead of an outdated hardcoded
`#SBATCH --time` value. See `docs/dev/slurm_submission.md` for the workflow.

For paper-facing benchmark release runs, use the dedicated release wrapper:

```bash
uv run python scripts/tools/run_benchmark_release.py \
  --manifest configs/benchmarks/releases/paper_experiment_matrix_v1_release_v0_1.yaml
```

Release-process references:

- `docs/benchmark_release_protocol.md`
- `docs/benchmark_release_reproducibility.md`

### Benchmark fallback policy

Benchmark work is fail-closed by default. Use the canonical policy notes:

- `docs/context/issue_691_benchmark_fallback_policy.md`
- `docs/context/issue_1436_reproducibility_flaky_acceptance.md`

Fallback execution can still be useful for diagnostics and reproduction probes, but it must not be
reported as a successful benchmark outcome. Reruns are allowed only for environment-class failures,
not for benchmark contract failures, fallback/degraded execution, or unfavorable statistical
outcomes.

### Environment factory pattern (CRITICAL)

**Always use factory functions** — never instantiate gymnasium environments directly:

```python
from robot_sf.gym_env.environment_factory import make_robot_env, make_image_robot_env, make_pedestrian_env

# Basic robot navigation
env = make_robot_env(debug=True)

# With image observations  
env = make_image_robot_env(debug=True)

# Pedestrian environment (requires trained robot model)
env = make_pedestrian_env(robot_model=model, debug=True)
```

The compact reviewer contract for environment creation, rollout ownership, reward ownership,
benchmark verifier boundaries, and PPO run-record provenance is
`docs/training/environment_contract.md`.

### Key architectural layers

- **`robot_sf/gym_env/`**: Gymnasium environment implementations with factory pattern
- **`robot_sf/baselines/`**: Baseline navigation algorithms (e.g., SocialForce) for benchmarking
- **`robot_sf/benchmark/`**: Benchmark runner, CLI, metrics collection, and schema validation
- **`robot_sf/sim/`**: Core simulation components (FastPysfWrapper for pedestrian physics)
- **`fast-pysf/`**: Git subtree providing optimized SocialForce pedestrian simulation
- **`docs/`**: Documentation, design notes, and development guides

### Schema Management

**Canonical schema location**: `robot_sf/benchmark/schemas/`
- Episode schemas: `episode.schema.v1.json` (single source of truth)
- Runtime resolution: Use `robot_sf.benchmark.schema_loader.load_schema()` for schema loading
- Schema validation: Automatic validation against JSON Schema draft 2020-12
- Version management: Semantic versioning with breaking change detection
- Git hooks: Prevent duplicate schema files from being committed

### Data flow and integration

- **Training loop**: `scripts/training/train_ppo.py` →
  factory functions → vectorized environments → StableBaselines3 (`uv sync --extra training`)
- **RLlib workflow**: `scripts/training/train_dreamerv3_rllib.py` →
  factory functions → RLlib env registration → DreamerV3
  (`uv sync --extra rllib --extra training`)
- **Benchmarking**: `robot_sf/benchmark/cli.py` → baseline algorithms → episode runs → JSON/JSONL output → analysis
- **Pedestrian simulation**: Robot environments → FastPysfWrapper → `fast-pysf` subtree → NumPy/Numba physics

### Configuration hierarchy

**For complete documentation, see [Configuration Architecture](./architecture/configuration.md)** (precedence rules, migration guide, module structure).

### Config-first workflow (default)

Prefer committed YAML configs under `configs/` as the default way to run training and sweeps.
This keeps runs reproducible, reviewable, and easy to replay on another machine.

- Commit stable run definitions (scenario, seeds, metrics, cadence) in config files.
- Document a canonical `uv run ... --config <path>` command in docs/PR text.
- Reserve direct CLI tuning flags for temporary local overrides.

Use unified config classes from `robot_sf.gym_env.unified_config`:

```python
from robot_sf.gym_env.unified_config import RobotSimulationConfig, ImageRobotConfig

config = RobotSimulationConfig()
config.peds_have_static_obstacle_forces = True  # Enable pedestrian-obstacle forces
config.peds_have_robot_repulsion = True  # Enable pedestrian-robot repulsion
env = make_robot_env(config=config)
```

### Backend selection (simulator swap)

The simulation backend can be selected via configuration without modifying environment code. Available backends are registered in `robot_sf.sim.registry`:

```python
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.unified_config import RobotSimulationConfig

# Use fast-pysf backend (default)
config = RobotSimulationConfig()
config.backend = "fast-pysf"  # Default; can be omitted
env = make_robot_env(config=config)

# Use dummy backend (for testing)
config = RobotSimulationConfig()
config.backend = "dummy"
env = make_robot_env(config=config)
```

**Available backends:**
- `"fast-pysf"` (default): SocialForce pedestrian simulation via fast-pysf subtree
- `"dummy"`: Minimal test simulator with constant positions (for smoke tests)

**Backend registration:**
Custom backends can be registered via `robot_sf.sim.registry.register_backend()`. See `robot_sf/sim/backends/` for implementation examples.

**Error handling:**
Unknown backend names fall back to legacy `init_simulators()` with a warning. For strict validation, use `robot_sf.gym_env.config_validation.validate_config()` before environment creation.

### Planner selection (visibility vs classic grid)

Global planning can be toggled via `RobotSimulationConfig`:

```python
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.planner.classic_global_planner import ClassicPlannerConfig

config = RobotSimulationConfig(
    use_planner=True,
    planner_backend="classic",  # or "visibility"
    planner_classic_config=ClassicPlannerConfig(cells_per_meter=1.0, inflate_radius_cells=2),
)
env = make_robot_env(config=config)
```

- `"classic"` uses the grid-based planner (Theta*/A* family) and is the default.
- `"visibility"` uses the visibility-graph planner.

### Utility Modules

All shared utility functions and type definitions live in `robot_sf/common/`:
- `robot_sf/common/types` - Type aliases (Vec2D, Line2D, RobotPose, Circle2D, etc.)
- `robot_sf/common/errors` - Error handling utilities (raise_fatal_with_remedy, warn_soft_degrade)
- `robot_sf/common/seed` - Random seed management for reproducibility (set_global_seed, SeedReport)
- `robot_sf/common/compat` - Compatibility helpers (validate_compatibility)

**Example imports:**
```python
from robot_sf.common.types import Vec2D, RobotPose, Line2D
from robot_sf.common.errors import raise_fatal_with_remedy
from robot_sf.common.seed import set_global_seed

# Convenience imports also available:
from robot_sf.common import Vec2D, RobotPose, set_global_seed
```

**Troubleshooting:**
- If IDE autocomplete doesn't work after importing from `robot_sf.common`, restart your IDE's language server:
  - **VS Code**: Command Palette → "Python: Restart Language Server"
  - **PyCharm**: File → Invalidate Caches / Restart

## Design and development workflow recommendations

- Consider using <https://github.com/github/spec-kit> for complex, multi-contract specifications
  and design docs. Do not use it as the default governance layer for ordinary work.
  - Examples can be found in the `specs` directory.
  - Prompts are unique to the llm provider used. Adjust accordingly.
  - Canonical AI assistant content lives in `.agents/`:
    - Canonical skills live in `.agents/skills/`.
    - `.agents/skills/` is mirrored at `.codex/skills/` and `.opencode/skills/`.
    - `.agents/prompts/codex/` is mirrored at `.codex/prompts/`.
    - `.agents/prompts/github/` is mirrored at `.github/prompts/`.
    - `.agents/agents/github/` is mirrored at `.github/agents/`.
    - `.agents/commands/gemini/` is mirrored at `.gemini/commands/`.
  - Validate or repair supported mirrors with
    `uv run python scripts/tools/sync_ai_config.py --check` or
    `uv run python scripts/tools/sync_ai_config.py --fix`.
  - LLM Constitution and guides can be found here:
    - `docs/maintainer_values.md`
    - `.specify/memory/constitution.md`
    - `AGENTS.md`
  - For the repository's cross-agent compatibility stance and the retrieval → planning → execution
    → verification discipline mapped to repo-local skills, see
    `docs/context/issue_728_coding_agents_compatibility.md`.
  - For autonomous goal-loop skills around issue discovery, issue implementation, PR review, and
    user-in-the-loop issue audit, see
    `docs/context/goal_driven_agent_loops_2026-05-13.md`.
- Clarify exact requirements before starting implementation.
- If necessary, ask clarifying questions (with options) to confirm scope, interfaces, data handling, UX, and performance.
  - Discuss possible options and trade-offs.
  - Give arguments to the options for easy decision-making.
  - Provide options to quickly converge on a decision.
- For complex tasks:
  - Create a design doc (see template below) for non-trivial changes.
  - Create a file based TODO list (see example below).
  - Break task down into smaller subtasks and tackle them iteratively.
- Prioritize must-haves over nice-to-haves
- Document assumptions and trade-offs.
- Ensure that the documentation, docstrings, and comments are updated to reflect code changes.
- Docstring style is specified in `pyproject.toml` -> `[tool.ruff.lint.pydocstyle]` -> `convention`
- Progress cadence: always keep tests and documentation up-to-date. As long as you document your chain of thought and what ran, you can report outcomes after finishing the work.
- Prefer programmatic use and factory functions over CLI; the CLI is not important.
- Working mode: prioritize a thin, end-to-end slice that runs. Optimize and polish after a green smoke test (env reset→step loop or demo run).
- Whenever possible, add a demo or example to illustrate new functionality.
- Avoid disabling linters, type checks, or tests unless absolutely necessary.
  - Whenever you have the chance, refactor to fix issues rather than suppressing them. Especially `# noqa: C901` (complexity) and `# type: ignore` (type hints).
- Prefer refactoring over adding `# noqa` suppressions; only use `# noqa` as a short-lived exception with a clear plan to remove it.
- Always document the purpose of documents at the top of the file. (e.g., Python files, README.md, design docs, issue folders)
- Use American English.

### One-liner architecture summary

- Architecture in one line: Gymnasium envs → factory functions → FastPysfWrapper → fast-pysf physics; training/eval via StableBaselines3; baselines/benchmarks under `robot_sf/baselines` and `robot_sf/benchmark`.
- Environments: always create via factories (`make_robot_env`, `make_image_robot_env`, `make_pedestrian_env`). Configure via `robot_sf.gym_env.unified_config` only; toggle flags before passing to the factory.
- Simulation glue: interact with pedestrian physics through `robot_sf/sim/fast_pysf_wrapper.py`. Don’t import from `fast-pysf` directly inside envs.
- Baselines/benchmarks: get planners with `robot_sf.baselines.get_baseline(...)`. Prefer programmatic runners; CLI exists at `robot_sf/benchmark/cli.py` for convenience.
- Local planner adapters: start from `docs/dev/planner_adapter_template.md` and the diagnostic
  `reference_adapter` path before adding a new map-runner planner key.
- Demos/trainings: keep runnable examples in `examples/` and scripts in `scripts/`. Place models in `model/`, maps in `maps/svg_maps/`, and write outputs under `output/`.
- Tests: core in `tests/`; GUI in `tests/pygame/` (headless: `DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy`). Physics-specific tests live in `fast-pysf/tests/`.
- Quality gates (local): Install Dependencies → Ruff: Format and Fix → Check Code Quality (Ruff +
  advisory ty) → Type Check (advisory) → Run Tests (see VS Code Tasks).

### Test style conventions

- Default to pytest-style tests in `tests/` (functions + pytest fixtures).
- Using `unittest.mock` for mocks/stubs is fine, but avoid adding new `unittest.TestCase` suites.
- Legacy `fast-pysf/tests/unittest` remains unittest-based for upstream compatibility; new fast-pysf tests should still prefer pytest.

### Docstring-on-touch

- If you change a function/class body or signature, replace any `TODO docstring` placeholder in that scope with a real docstring.
- Keep it brief and accurate (one or two sentences is enough); focus on intent and non-obvious behavior.
- Purely mechanical edits (formatting, imports, lint fixes) do not require docstring updates.
- Avoid mass docstring sweeps; improve documentation incrementally as code changes.
- Use `uv run python scripts/validation/check_docstring_todos.py --mode report` to inspect the current placeholder backlog by top-level area and file.
- `scripts/validation/docstring_todo_baseline.json` is an increase-only ratchet. Update it with `--mode write-baseline` only after an intentional cleanup or maintainer-approved backlog change.
- Use `uv run python scripts/validation/check_active_doc_examples.py` to report stale active-doc command and artifact examples. Add `--fail-on-diagnostic` when a PR or CI lane should fail on new hits, and use an inline `active-docs-check: allow` marker only for intentional examples.

### Map Bounds Format

- `MapDefinition.bounds` accepts either flat tuples `(x_start, x_end, y_start, y_end)` or pair-of-points `((x1, y1), (x2, y2))`.
- At runtime, bounds are normalized to the flat tuple format because the fast-pysf backend and legacy utilities expect it.

### Artifact policy & tooling

- Canonical outputs live under `output/` with stable subdirectories: `output/coverage/`, `output/benchmarks/`, `output/recordings/`, `output/wandb/`, and `output/tmp/`.
- Run `uv run python scripts/tools/migrate_artifacts.py` (or the console entry point `uv run robot-sf-migrate-artifacts`) after pulling to consolidate any legacy `results/`, `recordings/`, `htmlcov/`, or `coverage.json` paths.
- Enforce the policy locally and in CI with `uv run python scripts/tools/check_artifact_root.py`; the guard fails fast when new top-level artifacts appear.
- Override the artifact destination by exporting `ROBOT_SF_ARTIFACT_ROOT=/path/to/custom/output` before invoking scripts; the helpers and guard honor the override consistently.
- Canonical helpers in `robot_sf.common.artifact_paths` (e.g., `ensure_canonical_tree`) create the required layout for tests and tooling—prefer them over hard-coded paths.
- Need a guided walkthrough? Follow the [artifact policy quickstart](../specs/243-clean-output-dirs/quickstart.md) for migration, guard usage, and override examples end to end.

### Testing strategy (UNIFIED test suite)

**The project now uses a unified test suite** running both robot_sf and fast-pysf tests via a single command.

#### Unified Test Suite

```bash
# Run ALL tests (robot_sf + fast-pysf) - RECOMMENDED
uv run pytest -n auto  # Number of test is steadily increasing, ca. 1200

# Run fast unit tests only (excludes slow/integration)
uv run pytest -m "not slow" tests
  # Note: integration/perf-heavy directories are auto-marked as slow in tests/conftest.py.

# Run only robot_sf tests
uv run pytest tests

# Run only fast-pysf tests  
uv run pytest fast-pysf/tests  # → 12 tests

# Run with parallel execution (faster)
uv run pytest -n auto
```

#### Legacy / Specialized Test Suites

```bash
# 1. Main unit/integration tests (2-3 min) - NOW PART OF UNIFIED SUITE
uv run pytest -n auto tests  # → 881 tests

# Fast unit test pass (skip slow/integration)
uv run pytest -m "not slow" tests

# 2. GUI/display-dependent tests (headless mode)  
DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy uv run pytest tests/pygame

# 3. fast-pysf subtree tests - NOW PART OF UNIFIED SUITE
uv run pytest fast-pysf/tests  # → 12 tests (all passing with map fixtures)
```

**Note**: The unified test command (`uv run pytest`) automatically discovers and runs tests from both `tests/` and `fast-pysf/tests/` directories. Test count increased from ~43 (legacy documentation) to 893 tests after fast-pysf integration.

#### Test Significance Verification

**Before fixing or investigating test failures, verify the test's value and necessity.** Not all tests provide equal value; some may be outdated, overly brittle, or testing non-critical behavior.

**Evaluation Questions** (ask before investing fix effort):

1. **Core Feature Coverage**: Does this test verify a public contract?
   - Factory behavior, schema compliance, metric correctness, deterministic reproducibility
   - **If YES** → High priority, fix immediately

2. **User Impact**: Would failure in production affect users?
   - Incorrect metrics, broken benchmarks, environment crashes
   - **If YES** → High priority, fix immediately

3. **Regression Prevention**: Does it catch known past bugs?
   - Validates recent fixes, prevents known failure modes
   - **If YES** → Medium priority, fix within sprint

4. **Edge Case vs. Common Path**: Does it test rare scenarios?
   - Low real-world occurrence, no documented incidents
   - **If YES and no incidents** → Low priority, consider archiving

5. **Brittleness**: Does it fail frequently without indicating real bugs?
   - Timing issues, display dependencies, environmental flakiness
   - **If YES** → Candidate for refactoring or removal

6. **Redundancy**: Is the same behavior tested elsewhere?
   - Identical logic covered by unit and integration tests
   - **If YES** → Consider consolidation

**Decision Actions**:
- **High Priority** (Core/User Impact): Fix immediately, these protect critical invariants
- **Medium Priority** (Regression/Important Edges): Fix or update within sprint; document if deferred
- **Low Priority** (Rare edges, redundant): Consider archiving; reassess value before fix effort
- **Flaky/Brittle**: Stabilize with retries/mocks if valuable; otherwise remove with documented rationale

**Maintenance Discipline**:
- Removing tests requires documented reason in commit message
- Deferred low-priority failures need tracking issue with "test-debt" label
- New tests MUST include docstring stating: (1) what contract/behavior is verified, (2) why it matters
- Quarterly audit: review tests by runtime and failure frequency; challenge bottom 10% on value

**Example**: If a test for an obscure edge case in trajectory smoothing fails but:
- No user has ever reported this scenario
- The edge case requires artificial setup unlikely in real usage
- Core smoothing is covered by other tests

→ Consider documenting the edge case in code comments and archiving the test, rather than spending hours debugging environmental setup issues.

### Coverage workflow (explicit opt-in)

Coverage collection is no longer enabled by default. Run tests normally for fast execution,
and enable coverage explicitly when needed.

The test harness sets the ``ROBOT_SF_ARTIFACT_ROOT`` environment variable so that
example scripts and helpers write into a temporary directory instead of the
repository tree. This keeps the canonical ``output/`` hierarchy clean while
preserving normal example behavior.

Try to increase the test coverage over time by adding tests when touching code. See the must-have checklist below for guidance.

#### Quick start
```bash
# Run tests (no coverage by default)
uv run pytest tests

# Run tests with coverage (CI and explicit opt-in local runs)
ROBOT_SF_PYTEST_COVERAGE=1 scripts/dev/run_tests_parallel.sh tests

# Run a focused local check and discard generated coverage output after success
scripts/dev/run_focused_tests.sh tests/test_force_flags.py -q

# View HTML report (preferred helper)
uv run python scripts/coverage/open_coverage_report.py

# Manual fallback
open output/coverage/htmlcov/index.html

# Or use VS Code task: "Run Tests with Coverage" → "Open Coverage Report"
```

#### What gets measured

- **Included**: All code in `robot_sf/` package
- **Excluded**: Tests, examples, scripts, `fast-pysf/` subtree
- **Output formats**: 
  - Terminal summary (printed after test run)
  - HTML report (`output/coverage/htmlcov/index.html` - interactive, detailed)
  - JSON data (`output/coverage/coverage.json` - for tooling)

#### Understanding coverage output

```
Name                                    Stmts   Miss  Cover   Missing
---------------------------------------------------------------------
robot_sf/gym_env/environment_factory.py   150     15  90.00%  42-45, 89-92
robot_sf/sim/simulator.py                 200     50  75.00%  10-20, 150-180
---------------------------------------------------------------------
TOTAL                                   10605    876  91.73%
```

- **Stmts**: Total executable lines
- **Miss**: Uncovered lines
- **Cover**: Percentage covered
- **Missing**: Line numbers not executed by tests

#### Coverage configuration

Configured in `pyproject.toml`:
- `[tool.coverage.run]` — collection settings (source, omit patterns, parallel support)
- `[tool.coverage.report]` — report formatting (precision, exclusions)
- `scripts/dev/run_tests_parallel.sh` — explicit pytest coverage opt-in for local wrapper and CI
  runs

No changes needed for normal development — default pytest runs skip coverage output for faster
feedback, while CI and explicit wrapper opt-in still generate reports.

#### Advanced usage
```bash
# Run with parallel workers (faster local feedback)
uv run pytest tests -n auto

# Run with parallel workers and explicit coverage collection
ROBOT_SF_PYTEST_COVERAGE=1 scripts/dev/run_tests_parallel.sh tests

# Run specific test file with coverage
ROBOT_SF_PYTEST_COVERAGE=1 scripts/dev/run_tests_parallel.sh tests/test_gymnasium_env_contracts.py -v
# Run specific test file without coverage
uv run pytest tests/test_gymnasium_env_contracts.py -v

# View coverage data programmatically
python -c "import json; print(json.load(open('output/coverage/coverage.json'))['totals'])"
```

#### Known limitation: focused `--cov` with Torch

Passing `--cov` directly to `uv run pytest` on a focused test file can trigger a
`RuntimeError: function '_has_torch_function' already has a docstring` from `torch/overrides.py`
when pytest-cov's trace hook causes a Torch C-extension to be partially re-initialised.
This was fixed in `robot_sf/telemetry/tensorboard_adapter.py` (#5101) but the underlying
pytest-cov+torch incompatibility persists on CPython 3.13 + torch ≥ 2.10.

Use the canonical wrapper for any coverage run that imports Torch:

```bash
# Correct: wrapper sets up coverage in a Torch-safe import order
ROBOT_SF_PYTEST_COVERAGE=1 scripts/dev/run_tests_parallel.sh tests/test_batched_lidar_kernel.py -v

# Incorrect: direct --cov flag may crash conftest collection
# uv run pytest tests/test_batched_lidar_kernel.py --cov=robot_sf.sensor.range_sensor
```

If you need focused changed-line coverage during development, run the test without `--cov` first
to confirm it passes, then use the wrapper for the coverage snapshot.

For coverage gap analysis, trend tracking, and CI integration, see `docs/coverage_guide.md` (created as part of US2/US3).

### Must-have checklist

- [ ] Use factory env creators; do not instantiate env classes directly.
- [ ] Set config via `robot_sf.gym_env.unified_config` before env creation; avoid ad‑hoc kwargs.
- [ ] Keep lib code print-free; use logging from loguru for info and warnings.
- [ ] Run VS Code Tasks: Install Dependencies, Ruff: Format and Fix, Check Code Quality (Ruff +
      advisory ty), Type Check (advisory), Run Tests.
- [ ] Add a test or smoke (e.g., env reset/step) when you change public behavior.
- [ ] For GUI-dependent tests, set headless env vars; avoid flaky display usage in CI.
- [ ] Treat `fast-pysf/` as part of the repository, changes can be made.
- [ ] Put new demos under `examples/` and new runners under `scripts/`.
- [ ] Whenever a demo is possible, add one.

### Optional backlog (track but don’t block)

- [ ] Tighten type hints for new public APIs; migrate call sites gradually.
- [ ] Add programmatic benchmark examples and extend baseline coverage.
- [ ] Update or add docs under `docs/` for new components; include diagrams when useful.
- [ ] Add performance smoke (steps/sec) when touching hot paths.
- [ ] Add proper docstrings to comply with pydoclint and pydocstyle

### Quick links

- Environment overview: `docs/ENVIRONMENT.md`
- Simulation view: `docs/SIM_VIEW.md`
- Refactoring and architecture notes: `docs/refactoring/`
- SNQI tools and metrics: `docs/snqi-weight-tools/README.md`
- Data analysis helpers: `docs/DATA_ANALYSIS.md`
- Contributor onboarding / repo structure: `AGENTS.md`

### Executive summary

- **Architecture**: Social navigation RL framework with Gymnasium environments, SocialForce pedestrian simulation via `fast-pysf` subtree, StableBaselines3 training pipeline, and optional RLlib DreamerV3 workflow
- **Core pattern**: Factory-based environment creation (`make_robot_env()` etc.) — never instantiate environments directly
- **Dependencies**: `fast-pysf` git subtree for pedestrian physics (automatically included after clone, see [Subtree Migration Guide](./SUBTREE_MIGRATION.md))
- **Toolchain**: uv + Ruff + ty + pytest with VS Code tasks; run quality gates before pushing
- **Testing**: Unit tests in `tests/`, GUI-dependent tests in `tests/pygame/` (with headless env vars), integration tests for smoke/performance validation
- **Documentation**: Comprehensive docs under `docs/` with design principles, architecture, usage, and migration notes
  - Development notes: `docs/dev/*`

### Logging & Observability (Principle XII)

The canonical logging facade is **Loguru**. Library code (anything under `robot_sf/` or wrappers over `fast-pysf`) must not use bare `print()` for informational or warning messages. Acceptable `print()` exceptions: (1) short CLI entry scripts in `scripts/` or `examples/` where stdout is the UX, (2) early bootstrap failures before logging configuration, (3) tests explicitly asserting stdout content. Migration of stray prints to `from loguru import logger` with `logger.info|warning|error` is treated as maintenance (PATCH) unless it changes user‑visible contract output.

Guidelines:
 - Prefer structured context (e.g., `logger.info("Reset complete seed={seed} scenario={sid}")`).
 - Avoid inside per‑timestep loops; aggregate and log at episode boundaries to protect performance budgets.
 - Use WARNING for degraded but continuing states (e.g., zero frames when recording requested), ERROR for aborting conditions, CRITICAL for irreversible state corruption.
 - Tests may temporarily raise log level to DEBUG for diagnosing flakes but should reset after.
 - Provide a toggle (env var or parameter) when adding verbose debug logging to hot paths.

Rationale: Centralized logging enables deterministic capture/suppression in benchmarks, simplifies CI noise control, and aligns with Constitution Principle XII (Preferred Logging & Observability).

### Code quality standards

- Clear, intent‑revealing names; small, cohesive functions; robust error handling.
- Follow existing style; document non‑obvious choices with comments/docstrings.
- Add helpful comments to quickly understand the code’s purpose and logic.
- Avoid duplication; prefer composition and reuse.
- Keep public behavior backward‑compatible unless explicitly stated.
- Write comprehensive unit tests for new features and bug fixes (GUI tests in `tests/pygame/`).
- **Verify test value before investing fix effort** (see Test Significance Verification in Testing Strategy section).
- Math vs numpy: use `math` for scalar ops/constants, `numpy` for vectorized/array ops, and avoid mixing within a single expression.

### Design decisions

- Favor readability and maintainability over micro‑optimizations.
- Use type hints for all public functions and methods; prefer `typing` over `Any`.
- Use exceptions for error handling; avoid silent failures.

#### CLI vs programmatic use

- This project prioritizes traceability and reproducibility of benchmarks. Prefer generating script- and config-driven workflows over ad-hoc command lines or inline parameter tweaks.
- Do not focus on the cli directly; prefer programmatic use and factory functions.
- The CLI is not important; prefer programmatic use and factory functions.
- Use logging for non‑error informational messages; avoid print statements except in CLI entry points.

- Configs: configs/<area>/<name>.yaml (single source of truth for all hyperparameters, seeds, envs).
-	Scripts: scripts/<task>_<runner>.py (read config path, set up run dirs, log metadata, call library code).
-	Runs/outputs/benchmarks: output/benchmarks/<timestamp>_<shortname>/ (store config.yaml, git_meta.json, logs, metrics, artifacts).
-	Deterministic seed in both config and code

### Code reviews

- All changes must be reviewed by at least one other team member.
- Reviewers should check for correctness, style, test coverage, and documentation.
- Use GitHub’s review tools to leave comments and approve changes.

#### One-real-path-test rule

For serialization, subprocess, GPU-isolation, artifact-promotion, and CLI-handoff code, keep at
least one test on the same serialization and invocation path production uses. Do not manually
pre-transform a fixture before the boundary: it can bypass the exact conversion or dispatch defect
the test is meant to catch. The subprocess-isolation regression test at
`tests/benchmark/test_camera_ready_subprocess_isolation.py` is the reference pattern.

For a shared-helper migration, record a per-call-site contract table in the PR and test every
applicable row: return type/top-level schema, missing and malformed input behavior (including
caller exit code), minimal import footprint, eager versus streaming reads, `path:line` context, and
output ordering. Mark genuinely inapplicable rows explicitly.

#### Docstrings

- Every module, function, class, and method should have a docstring.
- Docstrings should use triple double quotes (""").
- The first line should be a short summary of the object’s purpose, starting with a capital letter and ending with a period.
- If more detail is needed, leave a blank line after the summary, then continue with a longer description.
- For functions/methods: document parameters, return values, exceptions raised, and side effects.
- Private/internal code should also have docstrings explaining their purpose for easier maintainability.
- Follow the pydocstyle convention specified in `pyproject.toml`.

### Clarify questions (with options)
- In case of ambiguity or uncertainty about requirements, always ask clarifying questions before starting implementation. Provide multiple-choice options to facilitate quick decision-making. Group questions by scope, interfaces, data handling, UX, and performance.
- Before implementing, confirm requirements with targeted questions.
- Prefer multiple‑choice options to speed decisions; group by scope, interfaces, data, UX, performance.
- Add arguments to the options for easy decision-making.
- If answers are unknown, propose sensible defaults and proceed (don't block on non‑essentials).

Examples (copy‑ready):
- Scope: Is the metric per episode or a per‑timestep aggregate?
- Interfaces: Return shape `dict[str, float]` or a dataclass?
- Data: How to handle NaN/missing — drop, impute, or error?
- UX: Any hotkey conflicts with existing controls; prefer `,` and `.`?
- Performance: Target budget for feature X (ms/frame)?

### Problem‑solving approach
- Break problems into smaller tasks; research prior art and patterns.
- Clearly prioritize must‑haves vs nice‑to‑haves.
- Consider system‑wide impact, edge cases, error handling, and failure modes.
- Document architectural decisions and trade‑offs.

### Tooling and tasks (uv, Ruff, pytest, ty, VS Code)
- Dependencies/runtime: uv
  - Install/resolve: VS Code task “Install Dependencies” (uv sync)
  - Run: `uv run <cmd>` for any Python command
  - Add deps: `uv add <package>` (or edit `pyproject.toml` and sync)
- Lint/format: Ruff
  - VS Code task “Ruff: Format and Fix” (keeps repo ruff‑clean with the expanded rule set; document exceptions with comments)
- Type checking: ty
  - VS Code task "Type Check (advisory)" (`uvx ty check . --exit-zero`; reports findings while
    exiting zero for current compatibility)
  - Type findings are useful quality signals and should be fixed when practical, especially in
    substantially touched files or stable contracts such as public interfaces, benchmark schemas,
    planner contracts, config parsing, map definitions, artifact metadata, and CLI boundaries.
  - PRs are not blocked solely because the advisory `ty` phase reports findings. Reviewers may
    still request typing fixes when findings affect changed code or stable contracts.
  - A fail-closed typecheck gate, changed-files ratchet, or baseline-reduction workflow must be
    proposed separately before becoming a merge requirement.
- Tests: pytest
  - VS Code task “Run Tests” (default suite)
  - “Run Tests (Show All Warnings)” for diagnostics
  - “Run Tests (GUI)” for display‑dependent tests (headless via environment vars)
  - VS Code task “PR Ready Check” runs Ruff fix/format, full tests (incl. slow), changed‑files coverage gate, diff‑only TODO docstring warnings, and the TODO-docstring backlog ratchet
- Code quality checks: VS Code task “Check Code Quality (Ruff + advisory ty)”
- Diagrams: VS Code task “Generate UML”

Quality gates to run locally before pushing:
1) Install Dependencies → 2) Ruff: Format and Fix → 3) Check Code Quality (Ruff + advisory ty) → 4) Type Check (advisory) → 5) Run Tests

Shortcuts (optional shell):
- Break down complex problems into smaller, manageable tasks
- Research existing solutions and patterns before implementing new approaches
- Use existing libraries and frameworks when possible to avoid reinventing the wheel
- Consider the impact of changes on the entire system, not just the immediate problem
- Document architectural decisions and trade-offs made during implementation
- Think about edge cases, error handling, and potential failure modes

## Documentation Standards

### Technical Documentation

- Create comprehensive documentation for all significant changes and new features
- Save documentation files in the `docs/` directory using a clear folder structure
- Each major feature or issue should have its own subfolder named in kebab-case
  - Format: `docs/dev/issues/42-fix-button-alignment/` or `docs/dev/issuesfeature-name/`
- Use descriptive README.md files as the main documentation entry point for each folder

### Docs Folder Structure

Here’s a concise map of the docs folder to help you find the right guidance quickly. Each folder should include a README.md for context, links, and references.

#### Top-level guides (entry points)
- README.md — Main docs landing page.
- dev_guide.md — Primary development reference (setup, workflow, testing, CI).
- `ENVIRONMENT.md` — Environment overview and usage.
- `SIM_VIEW.md` — Simulation view/UI notes.
- `UV_MIGRATION.md` — Migration notes to uv.
- Topic-specific guides:
  - `DATA_ANALYSIS.md`, `trajectory_visualization.md`, `SVG_MAP_EDITOR.md`, `fast_pysf_wrapper.md`, `pyreverse.md`, `curvature_metric.md`, `snqi_weight_cli_updates.md`.

#### Focused subfolders
- `2x-speed-vissimstate-fix/`
  - README.md — Notes and outcome for the VissimState 2x speed fix.
- `baselines/`
  - `social_force.md` — Baseline Social Force documentation.
- `docs/dev/` — In-progress/engineering docs and design notes
- `extract-pedestrian-action-helper/`
  - README.md — Helper tool documentation.
- `img/` — Images used across docs
- `ped_metrics/` Pedestrian metrics documentation and analysis notes.
- `refactoring/` Migration/architecture reports and plans
- `snqi-weight-tools/` — SNQI weight tooling user docs and schema
- `templates/` Template for new design docs.
- `video/` Demo animations for docs.

### Documentation Content Requirements

Documentation should include:
- **Problem Statement**: Clear description of the issue being addressed
- **Solution Overview**: High-level approach and architectural decisions
- **Implementation Details**: Code examples, API changes, and technical specifics
- **Impact Analysis**: What systems/users are affected and how
- **Testing Strategy**: How the changes were validated
- **Future Considerations**: Potential improvements or known limitations
- **Related Links**: References to GitHub issues, pull requests, or external resources

### Documentation Best Practices

- Use proper markdown formatting with clear headings and structure
- Include code examples with syntax highlighting
- Add diagrams or screenshots when they improve understanding
  - Mermaid diagrams are welcome and encouraged for visualizing workflows, architecture, and relationships
- Write for future developers who may be unfamiliar with the context
- Keep documentation up-to-date as code evolves
- Use consistent formatting and follow markdown linting standards
- Prefer GitHub-flavored Markdown (GFM) conventions so docs render correctly on GitHub
- Write issue references as `Issue #123` in prose, lists, and tables instead of starting a
  Markdown line with a bare `#123` token.
- Avoid duplications. Link to existing documentation when relevant.
- Always provide README.md files in new documentation folders for overview and reference.
- When the document is longer than 50 lines, create a table of contents at the top for easy navigation. Ideally, use `markdown.extension.toc.create` to *Markdown All in One: Create Table of Contents*.

#### Visualizations and Reports

- Use visualizations to illustrate complex concepts or data flows
- Include performance reports or benchmarks when relevant
- Ensure all visual assets are stored in the `docs/img/`, `docs/figures/` or `docs/video/` directories for easy access and consistency
- Generate figures using code when possible to ensure reproducibility
- Figures should be exported in high-quality vector formats (e.g., SVG, PDF) for clarity

#### Figure and Visualization Guidelines
All figures must be **reproducible from code** and directly **integratable into LaTeX documents**:
- **Output format**
  - Always export **vector PDFs** (`.pdf`) for inclusion in LaTeX.
  - Optionally export `.png` (300 dpi) for slides/presentations.
- **Reproducibility**
  - Each figure = one tracked script or CLI command in `robot_sf/benchmark/figures/`,
    `scripts/generate_figures.py`, or the `robot_sf_bench` CLI.
  - The generator must read data, generate the plot, and save into `docs/figures/`.
  - No manual edits in Illustrator, Inkscape, etc.
  - Clear and unique output filenames: `fig-<short-description>.pdf`.
- **Version control**
  - Scripts and generated figures go into version control.
  - Data files (if any) go into `output/figures/` (respecting the canonical artifact root).
- **Consistent style**
  - Use Matplotlib with predefined `rcParams`:
    - `savefig.bbox = "tight"`
    - `pdf.fonttype = 42`
    - font sizes: 9 pt labels, 8 pt ticks/legend
    - line width ~1.2–1.6 pt
  - Axis labels and math should use LaTeX syntax: `r"$\sin(x)$"`.
- **Figure sizing**
  - Provide helper function for resizing
  - Default: single-column width (`fraction=1.0`).
- **File locations**
  - Figures go into `docs/figures/` (tracked).
  - Data exports (if used) into `output/figures/`.


## CI/CD expectations
- Tests: `uv run pytest tests`
- Lint: `uv run ruff check .` and `uv run ruff format --check .`
- The pipeline mirrors the local quality gates. Ensure green locally first.
- After merging fresh `origin/main`, or after CI reports a formatting failure outside the files you
  intentionally touched, run the repo-wide lightweight lint/format gate that CI uses. A changed-file
  format check can be stale when the shared baseline moved.

CI mapping to local tasks and CLI:
- `fast-feedback` job → `scripts/dev/ci_driver.sh lint typecheck test`
- `smoke-artifacts` job → `scripts/dev/ci_driver.sh smoke artifact-policy`
- aggregate `ci` job → waits for both split jobs so existing required-check naming remains stable
- local full equivalent → `scripts/dev/run_ci_local.sh`

Workflow location: `.github/workflows/ci.yml`.

### Red-main merge hold

When `main` CI is red, do not merge unrelated PRs onto it. A merge that lands
while the required check is already failing hides its own new breakage under
the existing red, so recovery cost compounds with every merge in the window
(three such incidents on 2026-07-11/12). The deterministic green/red signal is:

```bash
uv run python scripts/dev/main_ci_is_green.py   # exit 0 green, 1 not-green
```

It decides from the most recent **completed** CI run on `main` — an in-progress
run never counts as green or red. Only PRs that fix the breakage (title-prefixed
`fix(ci): unbreak main` or labeled `unbreak-main`) may merge while red; they are
the cure and must land. Reviewing a PR while main is red is fine — only the
merge is held.

## CI Performance Monitoring
The CI pipeline separates fast feedback from the heavier smoke/artifact tail:

- `fast-feedback` runs lint, advisory type checking, and the main pytest suite.
- `smoke-artifacts` runs validation smoke checks, uploads benchmark/recording artifacts, and enforces
  the artifact-root policy.
- Both jobs call the canonical `scripts/dev/ci_driver.sh` phases instead of duplicating validation
  semantics in workflow YAML.
- System packages are installed through the supported `apt-get` path in one update/install step per
  job; the workflow does not download `apt-fast` at runtime.

The smoke lane still includes performance monitoring and regression checks through current,
committed entry points:

**Workflow Integration**:
- Fast lint/typecheck/test feedback is reported before smoke/artifact completion.
- `smoke-artifacts` uploads map verification, benchmark, recording, and cold/warm performance
  artifacts from the canonical `output/` tree.
- Cold/warm regression smoke is driven by
  `uv run python -m robot_sf.benchmark.perf_cold_warm` through
  `scripts/dev/ci_driver.sh smoke`.
  Pull requests run the check in advisory mode; `main` and `workflow_dispatch` runs use the
  stricter regression gate.
- Startup/reset performance is measured by `scripts/validation/performance_smoke_test.py` during
  strict smoke runs, and telemetry smoke/perf coverage is exercised by
  `scripts/validation/run_examples_smoke.py --perf-tests-only`.

**Local Testing**:
- Use `scripts/dev/run_ci_local.sh` for the canonical local CI-equivalent path.
- Use `scripts/dev/run_ci_local.sh --no-setup <phase> ...` for repeat local phase runs after the
  worktree has already been synced.
- Use `scripts/dev/ci_driver.sh <phase>` for narrower local phases such as `lint`, `typecheck`,
  `test`, `smoke`, or `artifact-policy`.
- Use `uv run python scripts/dev/ci_timing_summary.py --run-id <github-actions-run-id> --top 10`
  to inspect GitHub-hosted CI queue time, job duration, and slowest-step timing from a completed
  run.
- Use `uv run python scripts/dev/complexity_runtime_baseline.py --top 10 robot_sf scripts tests`
  when a refactor needs a local snapshot of large modules, long functions, or captured pytest
  duration rows.
- Optional local GitHub Actions execution with `act` is not currently a supported repository
  workflow. Issue #1308 evaluated this path on 2026-05-18 and did not adopt it because Docker was
  available but `act` was not installed, so no workflow job could be proven locally. See
  [the evaluation note](context/issue_1308_act_local_workflow_evaluation.md).
- `gh act` is now installed on one local machine and Issue #1342 proved non-interactive `--dryrun`
  workflow graph validation, but not a real local workflow execution. Keep using
  `scripts/dev/run_ci_local.sh` and `BASE_REF=origin/main scripts/dev/pr_ready_check.sh` as the
  supported local proof paths until a real narrow `gh act` target is recorded.

**Performance Breach Handling**:
- Cold/warm PR smoke uses advisory thresholds by default; `main` and `workflow_dispatch` runs
  enforce the stricter regression gate.
- Startup/reset smoke supports soft and hard thresholds through
  `ROBOT_SF_PERF_CREATION_SOFT`, `ROBOT_SF_PERF_CREATION_HARD`,
  `ROBOT_SF_PERF_RESET_SOFT`, `ROBOT_SF_PERF_RESET_HARD`, and
  `ROBOT_SF_PERF_ENFORCE=1`.

## Validation scenarios and performance
### Validation scenarios (run after changes)
```bash
./scripts/validation/test_basic_environment.sh
./scripts/validation/test_model_prediction.sh
./scripts/validation/test_complete_simulation.sh
uv run python scripts/validation/run_examples_smoke.py --dry-run
uv run python scripts/validation/run_examples_smoke.py --perf-tests-only
uv run python scripts/validation/run_examples_smoke.py
uv run python scripts/validation/svg_inspect.py maps/svg_maps --pattern "classic_*.svg" --strict warning
uv run python scripts/tools/check_artifact_root.py

# Performance baseline validation
DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy \
  uv run python scripts/validation/performance_smoke_test.py
```
Success criteria:
- Basic environment: exits 0; no exceptions.
- Model prediction: exits 0; logs model load and inference without errors.
- Complete simulation: exits 0; simulation runs to completion without errors.
- Example smoke harness: exits 0; all `ci_enabled` examples pass and archived entries are reported as skipped via manifest metadata.
- SVG inspection: exits 0 in strict mode only if no warning/error findings are detected at the selected threshold.
- Artifact guard: exits 0; repository root remains clean with all artifacts under `output/` (mirror of the CI enforcement step).
- Performance smoke test: exits 0; meets baseline performance targets (see `docs/performance_notes.md`).
  - Threshold logic now includes soft vs hard tiers with environment overrides. Soft breaches on CI default to WARN (exit 0) unless `ROBOT_SF_PERF_ENFORCE=1`.
    - Environment variables:
      - `ROBOT_SF_PERF_CREATION_SOFT` (default 3.0)
      - `ROBOT_SF_PERF_CREATION_HARD` (default 8.0)
      - `ROBOT_SF_PERF_RESET_SOFT` (default 0.50 resets/sec)
      - `ROBOT_SF_PERF_RESET_HARD` (default 0.20 resets/sec)
  - `ROBOT_SF_PERF_ENFORCE=1` to fail on soft (and hard) breaches (use locally for strict tuning).
  - (Advanced) `ROBOT_SF_PERF_SOFT` / `ROBOT_SF_PERF_HARD` may be set to numeric seconds to temporarily override thresholds (intended only for internal testing of enforcement logic; not part of the stable public interface).
    - Hard threshold breaches always FAIL.

  ### Example maintenance workflow

  1. **Validate catalog** – `uv run python scripts/validation/validate_examples_manifest.py` ensures the manifest enumerates every script and that docstrings stay aligned with summaries.
  2. **Review planned changes** – `uv run python scripts/validation/run_examples_smoke.py --dry-run` prints the `ci_enabled` set before executing pytest, making it easy to confirm archive decisions.
  3. **Run tracker/perf gate** – `uv run python scripts/validation/run_examples_smoke.py --perf-tests-only --perf-num-resets 2` exercises the imitation pipeline tracker smoke plus telemetry perf wrapper without re-running the entire pytest suite.
  4. **Execute smoke harness** – `uv run python scripts/validation/run_examples_smoke.py` runs all active examples headlessly; pytest fixtures already configure pygame for a dummy display.
  5. **Archive responsibly** – whenever a script moves into `examples/_archived/`, update `examples/_archived/README.md`, set `ci_enabled: false` with a `ci_reason`, and point the module docstring at the maintained replacement.

### Run tracker & history CLI

- Enable the tracker with `--enable-tracker` on `examples/advanced/16_imitation_learning_pipeline.py`. A background guard now snapshots manifests roughly every five seconds and traps `SIGINT`/`SIGTERM`, so failed or cancelled runs emit a `failed` manifest entry automatically.
- Inspect live progress with `status` or `watch`:
  ```bash
  uv run python scripts/tools/run_tracker_cli.py status <run_id>
  uv run python scripts/tools/run_tracker_cli.py watch <run_id> --interval 1.0
  ```
  Both commands read the latest manifest snapshot and show current step, elapsed time, ETA, and the last completed step.
- Use `list` to review prior runs (defaults to the most recent 20). Helpful filters:
  - `--status pending|running|completed|failed|cancelled`
  - `--since 2025-01-15T00:00:00+00:00` (UTC ISO timestamps)
  - `--format table|json` for human vs machine-readable output
- `summary` (aliased as `show`) prints per-run breakdowns, with `--format text|json|markdown`. Markdown output intentionally mirrors the exported summaries so docs/changelogs can embed them verbatim.
- `export` writes Markdown or JSON summaries directly to disk:
  ```bash
  uv run python scripts/tools/run_tracker_cli.py export <run_id> \
    --format markdown \
    --output output/run-tracker/summaries/<run_id>.md
  ```
  Exports include per-step durations, artifact paths, and any failure context produced by the guard.
- Mirror telemetry to TensorBoard when you need dashboards:
  ```bash
  uv run python scripts/tools/run_tracker_cli.py enable-tensorboard <run_id> --logdir output/run-tracker/tb/<run_id>
  uv run tensorboard --logdir output/run-tracker/tb
  ```
  The CLI replays `telemetry.jsonl` into SummaryWriter so you can inspect CPU/GPU trends without touching the canonical JSON artifacts.
- Run the performance smoke wrapper straight from the CLI instead of calling scripts manually:
  ```bash
  uv run python scripts/tools/run_tracker_cli.py perf-tests \
    --scenario configs/validation/minimal.yaml \
    --output output/run-tracker/perf-tests/latest \
    --num-resets 5
  ```
  Results are persisted in `perf_test_results.json` with pass/soft-breach/fail classification plus any recommendations triggered by the telemetry rules.
- Because the guard writes manifests on a timer and on signals, partial runs survive restarts—`list`/`show` will always have at most a five-second gap between what ran and what was recorded.

### Performance benchmarking (optional)
```bash
# Run maintained performance smoke when performance impact is suspected
DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy \
uv run python scripts/validation/performance_smoke_test.py
```

### Performance expectations
- Environment creation: < 1 second
- Model loading: 1–5 seconds
- Simulation performance: ~22 steps/second (~45ms/step)
- Build time: 2–3 minutes (first time)
- Test suite: 2–3 minutes (≈170 tests)

---

## Benchmark runner: parallel workers and resume

The benchmark runner supports process-based parallel execution and safe resume.

- Parallelism: Use multiple workers to run independent episodes concurrently.
- Resume: Skips episodes that are already present in the output JSONL.

Key points
- Parent-only writes: only the parent process writes JSONL lines to avoid corruption.
- Episode identity: jobs are identified deterministically from scenario params and seed; existing episodes are skipped when resume is enabled.
- macOS: workers > 1 uses the spawn start method; ensure worker code is importable/picklable and defined at module top level (no lambdas/closures).

CLI usage
- Run a batch with parallel workers and default resume behavior:
  - robot_sf_bench run --matrix configs/baselines/example_matrix.yaml --out output/benchmarks/episodes.jsonl --workers 4
- Force recomputation (disable resume):
  - robot_sf_bench run --matrix configs/baselines/example_matrix.yaml --out output/benchmarks/episodes.jsonl --workers 4 --no-resume
- Opt in to schema-backed pedestrian-impact reductions:
  - robot_sf_bench run --matrix configs/scenarios/planner_sanity_matrix_v1.yaml --out output/benchmarks/ped_impact/episodes.jsonl --experimental-ped-impact
- Baseline computation also accepts the same flags:
  - robot_sf_bench baseline --episodes output/benchmarks/episodes.jsonl --output output/benchmarks/baseline.jsonl --workers 4

Programmatic usage
- Prefer factory functions and programmatic APIs in library code:
  - from robot_sf.benchmark.runner import run_batch
  - from robot_sf.benchmark import baseline_stats
  - run_batch(scenarios, out_path=..., schema_path=..., workers=4, resume=True)
  - baseline_stats.run_and_compute_baseline(episodes_path=..., out_path=..., workers=4, resume=True)

Notes
- Default behavior is resume=True for programmatic APIs and CLI (omit --no-resume to keep it enabled).
- When resuming, open files in append mode if you want to keep existing lines; the runner will not duplicate episodes.
- On macOS spawn, module-level top-level functions are required for worker processes to import successfully.
- Resume accelerator: The runner writes a small sidecar manifest (episodes.jsonl.manifest.json) caching episode ids and file stat. On subsequent runs, resume uses this manifest when valid and transparently falls back to scanning the JSONL if the sidecar is stale or missing. No user action required.

## Aggregation and Confidence Intervals

Once you have a JSONL of episodes, you can aggregate metrics by group and optionally attach bootstrap confidence intervals.

CLI usage

- Aggregate without CIs (default):
  - robot_sf_bench aggregate --in output/benchmarks/episodes.jsonl --out output/benchmarks/summary.json
- Aggregate with CIs (enable with >0 samples):
  - robot_sf_bench aggregate --in output/benchmarks/episodes.jsonl --out output/benchmarks/summary_ci.json --bootstrap-samples 1000 --bootstrap-confidence 0.95 --bootstrap-seed 123

Options

- --group-by: Dotted path for grouping (default: scenario_params.algo)
- --fallback-group-by: Used when group-by is missing (default: scenario_id)
- --bootstrap-samples: Number of bootstrap resamples; 0 disables CI keys
- --bootstrap-confidence: Confidence level, e.g., 0.90, 0.95
- --bootstrap-seed: Optional deterministic seed for CIs
- --snqi-weights/--snqi-baseline: Recompute metrics.snqi during aggregation
- Pedestrian-impact records: Aggregation flattens `metrics.pedestrian_impact.canonical_reductions`
  into `ped_impact_*` reduction columns, then applies the same mean/median/p95 summaries.

Output format

- For each group and metric, the aggregator returns mean, median, p95.
- When CIs are enabled, additional keys are included: mean_ci, median_ci, p95_ci as [low, high].
- When CIs are enabled and at least two groups share `(scenario_id, seed)` episode identities
  (`seed_index` is accepted as a fallback), an additive top-level `pairwise_contrasts` block reports
  paired bootstrap mean deltas, confidence intervals, two-sided bootstrap sign p-values,
  Holm-adjusted p-values, and paired Cohen's dz effect sizes. Deltas are `right_minus_left` for
  comparison keys like `A__vs__B`.
- Holm correction is applied within the current aggregate family (`family="all"`) separately for
  each metric. For scenario-family-specific correction, filter or split the input records by family
  before aggregation.

## Planner Inclusion Gate

Run the planner inclusion check when a planner is being considered for promotion from
experimental/testing-only status into a promoted benchmark set:

```bash
uv run robot_sf_bench planner-inclusion-check \
  --algo orca \
  --matrix configs/scenarios/planner_sanity_matrix_v1.yaml \
  --output-dir output/planner_inclusion/orca
```

The command writes `<algo>_episodes.jsonl` plus `<algo>_inclusion_report.json`. The report is a
review artifact, not an automatic status update. It fails closed with explicit reasons for runner
or schema failures, NaN/infinite aggregates, slow runtime, too few episodes, low success rate, or
excess collision rate.

Programmatic usage

```python
from robot_sf.benchmark.aggregate import read_jsonl, compute_aggregates_with_ci

records = read_jsonl("output/benchmarks/episodes.jsonl")
summary = compute_aggregates_with_ci(
    records,
    group_by="scenario_params.algo",
    fallback_group_by="scenario_id",
    bootstrap_samples=1000,
    bootstrap_confidence=0.95,
    bootstrap_seed=123,
)
```

## Training and examples
### Available demos
```bash
uv run python examples/quickstart/01_basic_robot.py
uv run python examples/quickstart/02_trained_model.py
uv run python examples/quickstart/03_custom_map.py
uv run python examples/advanced/06_pedestrian_env_factory.py
```

### Training scripts
```bash
uv run python scripts/training/train_ppo.py --config configs/training/ppo/expert_ppo_issue_576_br06_v3_15m_all_maps_randomized.yaml
uv run python scripts/training/launch_optuna_expert_ppo.py --config configs/training/ppo_imitation/optuna_expert_ppo.yaml
uv run python scripts/evaluate.py
```

### Imitation Learning Pipeline (PPO Pre-training)

The project supports accelerating PPO training via behavioral cloning pre-training from expert trajectories. This enables sample-efficient training by warm-starting agents with expert demonstrations.

Install the optional imitation stack before running BC pre-training:

```bash
uv sync --group imitation
```

Use `uv run --group imitation ...` for BC pre-training commands.

**Quick Overview:** Expert PPO Training → Trajectory Collection → BC Pre-training → PPO Fine-tuning → Comparison Analysis

**For complete documentation, see [Imitation Learning Pipeline Guide](./imitation_learning_pipeline.md)** which includes:
- Detailed step-by-step workflow
- Configuration file examples
- Validation and debugging tools
- Artifact locations and manifest tracking
- Sample-efficiency metrics (target: ≤70% of baseline timesteps)
- Troubleshooting and best practices

**Quick Start:**
```bash
# End-to-end wrapper (recommended for new users)
uv run python examples/advanced/16_imitation_learning_pipeline.py

# Or run individual steps manually:
uv run python scripts/training/train_ppo.py --config configs/training/ppo/expert_ppo_issue_576_br06_v3_15m_all_maps_randomized.yaml
uv run python scripts/training/collect_expert_trajectories.py --dataset-id expert_v1 --policy-id ppo_expert_v1 --episodes 200
uv run --group imitation python scripts/training/pretrain_from_expert.py --config configs/training/ppo_imitation/bc_pretrain.yaml
uv run python scripts/training/train_ppo_with_pretrained_policy.py --config configs/training/ppo_imitation/ppo_finetune.yaml
```

Set `--log-level DEBUG` if you need the full resolved-config dumps from the factory helpers (default is INFO to keep console noise down). Use `--backend <name>` to override the auto-selected simulator backend (defaults to the fastest available choice via `select_best_backend`). The end-to-end example auto-generates BC/ppo fine-tuning configs under `output/tmp/imitation_pipeline/`, so you only need to edit the YAML files when running the scripts manually.

BC pre-training configs default to `device: auto`, which lets Stable-Baselines3 and
`imitation` use available accelerators. Set `device: cpu` in the BC YAML when you need
a deterministic or resource-constrained CPU-only run.

**Also see:**
- End-to-end example: `examples/advanced/16_imitation_learning_pipeline.py`
- Detailed workflows: `specs/001-ppo-imitation-pretrain/quickstart.md`

### RLlib DreamerV3 (`drive_state` + `rays`)

RLlib DreamerV3 can be trained on the default non-image observation contract.

```bash
# Install RLlib optional dependency
uv sync --extra rllib

# Validate config
uv run --extra rllib python scripts/training/train_dreamerv3_rllib.py \
  --config configs/training/rllib_dreamerv3/drive_state_rays.yaml \
  --dry-run

# Run training
uv run --extra rllib python scripts/training/train_dreamerv3_rllib.py \
  --config configs/training/rllib_dreamerv3/drive_state_rays.yaml
```

The workflow uses deterministic flattening order (`drive_state`, `rays`) and can normalize
actions to `[-1,1]` for DreamerV3.
The launcher also pins Ray workers to the active interpreter and disables `uv run`
runtime-env propagation to avoid worker-side environment rebuilds.
See `docs/training/dreamerv3_rllib_drive_state_rays.md` for the Auxme launch/monitor/recovery runbook.

### Docker training (advanced)
```bash
# Build and run GPU training (requires NVIDIA Docker)
# NOTE: May fail in CI environments due to network restrictions
```

---

## Common issues and solutions
### Build issues
- uv not found → `curl -LsSf https://astral.sh/uv/install.sh | sh` (or use the package manager path in `docs/ENVIRONMENT.md`)
- ffmpeg missing → `sudo apt-get install -y ffmpeg`

### Runtime issues
- Import errors → ensure venv is activated: `source .venv/bin/activate`
- Display errors → run headless: `DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy`
- Model loading warnings → StableBaselines3 warnings about legacy Gym-trained models are expected (re-save models to clear them)
- Model compatibility → use newest models for best compatibility (e.g., `ppo_model_retrained_10m_2025-02-01.zip`)

## Migration notes
- The project uses uv for env/runner and a factory pattern for environment creation.
- See `docs/UV_MIGRATION.md` and `docs/refactoring/` for details.

---

## Helpful definitions and repository structure
### Helpful definitions
- uv: Fast Python package/dependency manager and runner (`uv sync`, `uv run`).
- Ruff: Python linter and formatter (run via "Ruff: Format and Fix" task).
- pytest: Testing framework (run via "Run Tests" tasks).
- VS Code tasks: Standardized workflows (install, lint, test, diagram).
- Quality gates: Minimal checks before pushing (install → lint/format → quality check → tests).

### Repository structure (key dirs)
- `robot_sf/` (source), `examples/`, `tests/`, `tests/pygame/`, `fast-pysf/` (subtree), `scripts/`, `model/`, `docs/`

---

## Definition of Done (DoD)
- Requirements clarified (with options/assumptions recorded).
- Design doc added/updated and linked (if non‑trivial).
- Code implemented with tests (unit/integration; GUI when needed).
- For research/benchmark/metric/paper-facing analysis-tool PRs: include one representative use on
  durable/versioned input (tracked config, model checkpoint, committed fixture, or versioned W&B
  artifact), or link a concrete follow-up issue that names the decision, claim boundary, or
  synthesis surface the tool will update. Local-only `output/` files are not durable proof unless
  promoted or represented by a tracked manifest. Small support helpers (formatters, CLI wrappers,
  quick diagnostics) that make no research/benchmark/metric/paper claim should state
  `NA - support helper` with the reason. Trace-panel generators, topology-score instrumentation,
  seed-sufficiency analysis, and why-report generation are research-facing examples that need first
  use or a concrete follow-up.
- Ruff clean and “Check Code Quality (Ruff + advisory ty)” reviewed locally.
- Advisory typecheck reviewed. Fix practical findings in touched files and stable contracts, and
  document any meaningful remaining findings in the PR when they affect the change.
- Docs updated (README in feature folder, diagrams if changed).
- Validation matched to risk per [maintainer_values.md](./maintainer_values.md): runtime, benchmark, metric, schema,
  model-provenance, and paper-facing changes need executable proof; low-risk docs/instruction
  changes use diff review, referenced path/link checks, and lightweight automated checks when
  available. State explicitly in the PR which heavier gates were skipped and why.
- Feature branch synced with latest `origin/main` before PR creation, then required validation
  scripts run and pass for the selected risk level.
- CI green (lint + tests) and PR opened with appropriate links.

## Templates

Use the following templates for specific tasks.

- [issue template](../.github/ISSUE_TEMPLATE/issue_default.md) - Agent-ready fallback for small executable tasks
- YAML issue forms for common backlog lanes:
  [research validation](../.github/ISSUE_TEMPLATE/research-validation.yml),
  [test debt](../.github/ISSUE_TEMPLATE/test-debt.yml),
  [blocked external artifact/runtime](../.github/ISSUE_TEMPLATE/blocked-external-artifact.yml),
  [execution run](../.github/ISSUE_TEMPLATE/execution-run.yml), and
  [epic](../.github/ISSUE_TEMPLATE/epic.yml). These supplement the Markdown templates; they do
  not replace the fallback template.
- [issue creator skill](../.agents/skills/gh-issue-creator/SKILL.md) - Turn vague prompts into structured issues
- [issue template auditor skill](../.agents/skills/gh-issue-template-auditor/SKILL.md) - Review and repair underspecified issues
- [priority assessor skill](../.agents/skills/gh-issue-priority-assessor/SKILL.md) - Review Project #5 priority inputs against the rubric and explain plausibility
- [PR opener skill](../.agents/skills/gh-pr-opener/SKILL.md) - Open ready PRs by default with the repository template, issue-scope verification, and conservative readiness freshness checks
- [design doc template](./templates/design-doc-template.md)
- [external data audit template](./templates/external_data_audit.md) - Record source, license, redistribution, checksum, and Robot SF use decisions before staging external assets
- [PR template](../.github/PULL_REQUEST_TEMPLATE/pr_default.md)


## Security & network policy
- No secrets in code, configs, or commit messages.
- Avoid network access in tests; prefer local fixtures. If unavoidable, document and gate behind flags.
- Don't exfiltrate data; handle PII safely (none expected in this repo).

## Large files & artifacts policy
- Don't commit large binaries to the repo; prefer Git LFS for models/datasets when needed.
- Use the `model/` directory conventions; document artifact sources and versions.


## Quick reference and TL;DR checklist
### Quick reference commands
```bash
# Setup after installation
uv sync && source .venv/bin/activate

# Validate changes
uv run ruff check . && uv run ruff format . && uvx ty check . --exit-zero && uv run pytest tests

# Functional smoke (headless)
DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy \
uv run python -c "from robot_sf.gym_env.environment_factory import make_robot_env; env = make_robot_env(); env.reset(); print('OK')"

# Optional perf smoke
DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy \
uv run python scripts/validation/performance_smoke_test.py

# If running commands outside of `uv run`, activate the virtual environment:
source .venv/bin/activate
```

The `uvx ty check . --exit-zero` step is advisory: it should report findings without failing the
command. Treat findings in touched code and stable contracts as reviewer-actionable even though the
phase exits zero.

### Proportional validation

Validation depth follows [`docs/maintainer_values.md`](./maintainer_values.md): apply proof in
proportion to risk. Do not treat the heaviest path as the default for every change.

- **Low-risk docs/instruction changes** use the cheap path by default: inspect the diff, verify
  changed links or referenced paths, and run lightweight automated checks when they exist
  (for example, `BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh` for
  context-only or proof-heavy docs surfaces).
- **Runtime, benchmark, metric, schema, model-provenance, and paper-facing changes** need
  executable proof appropriate to the claim (tests, benchmark runs, schema checks, etc.).
- **Strong claims escalate the bar even when the touched file is documentation.** A docs-only
  change that makes a benchmark, metric, schema, model-provenance, or paper-facing claim still
  needs the corresponding strength of evidence.

If this section conflicts with current maintainer direction or [maintainer_values.md](./maintainer_values.md),
follow the higher-precedence source and make the smallest doc update needed to remove the drift.

### TL;DR workflow checklist

1) Clarify requirements and pick the validation path by change type (see
   [Proportional validation](#proportional-validation) above; `docs/maintainer_values.md` is the
   higher-precedence source).
2) For non-trivial runtime/benchmark/metric/schema/paper-facing changes, draft a design doc under
   `docs/` and link the issue; for low-risk docs/instruction changes, skip the design doc unless
   it clarifies scope.
3) Implement with small, reviewed commits.
4) Add/extend tests in `tests/` or `tests/pygame/` when touching runtime behavior.
5) Run the gates that match the risk: diff/link/path checks and lightweight automated checks for
   docs/instruction changes; Install Dependencies → Ruff: Format and Fix → Check Code Quality
   (Ruff + advisory ty) → Type Check (advisory) → Run Tests for runtime and claim-heavy changes.
6) Update docs/diagrams; run “Generate UML” if classes changed.
7) Open PR with summary, risks, validation evidence, and links to docs/tests. Explicitly state
   which gates were skipped when using the cheap docs/instruction path.

### Agent run manifest for PRs

For nontrivial agent-assisted runs (Codex, Claude Code, Copilot, or other), attach or link an
`agent_run_manifest.yaml` so the run is auditable. See
[Agent Run Manifest](./agent_run_manifest.md) for when this is required versus optional, where to
store it, and trace/log hygiene. Start from
[`docs/templates/agent_run_manifest.yaml`](./templates/agent_run_manifest.yaml). Do not block all
PRs on this yet; it is required for major agent-assisted work that creates or changes durable
evidence, benchmark/reporting gates, generated artifacts, CI/release policy, or substantial code
paths, and recommended for multi-agent or multi-run tasks.

- [ ] If this PR used a nontrivial agent run, attach or link an agent_run_manifest.yaml and confirm trace/log redaction was checked.

### Final-readiness checklist for scripted tooling work
- Run `uv run ruff check <touched_files>` and `uv run ruff format <touched_files>` before finalizing.
- Run focused tests that cover modified paths (for example `uv run pytest tests/dev/test_snapshot_pr_queue.py`).
- Run changed-file coverage if practical:
  `uv run python scripts/coverage/check_changed_files_coverage.py --base $BASE_REF --include "scripts/dev/*" --include "tests/dev/*"`.
- Ensure the worktree is clean for final PR-ready evidence:
  `git status --short` should be empty, then rerun final proof with `PR_READY_MODE=final`.
  Before running a broad `git status --short --untracked-files=normal`, use the compact state
  snapshot helper to see tracked changes and generated roots without dumping generated trees.

### Per-Test Performance Budget

To prevent regression of integration test runtime, a performance budget policy is enforced for all tests:

Policy defaults (feature 124):
- Soft threshold: < 20s (advisory – prints guidance when exceeded)
- Hard timeout: 60s (enforced via `@pytest.mark.timeout(60)` or signal alarms inside long-running integration tests)
- Report count: Top 10 slowest tests printed at session end
- Relax mode: Set `ROBOT_SF_PERF_RELAX=1` to suppress soft breach warnings (use sparingly; still prints report)
- Enforce mode: Set `ROBOT_SF_PERF_ENFORCE=1` to escalate any soft or hard breach to a test session failure

Implementation components (all under `tests/perf_utils/`):
- `policy.py` – `PerformanceBudgetPolicy` dataclass providing `classify(duration)-> none|soft|hard`
- `reporting.py` – aggregation and formatted slow test report
- `guidance.py` – deterministic heuristic suggestions (reduce episodes, horizon, matrix size, etc.)
- `minimal_matrix.py` – single-source helper for minimal benchmark scenario matrix (used by resume & reproducibility tests)

Collector flow:
1. Each test call duration captured via a timing hook in `tests/conftest.py`.
2. At terminal summary the top-N slow tests are ranked and printed with breach classification & guidance lines.
3. If `ROBOT_SF_PERF_ENFORCE=1` (and relax not set) any soft or hard breach converts the run to a failure (exit code changed). Optional internal overrides: set `ROBOT_SF_PERF_SOFT` / `ROBOT_SF_PERF_HARD` for targeted enforcement tests.

Guidance examples:
- Soft breach near 25s: "Reduce episode count / seeds", "Use minimal scenario matrix helper"
- Very long (>40s) test: horizon + matrix recommendations prioritized

Authoring guidance for new tests:
- Keep semantic assertions; minimize episodes (`max_episodes=2`), horizon, seed list
- Reuse `write_minimal_matrix` instead of duplicating inline YAML
- Assert absence of heavy artifacts (videos) where not required

Performance troubleshooting checklist:
1. Confirm `smoke=True` or minimal workload flags applied
2. Reduce `max_episodes`, `initial_episodes`, `batch_size`
3. Disable bootstrap sampling (`bootstrap_samples=0`)
4. Lower `horizon_override`
5. Ensure `workers=1` for deterministic ordering in timing-sensitive tests

When relaxing:
Use `ROBOT_SF_PERF_RELAX=1` temporarily only for known CI variance; file a follow-up issue if sustained.

Hard timeout breaches should be rare; investigate infinite loops or large scenario expansions if encountered.

## Multi-robot LiDAR and sprite rendering

### LiDAR robot detection toggle
- `LidarScannerSettings.detect_other_robots` controls whether LiDAR rays include other robots.
- Default is `True`.
- This modifies ray distances only and keeps observation keys unchanged (`drive_state`, `rays`).

### SimulationView representation modes
- `SimulationView` supports per-entity render mode fields:
- `robot_render_mode`, `ped_render_mode`, `ego_ped_render_mode` with values `circle` or `sprite`.
- Optional sprite paths:
- `robot_sprite_path`, `ped_sprite_path`, `ego_ped_sprite_path`.
- If sprite loading fails, rendering falls back to circles and emits a warning.

### Example
```python
from robot_sf.render.sim_view import SimulationView
from robot_sf.sensor.range_sensor import LidarScannerSettings

lidar_cfg = LidarScannerSettings(detect_other_robots=True)

view = SimulationView(
    robot_render_mode="sprite",
    ped_render_mode="sprite",
    ego_ped_render_mode="sprite",
)
```
