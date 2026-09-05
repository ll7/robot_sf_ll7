# Agent Workflow Entrypoints And Large-File Navigation

[Back to Documentation Index](../README.md)

This guide helps agents start from the right command and read only the code region they need.
Use it when a task needs repository commands, model lookup, validation, or navigation through
large files.

## Task Routes And Preflight Discipline

Agents must choose the bounded task route matching their assigned goal and consume existing deterministic
preflight and status outputs rather than repeatedly scanning instructions, reconstructing validation requirements,
or mutating branches during read-only review.

| Route | Purpose | Required context / evidence | First deterministic command | Permitted mutations | Authoritative acceptance command |
| --- | --- | --- | --- | --- | --- |
| **Read-only observation** | PR / issue audit, queue review, CI status, non-mutating review | Target/base/head SHAs, PR/issue metadata, triage state | `git rev-parse HEAD` or `python3 scripts/dev/watch_pr_ci_status.py <pr> --json --once` | None (fail-closed via #8321 guard; no branch pushes or merges) | Structured snapshot report or non-mutating review assessment |
| **Documentation-only edit** | Documentation, markdown, instructions, glossaries | Changed paths, referenced file/link targets | `git diff --name-only` or targeted link check | Markdown/text files under `docs/`, `.agents/`, or root instructions | `uv run python scripts/tools/sync_ai_config.py --check` and diff/link verification |
| **Implementation / runtime change** | Bugfix, feature, or refactor in runtime code/tests | Issue contract, reproduction test, plan | Focused test: `uv run pytest <path> -q` | Scoped code and tests within declared `owned_paths` | `BASE_REF=origin/main scripts/dev/pr_ready_check.sh` |
| **Scientific / benchmark interpretation** | Benchmark analysis, policy eval, metric review | Scenario/config/seed provenance, campaign runs | Canonical benchmark runner / analyzer or row inspection | None (or diagnostic scripts / artifact manifests only) | `uv run python -m robot_sf.benchmark.camera_ready_campaign --verify-only` (no fallback/degraded as success) |
| **Environment / worktree repair** | Capacity reclamation, venv repair, git worktree hygiene | Capacity inventory, worktree status, venv health | `python scripts/dev/check_worktree_capacity.py --inventory --json` | Worktree prune, `.venv` recreation, scratch cleanup | `python scripts/dev/check_worktree_optional_deps.py --profile all-extras` |

### Route Boundaries and Negative Rules

- **Read-only review never mutates branches**: A reviewer records target/base/head SHAs and inspects or fetches according to existing policy; it must never merge `origin/main` into the implementation branch or push to it. Review worktrees enforce this via the machine guard (`scripts/dev/review_worktree_guard.py`, issue #8321).
- **Validation proportional to change risk**: A pure documentation edit does not trigger an expensive simulation campaign; conversely, a runtime or benchmark change cannot pass on documentation or lint checks alone (see maintainer value hierarchy in `AGENTS.md`).
- **Environment blockers are not relaxation licenses**: Missing optional or native dependencies remain visible. An environment blocker is an explicit blocker that routes to environment repair or closes as `blocked`; it never authorizes lowering scientific gates or claiming fallback/degraded execution as benchmark success.
- **Freshness before expensive proof**: A moved PR head/base or changed material metadata invalidates prior readiness proof; re-validate against the exact current head before handoff (issue #7649).
- **Separation of observer/audit collection from mutations**: Observers and audit scripts emit bounded snapshots with producer revision, freshness timestamp, and data completeness marker. Quota exhaustion, truncated pagination, or a stale producer must never be treated as an empty-success result or authorize state mutations, issue updates, or label writes (issues #8304 and #8307).
- **Integrity of scientific indicators**: Refactoring or readability improvements must never strip units, uncertainty qualifiers, source pins, seed/config identifiers, or forbidden-inference boundaries.
- **Privacy and provenance boundaries**: Private infrastructure, account details, unpublished project context, and raw runtime logs must never leak into public PR bodies, comments, or committed artifacts.

## Command Entrypoints

Run Python through the project environment so imports such as `robot_sf` resolve consistently:

```bash
uv run python scripts/<path>.py
```

Use the same `uv run` prefix for focused validation:

```bash
uv run pytest tests/<path> -q
uv run ruff check <changed-file>
uv run ruff format --check <changed-file>
```

For broad pull request readiness, use the repository wrapper from the repository root:

```bash
BASE_REF=origin/main PYTEST_NUM_WORKERS=8 scripts/dev/pr_ready_check.sh
```

For final handoff readiness on a clean tree, prefer:

```bash
PR_READY_MODE=final BASE_REF=origin/main scripts/dev/pr_ready_check.sh
```

Model resolution is owned by `robot_sf/models/registry.py`. Search or extend that registry
instead of guessing there is a flat `robot_sf/models.py` module.

## Token-saving and shared routing compatibility

The repository keeps the shared provider/model policy in the canonical external resolver; it does
not copy a provider inventory or local route table. These compatibility entrypoints make the
token-saving workflow callable from a fresh checkout and return an explicit machine-readable
`unavailable` state when `CODEX_ROUTING_REPO` is not configured:

```bash
python3 scripts/save-codex-token-checkpoint.py --task-class issue_implementation --format text
python3 scripts/advise-provider-routing.py --json
python3 scripts/read-active-ledger.py --json --limit 1
python3 scripts/resolve-route.py --help
```

Set `CODEX_ROUTING_REPO` to a checkout of the canonical shared routing repository to delegate route
resolution/advice. Until then, continue with compact local snapshots and record
`route-unavailable`; route output remains evidence only and never substitutes for local diff,
validation, benchmark, evidence, or merge acceptance.

### Delegated-worker startup recovery

The routed-worker manifest records startup failures separately from failures after a worker has
started. Build or update it from the bounded attempt records with:

```bash
uv run python -m scripts.dev.routed_worker_manifest \
  --attempts-json <attempts.json> --chosen-index <index> \
  --target-repo <linked-worktree> --max-recovery-attempts 2
```

When `worker_started` is false and the attempt carries an HTTP 404 from the Codex responses
backend, the manifest emits `startup_backend_404` and a single next-attempt recommendation. The
recommendation is data only: it does not sleep or spawn a worker. After the retry budget is
exhausted, or for a non-retryable startup/task failure, `recovery.fallback` requires manual or
local review and keeps `independent_review_authorized` false. A successful prior worker suppresses
later retries to avoid duplicate work; every route manifest remains route evidence only.

### handoff.v2 request format

The accepted handoff input is a flat `handoff.v2` request (there is no nested `packet`):

<!-- handoff.v2-example:start -->

```yaml
schema_version: handoff.v2
handoff_type: request
task_id: ROBOTSF-EXAMPLE
provider: opencode_go
mode: issue_implementation
goal: Implement the bounded Robot SF packet and return frozen-head evidence.
owned_paths:
  - .agents/README.md
forbidden_actions:
  - push
  - open_pr
  - mutate_remote
required_context:
  - target repository frozen HEAD
  - accepted route-plan contract
required_output:
  - changed_files
  - validation_evidence
  - final_status
acceptance_gate:
  - all declared validation commands pass
  - changed files stay within owned_paths
validation_commands:
  - uv run pytest -q tests/dev/test_check_skills.py
execution_mode: external_runtime
dependencies: []
budget:
  runtime_minutes: 30
stop_conditions:
  - scope expands beyond owned_paths
  - a forbidden action is requested
side_effect_policy:
  remote_mutation: false
  local_edits: true
max_depth: 0
sync_barrier: null
```

<!-- handoff.v2-example:end -->

For a production `--out` plan, pass the explicit identity/risk/head contract
`--task-id`, `--task-class`, `--risk`, `--handoff-file`, `--frozen-head`, `--target-repo`, and
`--out`. Keep ownership and validation in the handoff file; do not duplicate them with
`--owned-paths`, `--validation`, or `--prompt` flags.

The equivalent checkout-based command below can be run from this repository without assuming
that the shared checkout is the current directory:

```bash
TARGET_REPO="$(pwd)"
TARGET_HEAD="$(git -C "$TARGET_REPO" rev-parse HEAD)"
ROUTING_REPO="${CODEX_ROUTING_REPO:?set CODEX_ROUTING_REPO to a codex-personal-skills checkout}"
HANDOFF_FILE="${HANDOFF_FILE:?set HANDOFF_FILE to the flat handoff.v2 YAML above}"
python3 "$ROUTING_REPO/scripts/resolve-route.py" \
  --task-id ROBOTSF-EXAMPLE --task-class issue_implementation --risk R1 \
  --handoff-file "$HANDOFF_FILE" \
  --frozen-head "$TARGET_HEAD" --target-repo "$TARGET_REPO" \
  --out "${TMPDIR:-/tmp}/robotsf-route-plan.json"
```

Read back the private plan's `selected_route`, `forbidden_actions`, and `acceptance_gate` before
dispatch. The resolver owns native-tier selection, evidence-gated escalation, and external-provider
budget alternatives; do not copy a volatile model inventory or add local legacy routes.

## Compact Final Handoff Contract

Every completed task must provide a concise, reproducible handoff record adhering to repository acceptance terminology. A model's prose assertion is not execution evidence:

1. **Result**: Final status (`success`, `blocked`, `diagnostic`, `not benchmark evidence`).
2. **Revisions**: Exact base SHA, head SHA, and relevant input/config digests.
3. **Changed paths**: Modified files (must stay strictly within declared `owned_paths`).
4. **Validation evidence**: Exact commands run with their exit codes and output summaries.
5. **Unrun or unavailable checks**: Explicit list of any skipped, unrun, or blocked checks, accompanied by technical rationale (e.g. GPU unavailable, optional dependency missing).
6. **Scientific scope & limitations**: Any caveats, assumptions, or boundary conditions.
7. **Next disposition**: Actionable next step (e.g. ready for PR review, follow-up issue required).

## Large-File Navigation

Large file work should be targeted. Locate an anchor first, read a bounded range, then re-locate
the anchor after edits because line numbers drift.

Useful commands:

```bash
rg -n "anchor text|function_name|class_name" <file-or-dir>
sed -n 'A,Bp' <file>
tail -n 120 <file>
```

Do not read a full large file just to find one function. If the first range is wrong, search for a
nearby symbol or heading again and then read another narrow range.

Common large or fragile files:

| File | Purpose | Navigation hint |
| --- | --- | --- |
| `robot_sf/benchmark/camera_ready_campaign.py` | Camera-ready benchmark orchestration and reporting. | Search for the specific command, planner family, or artifact phase before reading. |
| `robot_sf/benchmark/map_runner.py` | Benchmark map execution and policy construction. | Search for policy names, `_build_policy`, or scenario/map handling branches. |
| `robot_sf/benchmark/metrics.py` | Benchmark metric calculations and aggregation helpers. | Search for the metric name or schema field before changing formulas. |
| `scripts/training/train_ppo.py` | Proximal Policy Optimization training entrypoint. | Search for config loading, checkpoint, or callback anchors. |
| `scripts/validation/run_policy_search_step_diagnostics.py` | Policy-search step diagnostics launcher. | Search by candidate, diagnostic stage, or output field. |
| `docs/context/INDEX.md` | Retrieval-first context-note catalog. | Search by issue number, status marker, or topic before opening ranges. |
| `docs/dev_guide.md` | Broad development guide and command reference. | Search for the workflow or command family being edited. |
| `.agents/skills/goal-autopilot/SKILL.md` | Autonomous issue/PR workflow instructions. | Search for the phase name, claim protocol, or stop guard. |
| `docs/context/policy_search/experiment_ledger.md` | Policy-search result ledger. | Search by candidate, stage, date, or result keyword. |

## Editing Pattern

1. Search for the owner: `rg -n "<concept>" robot_sf scripts docs .agents`.
2. Read the smallest useful range around the best anchor: `sed -n 'A,Bp' <file>`.
3. Make the scoped edit.
4. Re-run `rg -n` for the anchor after editing before follow-up reads or line-specific claims.
5. Validate with focused commands first, then escalate only if the change affects shared runtime,
   benchmark semantics, schemas, or paper-facing evidence.
