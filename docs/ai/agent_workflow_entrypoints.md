# Agent Workflow Entrypoints And Large-File Navigation

[Back to Documentation Index](../README.md)

This guide helps agents start from the right command and read only the code region they need.
Use it when a task needs repository commands, model lookup, validation, or navigation through
large files.

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
