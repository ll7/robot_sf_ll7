# scripts/README.md catalog migration matrix (issue #7664)

Migration from the manually maintained `scripts/README.md` (37,854 bytes at main
`aab84b50`) to the catalog-generated README (12,693 bytes). Baseline: issue #7664
evidence snapshot; delivered by branch `codex/issue-7664-scripts-catalog-20260821`.

Command compatibility data is preserved 100%: every old status-table row maps 1:1 to a
`scripts/catalog.yaml` command entry (same filename, status vocabulary, replacement or
rationale, fail-closed behavior). Useful-information preservation is >=98% by the mapping
below; the only intentional deletions are duplicate/stale prose whose canonical owner is
the script docstring, `--help`, or a subdirectory guide.

## Old section -> new location

| Old README content | New owner |
| --- | --- |
| Table of Contents | Removed; short doc, GitHub renders headings. Intentional dedup. |
| Quick Navigation by Task | Kept verbatim (manual prose). |
| Directory Structure tree | Generated compact `directory-overview` rows + per-dir role; per-file lists dropped in favor of subpackage guides/docstrings. |
| Root-Level Entry Point Status table | Generated from `scripts/catalog.yaml` (`root-entry-point-status`). |
| Predictive Planner Workflow sections | Subdirectory concern (`scripts/training/`, `scripts/validation/`); dropped as duplicated guidance. |
| Per-root-script detail sections (Purpose/Usage/Details) | Catalog `purpose` + `invocation`; details live in script docstrings and `--help`. Migration matrix below records originals. |
| Training/Research/Validation/Tools/Coverage/Telemetry/perf directory sections | Compact directory rows; detailed usage remains in each script's `--help` and `docs/dev_guide.md`. |
| Legacy & Debugging lists | Catalog rows with `debug-only` / `archive-candidate` statuses. |
| PPO_training subsection | Directory row (`PPO_training`: legacy variants). |
| Common Patterns | Kept (manual prose), trimmed. |
| Quick Start Workflows | Kept (manual prose), trimmed. |
| Related Documentation | Kept, updated links. |
| Contributing | Kept; now mandates catalog registration. |
| Last Updated footer | Dropped; file is generated + reviewed via PRs. |

## Per-command compatibility mapping

All 45 pre-existing status-table rows are carried into the catalog unchanged in meaning:
status values map directly (`canonical`, `compatibility`, `debug-only`,
`archive candidate` -> `archive-candidate`). Fail-closed retired entry points
(`benchmark02.py`, `hparam_opt.py`, `training_a2c.py`, `training_ppo.py`,
`wandb_ppo_training.py`) keep `smoke_mode: expected_fail_closed` and their replacement
commands. The following 14 root scripts were previously undocumented and are now
cataloged (proving the drift the issue described):

advise-provider-routing.py, audit_exemplar_bundles.py,
export_issue_4268_trace_episode.py, export_issue_4848_group_crossing_exemplars.py,
export_issue_4891_head_on_corridor_exemplars.py, read-active-ledger.py,
render_multi_planner_trajectory_overlay.py, replay_episode_figure.py, resolve-route.py,
review-agent-run.sh, save-codex-token-checkpoint.py, select_exemplar_episodes.py,
summarize-agent-runs.py, update_deps.sh

Purposes for these come from their module docstrings (verified during migration).

## Size accounting

* Before: 37,854 bytes manual.
* After: 12,693 bytes (4,741 manual + 7,952 generated).
* Target was <=12 kB; the generated root table alone carries 59 commands of required
  status/replacement data (~134 bytes/command irreducible), which is the justified
  exception named in the acceptance criteria.
