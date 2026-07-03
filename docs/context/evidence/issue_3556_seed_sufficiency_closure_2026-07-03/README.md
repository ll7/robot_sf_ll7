# Issue #3556 Seed-Sufficiency Closure Packet

Plain-language summary: this packet resolves whether a retained ScenarioBelief campaign root with the analyzer-required seed-sufficiency reports exists under the searched durable roots, then either runs the analyzer or fails closed with an explicit missing-artifact blocker.

## Claim Boundary

- Evidence status: `blocked`.
- Decision label: `blocked_missing_retained_campaign_outputs`.
- No retained issue #3556 campaign root exposing the analyzer-required report files was found under the searched durable roots, so no seed-sufficiency evidence is promoted.

## Required Retained Report Files

- `reports/seed_variability_by_scenario.json`
- `reports/seed_episode_rows.csv`

## Searched Durable Roots

| search root | exists | campaign roots found | usable (all reports) | missing reports |
| --- | --- | --- | --- | --- |
| `docs/context/evidence` | True | — | — | — |
| `output/issue_3556_belief_mode_campaign` | False | — | — | — |

## Analyzer Command

```bash
uv run python scripts/tools/analyze_seed_sufficiency.py --campaign-output-root docs/context/evidence --campaign-output-root output/issue_3556_belief_mode_campaign --campaign-id issue_3556 --output-dir output/issue_3556_seed_sufficiency
```

## Decision

`blocked_missing_retained_campaign_outputs`: Restore or point to a retained issue #3556 campaign root that contains reports/seed_variability_by_scenario.json and reports/seed_episode_rows.csv under one of the searched roots, then rerun this resolver.

## Out of Scope (confirmed)

- No full benchmark campaign run.
- No Slurm/GPU submission.
- No ScenarioBelief belief-mode semantic change.
- No paper/dissertation claim edit.

Supersedes the single-path handoff probe in `docs/context/evidence/issue_3556_seed_sufficiency_handoff_2026-07-03/` by searching durable roots and recording a reproducible per-root manifest.
