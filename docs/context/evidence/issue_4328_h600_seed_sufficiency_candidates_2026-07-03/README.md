# Issue #4328 h600 Candidate Seed-Sufficiency Closure Packet

Plain-language summary: this packet evaluates the named retained h600 report roots proposed in issue #4328 against the issue #3556 ScenarioBelief seed-sufficiency closure contract (existence on host + the two analyzer-required reports + ScenarioBelief provenance), then either runs the analyzer on the best fully compatible candidate or fails closed with an explicit per-candidate blocker.

## Claim Boundary

- Closure target issue: `#3556` (attempt filed under `#4328`).
- Evidence status: `blocked`.
- Decision label: `blocked_no_compatible_candidate`.
- No named h600 candidate root satisfied the #3556 seed-sufficiency contract (existence on host + required reports + ScenarioBelief provenance), so no seed-sufficiency evidence is promoted.

## Required Report Files (per candidate)

- `reports/seed_variability_by_scenario.json`
- `reports/seed_episode_rows.csv`

## Candidate Compatibility

| candidate | root | exists on host | reports present | provenance compatible | blockers |
| --- | --- | --- | --- | --- | --- |
| `issue3810-h600-longhorizon-confirm-run` | `output/issue3810-h600-longhorizon-confirm-run/13268` | False | False | False | root_absent_on_host, missing_required_reports, provenance_incompatible_with_3556 |
| `issue3810-h600-extroster-run` | `output/issue3810-h600-extroster-run/13273` | False | False | False | root_absent_on_host, missing_required_reports, provenance_incompatible_with_3556 |
| `issue4230-h600-hybrid-roster-run` | `output/issue4230-h600-hybrid-roster-run/13282` | False | False | False | root_absent_on_host, missing_required_reports, provenance_incompatible_with_3556 |

## Analyzer Command

```bash
uv run python scripts/tools/analyze_seed_sufficiency.py --output-dir output/issue_4328_h600_seed_sufficiency
```

## Queue-Row Request (deferred; no execution)

- Kind: `scenario_belief_seed_sufficiency_campaign`
- Runner: `scripts/benchmark/run_belief_mode_safety_campaign_issue_3556.py`
- Why: A #3556-specific ScenarioBelief drop-vs-retain campaign is required to close seed-sufficiency; foreign h600 roster campaigns answer a different question.
- Required reports: reports/seed_variability_by_scenario.json, reports/seed_episode_rows.csv

## Decision

`blocked_no_compatible_candidate`: Run a #3556-specific ScenarioBelief drop-vs-retain campaign that emits reports/seed_variability_by_scenario.json and reports/seed_episode_rows.csv, or, if a maintainer accepts a foreign h600 root as a proxy, restore that root on the analysis host and rerun the analyzer command below.

## Out of Scope (confirmed)

- No full benchmark campaign run.
- No Slurm/GPU submission.
- No ScenarioBelief belief-mode semantic change.
- No paper/dissertation claim edit.

Complements the durable-root closure packet in `docs/context/evidence/issue_3556_seed_sufficiency_closure_2026-07-03/` by evaluating the specific named h600 candidate roots (which fall outside the resolver's default search roots) and adding an explicit ScenarioBelief provenance gate.
