# Issue #3556 Seed-Sufficiency Handoff Check

Plain-language summary: the exported ScenarioBelief seed-sufficiency command was run against the expected retained campaign output location, but no retained campaign report root was available there, so no seed-sufficiency evidence can be promoted from this check.

## Claim Boundary

- Evidence status: `blocked`.
- Conservative decision label: `blocked_missing_retained_campaign_outputs`.
- What ran: `scripts/tools/analyze_seed_sufficiency.py` only, using the handoff command exported by the issue #3556 screening report.
- What did not run: no full benchmark campaign, no new scenario family, no Slurm or GPU submission, no belief-mode semantic change, and no paper or dissertation claim edit.

## Command

```bash
uv run python scripts/tools/analyze_seed_sufficiency.py \
  --campaign-output-root output/issue_3556_belief_mode_campaign \
  --campaign-id issue_3556 \
  --output-dir output/issue_3556_seed_sufficiency
```

## Result

The command exited with code `1` before analysis because the expected retained campaign output root was missing:

```text
FileNotFoundError: Campaign output root does not exist: output/issue_3556_belief_mode_campaign
```

The analyzer therefore produced no interval-width, ranking-stability, or seed-budget artifact for issue #3556 in this run.

## Decision

`blocked_missing_retained_campaign_outputs`: seed sufficiency remains unproven until retained campaign outputs include the analyzer-required files:

- `reports/seed_variability_by_scenario.json`
- `reports/seed_episode_rows.csv`

Once those retained reports are available, rerun the same handoff command and replace this blocked note with the analyzer output summary.
