# Issue #9748: executable hybrid v4 development search


Current execution uses `issue_9748_hybrid_v4_dev_split_v2.yaml` and its
hash-pinned authored schedule: doorway/group crossing H500, perpendicular
traffic/crowd navigation H400. The original v1 config declared H600, but its
simulator already used those same authored budgets: 500/500/400/400. Main's
runner cap did not extend them. There is no effective tuning-budget mismatch
with 0.0.8 (corrected D-054). The original v1 inputs, recorded log and frozen v4
parameters stay byte-identical; no retuning is performed by this migration.
The details below describe the historical v1 declared protocol; use
`--config configs/benchmarks/issue_9748_hybrid_v4_dev_split_v2.yaml` for current
preflight/validation. Any future run needs a new frozen input closure and log;
the v1 log must not be rebound to v2. Native, complete, nondegraded development
cells remain required.

This runner consumes the development definitions in PR #9908. It uses
only the four `issue_9748_dev_*` scenarios, seeds 1001–1030, declared runner cap H600, and dt 0.1.
The historical effective simulator budgets were H500/H500/H400/H400.
It calls the map runner's native `run_map_episode` and validates every written
episode against the canonical schema. Its outputs are development diagnostics,
not held-out evaluation or 0.0.8 release evidence.

## Prespecified search

`configs/policy_search/issue_9748_v4_tuning_search_v1.yaml` contains the
baseline plus five one-factor changes for each of the two candidate IDs in
`issue_9748_hybrid_v4_dev_split_v1.yaml`:

| Resolved v4 parameter | Baseline | Trial value |
| --- | ---: | ---: |
| `v4_slow_clearance_human` | 0.35 m | 0.45 m |
| `v4_moderate_clearance_human` | 0.60 m | 0.80 m |
| `v4_braking_margin` | 0.10 m | 0.15 m |
| `dynamic_clearance_weight` | 1.8 | 2.2 |
| `goal_progress_weight` | 4.0 | 4.5 |

The stop-clearance threshold, physical drive limits, and braking-feasibility
check are unchanged. The search is an enumerated one-factor comparison, not a
Cartesian sweep: 2 arms × 6 trials × 4 scenarios × 30 seeds = **1,440
episodes**. The objective ranks complete native trials by fewer collision
events, then more route completions, then lower mean success-only simulated
goal time. Goal time is the canonical
`time_to_goal_norm_success_only × horizon × dt`. Failed, missing, fallback, or
degraded cells make a trial ineligible. The runner reports a diagnostic order
and never writes a frozen release config.

The two scenario-adaptive v4 twins in #9874 are outside the #9908 tuning-log
candidate roster. Their release-keyed overrides are not exercised by the
development identities. The release freeze must explicitly record them as
untuned or approve a new, reviewed development protocol that includes them;
the current runner makes no selection for those slots.

## Run and log boundary

Run only after the scenario definition set has been reviewed and frozen, and
after the required route fixes. Use a clean checkout at the exact source commit;
the runner refuses release or other non-development seeds. Set `--output-root`
to a new durable directory outside the checkout and `--tuning-log` to a new
repository path. With no `--seed`, `--candidate`, or `--trial` filters, it runs
the full predeclared search. `--workers` controls independent episode processes.

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/issue_9748_hybrid_v4_dev_split_v1.yaml \
  --mode preflight --output-root /durable/issue9748-preflight
uv run python scripts/benchmark/run_issue_9748_v4_tuning.py \
  --workers 8 --output-root /durable/issue9748-v4-search \
  --tuning-log docs/context/evidence/issue_9748_v4_tuning_log_v1.json
```

The runner writes schema-checked `episodes.jsonl`, `errors.jsonl`,
`run_summary.json`, and a structured `issue_9748.tuning_log.v1` file. Each log
entry binds one candidate, one development scenario, the exact seed list, the
base candidate file hash, the resolved config hash, raw-row and error-file
hashes, outcomes, and elapsed time. It records search and runner hashes; the
#9908 validator checks the campaign, scenario, candidate, transitive input,
loader, and resolver closure against the frozen source and committed HEAD.
Preserve the
raw output directory outside the worktree and verify its hashes before treating
the log as a complete tuning record.

The #9908 validator requires the log bytes in a *later tracked commit* than
the source commit. After the run, commit the log without changing governed
inputs, then validate it:

```bash
git add docs/context/evidence/issue_9748_v4_tuning_log_v1.json
git commit -m "data(benchmark): record issue 9748 development tuning"
uv run python scripts/validation/check_issue_9748_dev_split.py \
  --config configs/benchmarks/issue_9748_hybrid_v4_dev_split_v1.yaml \
  --tuning-log docs/context/evidence/issue_9748_v4_tuning_log_v1.json --json
```

Only a separately reviewed freeze PR can replace #9874's four unfrozen
placeholders with new hashed frozen configs, update the release and smoke
templates, and record which two scenario-adaptive twins stayed untuned.
