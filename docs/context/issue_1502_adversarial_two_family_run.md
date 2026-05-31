# Issue #1502 Adversarial Two-Family Run (2026-05-31)

Date: 2026-05-31

Related issues:

- <https://github.com/ll7/robot_sf_ll7/issues/1502>
- <https://github.com/ll7/robot_sf_ll7/issues/1501>
- <https://github.com/ll7/robot_sf_ll7/issues/1500>
- <https://github.com/ll7/robot_sf_ll7/issues/1488>
- <https://github.com/ll7/robot_sf_ll7/issues/691>

## Scope

This note records the bounded two-family adversarial comparison run for #1502. It follows the
frozen #1500 manifest and uses the successful #1501 crossing/TTC smoke as the execution gate.

Executed rows:

- `crossing_ttc`: policies `goal` and `orca`, samplers `random` and `optuna_tpe`, budget 256 per
  sampler, seed 42.
- `classic_head_on_corridor`: `guided_route_search`, 100 trials, seed 123.
- Design exclusions remain explicit: `guided_route_search` is `not_available` for
  `crossing_ttc`; random/TPE are `not_available` for `classic_head_on_corridor`.

This is bounded development stress evidence. It is not paper-facing benchmark evidence and does
not support direct cross-family absolute failure-count comparisons because the families use
different search paradigms.

## Launcher And Job

- Worktree: `../robot_sf_ll7.worktrees/issue-1502-adversarial-two-family`
- Branch: `issue-1502-adversarial-two-family`
- Commit: `d4a49b26772f72ad082a62d1485611c22034a6d9`
- Launcher: `SLURM/Auxme/adversarial_two_family_1502.sl`
- Submitted job: `12664`
- SLURM result: `COMPLETED`, exit code `0:0`, partition `a30`, elapsed `00:04:41`
- Label: `issue1502-two-family-d4a49b26`

Command:

```bash
ADVERSARIAL_1502_LABEL=issue1502-two-family-d4a49b26 \
scripts/dev/sbatch_use_max_time.sh --time 04:00:00 --partition a30 --qos a30-gpu \
  SLURM/Auxme/adversarial_two_family_1502.sl
```

## Outputs

Worktree-local output root:

`output/adversarial/issue_1502/issue1502-two-family-d4a49b26/`

Synced SLURM output root:

`output/slurm/adversarial-two-family1502-job-12664/`

Compact tracked evidence:

`docs/context/evidence/issue_1502_adversarial_two_family_2026-05-31/`

Tracked files:

- `row_status_summary.json`
- `goal_sampler_comparison.json`
- `orca_sampler_comparison.json`
- `archive.json`
- `classic_head_on_corridor_guided_summary.json`
- `classic_head_on_corridor_guided_report.md`
- `checksums.sha256`

Raw candidate bundles, route overrides, trajectory images, episode records, and full Slurm logs
remain under ignored `output/`.

## Row Status Summary

| Family | Policy | Sampler | Status | Valid Non-Failure | Valid Failure | Invalid Candidate | Route Valid Trial | Route Failed Trial | Not Available |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `crossing_ttc` | `goal` | `random` | available | 108 | 26 | 122 | 0 | 0 | 0 |
| `crossing_ttc` | `goal` | `optuna` | available | 5 | 5 | 246 | 0 | 0 | 0 |
| `crossing_ttc` | `orca` | `random` | available | 110 | 24 | 122 | 0 | 0 | 0 |
| `crossing_ttc` | `orca` | `optuna` | available | 5 | 5 | 246 | 0 | 0 | 0 |
| `crossing_ttc` | `all` | `guided_route_search` | not_available | 0 | 0 | 0 | 0 | 0 | 1 |
| `classic_head_on_corridor` | `classic_global_theta_star` | `guided_route_search` | available | 0 | 0 | 0 | 40 | 60 | 0 |
| `classic_head_on_corridor` | `all` | `random` | not_available | 0 | 0 | 0 | 0 | 0 | 1 |
| `classic_head_on_corridor` | `all` | `optuna_tpe` | not_available | 0 | 0 | 0 | 0 | 0 | 1 |

Aggregate counts:

- `valid_failure`: 60
- `valid_non_failure`: 228
- `invalid_candidate`: 736
- `valid_route_trial`: 40
- `failed_trial`: 60
- `not_available`: 3

`valid_failure`, `invalid_candidate`, `failed_trial`, `fallback`, `degraded`, and
`not_available` rows do not count as success evidence.

## Failure Archive

The crossing/TTC curated archive has schema `adversarial_failure_archive.v1` and contains:

- `source_candidate_count`: 1024
- `source_manifest_count`: 4
- `archived_failure_count`: 60
- `cluster_count`: 2

The two clusters are collision clusters separated by policy row. The archive preserves replay
commands for failure entries, but this job did not run a separate replay-determinism sweep.

## Interpretation

The run is successful as an execution and row-accounting packet for #1502:

- all intended available rows ran under the committed Slurm launcher,
- unavailable search-engine/family combinations were represented explicitly,
- crossing/TTC random rows produced many more valid failures than optuna rows at this budget,
- optuna rows had very high invalid-candidate counts, which should be treated as a sampler-quality
  signal rather than hidden success,
- guided route search found 40 valid head-on-corridor route trials out of 100.

Do not compare the 60 crossing/TTC archived failures directly against the head-on route-search
valid-trial count; the search spaces and objectives differ. #1503 may use this packet for
bounded synthesis only if it preserves these caveats.

## Validation And Artifact Policy

Pre-submit checks:

- `bash -n SLURM/Auxme/adversarial_two_family_1502.sl`
- `scripts/dev/sbatch_use_max_time.sh --dry-run --partition a30 --qos a30-gpu SLURM/Auxme/adversarial_two_family_1502.sl`
- tiny local preflight with `ADVERSARIAL_1502_SYNTHETIC=true`, crossing budget `1`, route trials
  `6`
- `git diff --check -- SLURM/Auxme/adversarial_two_family_1502.sl`

Post-run checks:

- `sacct -j 12664 --format=JobID,JobName%24,State,ExitCode,Partition,Elapsed,Start,End -P`
- `jq . output/adversarial/issue_1502/issue1502-two-family-d4a49b26/row_status_summary.json`
- `jq '{schema_version, created_at, clusters: (.clusters|length), entries: (.entries|length)}' output/adversarial/issue_1502/issue1502-two-family-d4a49b26/crossing_ttc/archive.json`

Artifact classification:

- `tracked-compact-evidence`: files under
  `docs/context/evidence/issue_1502_adversarial_two_family_2026-05-31/`
- `non-evidence-local-only`: raw candidate bundles, raw manifests, route override YAML,
  trajectory overlay PNG, and full Slurm log under `output/`
- `blocked/durable-required`: none for #1502 completion; downstream #1503 should cite the compact
  evidence and rehydrate raw local outputs only if it needs deeper failure inspection.

## Follow-Up

#1502 can be treated as completed for the bounded two-family execution stage once the tracked
evidence lands. The next child stage is #1503 synthesis/reporting. #1503 should preserve the
fail-closed row taxonomy and the non-paper-evidence boundary above.
