# Issue #1501 Adversarial Smoke Run (2026-05-28)

Date: 2026-05-28

Related issues:

- <https://github.com/ll7/robot_sf_ll7/issues/1501>
- <https://github.com/ll7/robot_sf_ll7/issues/1488>
- <https://github.com/ll7/robot_sf_ll7/issues/1571>
- <https://github.com/ll7/robot_sf_ll7/issues/1500>
- <https://github.com/ll7/robot_sf_ll7/issues/691>

## Scope

This note records the first executable `crossing_ttc` smoke for #1501. It follows the scope
clarified in `docs/context/issue_1571_adversarial_smoke_packet_sharpening.md`:

- run only `crossing_ttc`,
- run `random` and `optuna` samplers,
- map the frozen `classic_global_theta_star` planner row to the current `goal` benchmark policy,
- run the `orca` row directly,
- record `guided_route_search` as a `not_available` design exclusion for this family,
- keep `simulation_error` and `not_available` separate from success evidence.

## Launcher And Job

- Branch: `issue-1501-adversarial-smoke`
- Commit: `4bd5fb412d2bf023d26f674833dd379aedf8c17e`
- Launcher: `SLURM/Auxme/adversarial_smoke_1501.sl`
- Submitted job: `12656`
- SLURM result: `COMPLETED`, exit code `0:0`, partition `a30`, elapsed `00:01:59`
- Label: `issue1501-crossing-ttc-20260528-g4bd5fb41`
- Budget: 32 candidates per sampler
- Seed: 42

Command:

```bash
ADVERSARIAL_SMOKE_LABEL=issue1501-crossing-ttc-20260528-g4bd5fb41 \
ADVERSARIAL_SMOKE_BUDGET=32 \
ADVERSARIAL_SMOKE_SEED=42 \
scripts/dev/sbatch_use_max_time.sh --time 04:00:00 --partition a30 --qos a30-gpu \
  SLURM/Auxme/adversarial_smoke_1501.sl
```

## Outputs

Worktree-local output root:

`output/adversarial/issue_1501/issue1501-crossing-ttc-20260528-g4bd5fb41/`

Synced SLURM output root:

`output/slurm/adversarial-smoke1501-job-12656/`

Compact tracked evidence:

`docs/context/evidence/issue_1501_adversarial_smoke_2026-05-28/`

Tracked files:

- `row_status_summary.json`
- `goal_sampler_comparison.json`
- `orca_sampler_comparison.json`
- `archive.json`
- `checksums.md`

Raw candidate bundles, trajectories, episode records, and the full SLURM log remain under ignored
`output/`.

## Row Status Summary

The run produced four executed rows plus one design-exclusion row.

| Policy | Sampler | Status | Success | Valid Behavioral Failure | Simulation Error | Not Available |
|---|---|---|---:|---:|---:|---:|
| `goal` | `random` | available | 18 | 3 | 11 | 0 |
| `goal` | `optuna` | available | 5 | 5 | 22 | 0 |
| `orca` | `random` | available | 19 | 2 | 11 | 0 |
| `orca` | `optuna` | available | 5 | 5 | 22 | 0 |
| `all` | `guided_route_search` | not_available | 0 | 0 | 0 | 1 |

Aggregate counts:

- `success`: 47
- `valid_behavioral_failure`: 15
- `simulation_error`: 66
- `not_available`: 1

`simulation_error` and `not_available` rows do not count as success evidence.

## Sampler Comparison

Best objective values reported by `scripts/tools/compare_adversarial_samplers.py`:

| Policy | Sampler | Best Objective | Valid Candidates | Invalid Candidates | Failed Evaluations |
|---|---|---:|---:|---:|---:|
| `goal` | `random` | 15 | 21 | 11 | 0 |
| `goal` | `optuna` | 21 | 10 | 22 | 0 |
| `orca` | `random` | 12 | 21 | 11 | 0 |
| `orca` | `optuna` | 13 | 10 | 22 | 0 |

The Optuna rows found the highest objective values for both policies, but also produced more
simulation-error rows in this bounded smoke.

## Failure Archive

The curated archive has schema `adversarial_failure_archive.v1` and contains:

- `source_candidate_count`: 128
- `source_manifest_count`: 4
- `archived_failure_count`: 15
- `cluster_count`: 2

Clusters:

| Cluster | Policy | Primary Failure | Termination | Members | Representative |
|---|---|---|---|---:|---|
| `cluster_0000` | `goal` | collision | collision | 8 | `failure_0002` |
| `cluster_0001` | `orca` | collision | collision | 7 | `failure_0010` |

## Validation And Limits

Pre-submit checks:

- `bash -n SLURM/Auxme/adversarial_smoke_1501.sl`
- `scripts/dev/sbatch_use_max_time.sh --dry-run --partition a30 --qos a30-gpu SLURM/Auxme/adversarial_smoke_1501.sl`
- direct synthetic one-policy preflight with `compare_adversarial_samplers.py`
- full wrapper synthetic preflight with `ADVERSARIAL_SMOKE_SYNTHETIC=true ADVERSARIAL_SMOKE_BUDGET=1`
- `git diff --check`

Runtime proof:

- SLURM job `12656` completed with exit code `0:0`.
- Four available `crossing_ttc` policy x sampler rows emitted manifests.
- The archive builder completed and found two collision clusters.

Limitations:

- This is a smoke/execution packet, not paper-facing benchmark evidence.
- Replay determinism checks are represented by replay commands in the archive entries, but were not
  run as a separate verification sweep in this job.
- The high simulation-error count is part of the row-status evidence and should be interpreted as a
  candidate-generation/runtime quality signal, not successful stress evidence.
