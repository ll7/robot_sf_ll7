# 0.0.8 sealed-seed runbook

Use the 0.0.8 campaign template and `release_eval_0_0_8`; validate resolved seeds
against `EVAL_SEEDS_0_0_8` before admission. The full roster resolves to 14 arms x
48 scenarios x 30 seeds = 20,160 identities, 1,440 per arm. Config validation
computes these identities and never executes them.

Before minting identities or launching the sealed main campaign and its
materialized three-width slice at the freeze commit, refresh the bundled
pedestrian physics in the launch environment:

```bash
uv sync --all-extras --reinstall-package robot-sf
```

Ordinary `uv sync` can retain an older installed `pysocialforce` copy after
`fast-pysf` changes. The reinstall command rebuilds the `robot-sf` wheel and
refreshes that bundled copy; restart the launch process afterwards so it imports
the rebuilt files. If extra package files remain, rebuild a fresh venv rather
than reusing its directory. The sealed guard requires imported `robot_sf` to
come from the checked repository and every non-cache `pysocialforce` file to
match the freeze commit's file set and bytes. It names the first differing file
and refuses before spawn workers. `__pycache__` directories and `.pyc` files
are excluded from that comparison. Static post-run tools validating a sealed
identity also require this matching runtime at a clean checkout of its source.

Both the fresh sealed list and retired 111..140 band are held out for all
development, calibration, tuning and rehearsals. Spawn preflight with episode
steps is evaluation work and must follow the author admission barrier; this
seed-migration lane must not run it. Use dev seeds 1001..1030 for new episodes.

Historical 0.0.7 schedules and frozen files remain immutable. Paired-by-seed
0.0.7/0.0.8 outcome comparisons no longer support the fresh evaluation: the seed
sets differ. Preserve the historical comparison tool for provenance audits.

The private mint requires exact fresh seeds and the derivation pin. After both
PRs land, regenerate manifests/packets and their hashes at the selected source;
do not reuse a packet bound to the retired band. No packet is admitted here.

## Development packaging rehearsal (D-070 / D-086)

At a clean, exact source checkout, generate two diagnostic identities through
`scripts/tools/resolve_benchmark_release_identity.py generate` with
`--development-rehearsal`, the selected
`configs/benchmarks/releases/benchmark_data_release_s30_h600.template.yaml`,
`--source-commit <full SHA>`, `--release-tag development-rehearsal-<full SHA>`,
`--concept-doi 10.5281/zenodo.99000001` and
`--version-doi 10.5281/zenodo.99000002`. These fixed diagnostic coordinates
are never reserved or published. Use `--development-seeds 1001` for the
preparatory smoke and `--development-seeds 1001,1002,1003` for the campaign;
verify each with the same resolver's `verify --identity <path>` command.
Print the resolved inventory before submission.

Stage checkpoints with `scripts/benchmark/preflight_campaign_checkpoints.py
--config <D-083 authored campaign> --stage --json --report-path <receipt>`.
From the cluster submit node, check `squeue -u "$USER"`, then submit both executions through
`SLURM/submit_release_single_node.sbatch` with an explicit
`sbatch --cpus-per-task=32` override and fixed fresh campaign IDs. The smoke
uses all 14 arms × 48 scenarios × seed 1001, the environment variable
`ROBOT_SF_DEVELOPMENT_RUNTIME_SMOKE=1`, and `-` as the fifth receipt argument.
The subsequent 2,016-cell campaign uses the smoke's exact-source
`release/release_result.json` as the fifth argument, with the smoke environment
variable unset. Both invoke the common `run_benchmark_release.py` runner;
all authored budgets, checkpoint and spawn gates remain active.

The runner performs publication export through the common exporter. Validate
the resulting bundle with `scripts/tools/publication_preflight.py`, including
checksums, roles, commit/SNQI and campaign/result reconciliation. Verify the
SNQI development calibration's binding to the selected D-083 inputs. Run
`scripts/analysis/compare_release_distributions.py` against the pinned 0.0.7
bundle using `--diagnostic-partial`, the verified successor identity and its
SHA-256. This permits development coverage without changing source/schema
admission or the comparator's statistical implementation. Release mode refuses
the rehearsal identity. Record commands, exits and output paths, and preserve
the intake archive's size and SHA-256. These steps do not mint, publish, tag or
reserve a DOI, and their outputs remain permanently non-releasable.
