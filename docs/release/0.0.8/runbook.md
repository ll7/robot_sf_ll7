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
