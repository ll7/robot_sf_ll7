# VV-2 reference-planner oracles

Issue [#9732](https://github.com/ll7/robot_sf_ll7/issues/9732) adds a diagnostic
check over the release-pinned 48-scenario matrix and seeds 111–140. The config is
[`issue_9732_reference_oracles.yaml`](../configs/validation/issue_9732_reference_oracles.yaml).
It names the release manifest, reference arms, explicit probe exemptions, and
thresholds. It does not edit the source scenarios or release manifest.

The **goal** arm uses the existing blind goal policy and a `pedestrian_free_v1`
overlay. The overlay sets an exact population of zero and removes fixed SVG
pedestrians from a copied map definition. The map runner checks the actual
instantiated count against zero before recording an episode. The **stand-still**
arm uses the original pedestrian population and always commands `(0, 0)`.
The source release matrix deliberately has no pedestrians in
`classic_bottleneck_low`; its 30 rows remain required for coverage but are
listed separately and excluded from the contact-rate denominator. The config
pins this exception and preflight checks it against the source scenario.
Configured pedestrian-aware arms run under the same pedestrian-free overlay as
the goal arm. The default config checks the cheap `social_force` and `orca`
arms; the report names this roster so it cannot support a claim about any
unconfigured planner.

Run preflight before cluster submission:

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run python \
  scripts/validation/run_reference_planner_oracles.py \
  --config configs/validation/issue_9732_reference_oracles.yaml \
  --output-root output/validation/issue_9732/<run-id> --mode preflight
```

From a clean, exact-commit worktree **on a cluster compute node**, use the same
command with `--mode run`. This executes every configured arm, writes raw
`runs/<arm>/episodes.jsonl` and batch summaries, then writes
`oracle_report.json` and `oracle_report.md`. A nonzero exit means the release
gate failed; the report remains available. Preserve the raw rows, source
commit, config, run manifest, scheduler receipt, and checksums in durable
storage before using them as release evidence. A local output path or accepted
Slurm job alone is not durable or scientific proof.

To check preserved results in CI or release preparation without running the
simulator:

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run python \
  scripts/validation/run_reference_planner_oracles.py \
  --config configs/validation/issue_9732_reference_oracles.yaml \
  --output-root output/validation/issue_9732/<run-id> --mode release-gate
```

The gate requires one valid row for every scenario and seed in every configured
arm. It lists all goal failures, including probes; only explicitly declared
probes are exempt from the goal-failure threshold. Stationary contact rate is
the fraction of episodes with pedestrians present and
`metrics.ped_collision_count > 0`;
missing typed contact data blocks the gate. Dominance is paired by scenario
and seed: an aware arm fails when its goal-completion outcome is worse than the
blind goal arm. Missing, duplicated, degraded, or source-mismatched rows block
the gate. The thresholds are in YAML; this diagnostic does not alter the
pedestrian model or admit a benchmark or paper claim by itself.
