<!-- AI-GENERATED (robot_sf#5263, 2026-07-11) - NEEDS-REVIEW -->
# Issue #5263 exact-repeat runnable definitions

This evidence bundle recovers and hash-verifies the runnable definitions for all 140 targets in
the exact CPU-only repeat campaign. It is diagnostic definition-recovery evidence, not a campaign
result, determinism verdict, or benchmark claim.

## What is pinned

`build_exact_repeat_campaign_packet.py manifest` reads the committed #4978 flakiness report and
its 400 retained episode rows. It selects the seven cells marked `knife_edge`, retains all 20
original seeds from each, and requires exactly three repeats: 420 episode executions in total.
Each target records its scenario ID, planner, seed, horizon, source Git revision, and source
configuration hash. The manifest is CPU-only and single-worker; it never stores a target-host
assignment.

The retained fixture does not contain runnable `scenario_params` or `planner_config` objects.
`resolve-definitions` recovers them from the canonical source campaign configuration and reproduces
every retained map-runner identity hash before writing a bundle. The registered output resolves all
140 targets across seven cells against source revision
`a5516b432fceffa71573e458aaee31c00a0b6c81`; one mismatch fails the entire operation.

## Commands

```bash
uv run python scripts/benchmark/build_exact_repeat_campaign_packet.py manifest \
  --baseline-report tests/fixtures/benchmark/scenario_flakiness_issue_4978/real_campaign_flakiness_report.json \
  --source-episodes tests/fixtures/benchmark/scenario_flakiness_issue_4978/real_campaign_episodes.jsonl \
  --output docs/context/evidence/issue_5263_exact_repeat/exact_repeat_manifest.json

uv run python scripts/benchmark/build_exact_repeat_campaign_packet.py resolve-definitions \
  --manifest docs/context/evidence/issue_5263_exact_repeat/exact_repeat_manifest.json \
  --campaign-config configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500_s20.yaml \
  --output docs/context/evidence/issue_5263_exact_repeat/resolved_definitions.json

uv run python scripts/benchmark/build_exact_repeat_campaign_packet.py verify-host \
  --manifest output/issue_5263/exact_repeat_manifest.json \
  --host-report output/issue_5263/host_result.json \
  --output output/issue_5263/verified_host_result.json

uv run python scripts/benchmark/build_exact_repeat_campaign_packet.py compare-hosts \
  --manifest output/issue_5263/exact_repeat_manifest.json \
  --first output/issue_5263/verified_host_a.json \
  --second output/issue_5263/verified_host_b.json \
  --output output/issue_5263/cross_host_matrix.json
```

`resolved_definitions.json` contains 80 deduplicated seed-specific scenario definitions, the three
planner definitions (`goal`, `orca`, and `ppo`), and 140 target references with matching retained
and computed hashes. `verify-host` rejects missing targets, a Git revision or per-target
horizon/config hash that does
not match the manifest, non-CPU or multi-worker metadata, absent NumPy/Numba versions,
not-exactly-three repeats, malformed trajectory hashes, and unrecorded divergences. A
repeat is identical only when its binary outcome and SHA-256 trajectory hash agree. Otherwise the
host report must state the first differing repeat and field. `compare-hosts` accepts only distinct
machine identifiers with matching pinned NumPy and Numba versions; a version mismatch is
divergent, not a successful comparison.

## Evidence status and remaining action

No full benchmark campaign, Slurm/GPU submission, or paper/dissertation claim update was made.
The runnable-definition blocker is resolved. The remaining empirical actions are to run the 420
CPU-only repeats on one host, register the verified report under this evidence directory, then run
and compare the second-host near-miss repeat with its environment manifest.
