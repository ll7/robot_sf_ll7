# Issue #5263 exact-repeat evidence packet

This packet defines the exact CPU-only repeat campaign needed to determine whether the seven
retained knife-edge cells reproduce bit-for-bit. It is a diagnostic campaign contract, not a
campaign result or a benchmark claim.

## What is pinned

`build_exact_repeat_campaign_packet.py manifest` reads the committed #4978 flakiness report and
its 400 retained episode rows. It selects the seven cells marked `knife_edge`, retains all 20
original seeds from each, and requires exactly three repeats: 420 episode executions in total.
Each target records its scenario ID, planner, seed, horizon, source Git revision, and source
configuration hash. The manifest is CPU-only and single-worker; it never stores a target-host
assignment.

The retained fixture does not contain runnable `scenario_params` or `planner_config` objects. That
is an intentional fail-closed blocker: a host must acquire those definitions from the original
campaign configuration and prove that their hashes match the manifest before execution. The packet
therefore cannot be mistaken for an empirical result.

## Commands

```bash
uv run python scripts/benchmark/build_exact_repeat_campaign_packet.py manifest \
  --baseline-report tests/fixtures/benchmark/scenario_flakiness_issue_4978/real_campaign_flakiness_report.json \
  --source-episodes tests/fixtures/benchmark/scenario_flakiness_issue_4978/real_campaign_episodes.jsonl \
  --output output/issue_5263/exact_repeat_manifest.json

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

`verify-host` rejects missing targets, a Git revision or per-target horizon/config hash that does
not match the manifest, non-CPU or multi-worker metadata, absent NumPy/Numba versions,
not-exactly-three repeats, malformed trajectory hashes, and unrecorded divergences. A
repeat is identical only when its binary outcome and SHA-256 trajectory hash agree. Otherwise the
host report must state the first differing repeat and field. `compare-hosts` accepts only distinct
machine identifiers with matching pinned NumPy and Numba versions; a version mismatch is
divergent, not a successful comparison.

## Evidence status and remaining action

No full benchmark campaign, Slurm/GPU submission, or paper/dissertation claim update was made by
this packet. The remaining empirical action is to recover the original runnable definitions, run
the 420 CPU-only repeats on one host, register the verified report under this evidence directory,
then run and compare the second-host near-miss repeat with its environment manifest.
