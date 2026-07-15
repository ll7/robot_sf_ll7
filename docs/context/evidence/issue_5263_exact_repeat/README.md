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

uv run python scripts/benchmark/build_exact_repeat_campaign_packet.py execute \
  --resolved-bundle docs/context/evidence/issue_5263_exact_repeat/resolved_definitions.json \
  --output-dir output/issue_5263

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
absent `uv.lock` SHA-256 hashes, not-exactly-three repeats, malformed trajectory hashes, and unrecorded divergences. A
repeat is identical only when its binary outcome and SHA-256 trajectory hash agree. Otherwise the
host report must state the first differing repeat and field. `compare-hosts` accepts only distinct
machine identifiers and marks a cell `divergent` unless the hosts have matching NumPy, Numba,
Python, source-commit, and `uv.lock` identities as well as matching repeat fingerprints. Runtime
or provenance drift is recorded in `provenance_mismatches`; it cannot become an `identical` row.

## Evidence status and remaining action

No full benchmark campaign, Slurm/GPU submission, or paper/dissertation claim update was made.
The runnable-definition and executor blockers are resolved.

### Issue #5498 Single-Host Exact-Repeat Result (Registered 2026-07-13)

The predeclared 420 CPU-only repeats (140 targets, 3 repeats each, single worker) were executed on
one host and the verified report is registered under this directory:

- `issue_5498_host_result.json` — raw `scenario_exact_repeat_host_result.v1` payload (machine id redacted).
- `issue_5498_verified_host_result.json` — `scenario_exact_repeat_verified_host_result.v1` after `verify-host`.
- `issue_5498_provenance.json` — reproducible command, manifest/git revision, and artifact SHA-256.

Coverage: 140/140 targets executed, **no missing or silently skipped targets**. Planners `orca`
(60 targets) and `goal` (20 targets) run natively; all 80 native-run targets are bitwise-identical
across their three repeats (SHA-256 outcome+metric trajectory hashes agree). Planner `ppo` (60
targets) **degraded**: its forked planner-step worker crashes/timing-out on this host, the runner
falls back to a zero-velocity action, and every repeat is a no-op. Those 60 targets are recorded as
explicit `unrunnable` dispositions with the `degraded` flag and are **excluded from the
bitwise-identical determinism claim** (3 of 7 cells unrunnable). This is exactly the fail-closed
behavior issue #5498 requires: fallback/degraded rows are not success evidence.

Evidence grade: observed single-host diagnostic. NOT benchmark or paper-facing evidence. The ppo
degradation is a host/runner isolation defect, not a determinism verdict; it is recorded, not
promoted.

### Issue #5498 Native-PPO Primary-Host Registration (2026-07-15)

The native-PPO planner-step worker-isolation defect was fixed in #5740 and the exact-repeat
disposition regression in #5790; PPO now runs natively through the `run_episode` exact-repeat path.
The PPO cells (60 targets across 3 cells) were re-executed natively on the **primary host**, closing
the degraded-PPO gap left by #5499. Artifacts registered under this directory:

- `issue_5498_native_ppo_host_result.json` — raw `scenario_exact_repeat_host_result.v1` payload
  (PPO-only bundle, 60 targets x 3 repeats, single worker, CPU-only).
- `issue_5498_native_ppo_verified_host_result.json` — `scenario_exact_repeat_verified_host_result.v1`
  after `verify-host` against the re-hashed PPO-only manifest slice.
- `issue_5498_native_ppo_provenance.json` — reproducible command, source/generation commit, artifact
  SHA-256, and machine identity.

Coverage: **60/60 PPO targets runnable** on the primary host (`machine_id` `auxme-imech036`,
NumPy 2.4.6, Numba 0.66.0, Python 3.13.13, `uv.lock` identity present), **no missing or silently
skipped targets**. All 3 PPO cells are **bitwise-identical** across their three repeats
(SHA-256 outcome+metric trajectory hashes agree) — the first registered native-PPO primary-host
exact-repeat determinism result for this campaign. This is precisely the fail-closed behavior #5498
requires: ppo is no longer counted as `unrunnable`/`degraded`, and no fallback row was promoted.

Evidence grade: observed single-host primary-host diagnostic. NOT benchmark or paper-facing
evidence. This registration satisfies the in-scope, single-host-achievable capability for #5498; it
is not a duplicate of the #5499 single-host orca+goal run, the #5740 worker-isolation fix, or the
#5778 cross-host provenance matcher.

### Remaining Action After #5498

The predeclared **second-host near-miss comparison** is not yet run. It requires a second
distinct host with matching pinned NumPy/Numba versions; register `cross_host_matrix.json` via
`compare-hosts` once that run exists. `compare-hosts` hard-raises when both inputs share a
`machine_id` (`exact_repeat_campaign.py`), so the cross-host matrix is structurally impossible on a
single machine and is reserved for a two-host campaign owner. The native-PPO primary-host artifact
registered above is the input the second-host run will be compared against.
