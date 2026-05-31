# Issue #1861 Adversarial Replay Determinism Gate (2026-05-31)

Date: 2026-05-31

Related issues:

- <https://github.com/ll7/robot_sf_ll7/issues/1861>
- <https://github.com/ll7/robot_sf_ll7/issues/1488>
- <https://github.com/ll7/robot_sf_ll7/issues/1502>
- <https://github.com/ll7/robot_sf_ll7/issues/1503>

## Scope

This note records a compact replay-determinism gate for representative archived failures from the
Issue #1502 adversarial two-family run. It uses tracked Issue #1502/Issue #1503 compact evidence and
regenerates the crossing/TTC replay inputs from tracked archive candidate payloads plus committed
scenario/search config. It does not run new adversarial search and does not submit new Slurm work.

Evidence summary:

`docs/context/evidence/issue_1861_adversarial_replay_2026-05-31/replay_determinism_summary.json`

Raw regenerated replay bundles and episode JSONL were written under ignored local output:

`output/adversarial/issue_1861_replay_gate/`

Those raw files are disposable worktree-local evidence. The tracked summary above records the
selection, commands, classifications, and comparison signatures.

## Live Job And Issue Triage

Before this analysis, the live queue was empty:

```bash
squeue --me --format='%.18i %.28j %.10T %.12M %.12l %.10P %.8D %R'
```

Recent same-day `sacct` evidence showed only one completed new adversarial job and the earlier
canceled issue-852/791 probes:

- `12664` `adv1502-2fam`: `COMPLETED`, exit `0:0`, `a30`, elapsed `00:04:41`.
- `12665`/`12666`: failed immediately on the old `srun` launcher path.
- `12667`/`12668`: canceled after startup once duplicate historical value was identified.

The open Slurm-training issues inspected after this queue refresh were still blocked by launch
packet, artifact, or upstream-decision gates. The ready work item with new value was #1861, a
local analysis gate on the completed #1502 archive.

## Input Selection

The #1502 archive contains 60 archived crossing/TTC collision failures from 1024 source candidates
and two failure clusters:

| Cluster | Representative | Policy | Members | Failure |
|---|---|---|---:|---|
| `cluster_0000` | `failure_0015` | `goal` | 31 | collision / collision |
| `cluster_0001` | `failure_0045` | `orca` | 29 | collision / collision |

Both representatives share the same candidate payload and seed, but differ by policy row. The
source `manifest.json` and generated `scenario.yaml` paths recorded by #1502 are local `output/`
paths and are not tracked. To satisfy #1861's tracked-evidence rule, this gate regenerated
`scenario.yaml` and `route_overrides.yaml` from:

- `docs/context/evidence/issue_1502_adversarial_two_family_2026-05-31/archive.json`
- `configs/scenarios/templates/crossing_ttc.yaml`
- `configs/adversarial/crossing_ttc_space.yaml`
- `robot_sf.adversarial.bundle.write_candidate_inputs`

## Commands

The representative inputs were regenerated with a small Python snippet that called
`write_candidate_inputs` for `failure_0015` and `failure_0045`. Each representative was then run
twice:

```bash
uv run robot_sf_bench run \
  --matrix output/adversarial/issue_1861_replay_gate/failure_0015/bundle/scenario.yaml \
  --out output/adversarial/issue_1861_replay_gate/failure_0015/replay_1.jsonl \
  --algo goal \
  --no-video \
  --no-resume \
  --structured-output json \
  --external-log-noise suppress
```

The same command shape was repeated for `failure_0015` replay 2 and for `failure_0045` with
`--algo orca`.

## Results

| Representative | Policy | Classification | Execution mode | Status | Steps | Signature |
|---|---|---|---|---|---:|---|
| `failure_0015` | `goal` | deterministic | native | collision | 1 | `dee5cb239dcb744b` |
| `failure_0045` | `orca` | deterministic_adapter | adapter (`ORCAPlannerAdapter`) | collision | 1 | `aedd42c0c97a0a9a` |

For both representatives, two replays produced identical comparison signatures over status,
termination reason, steps, seed, outcome, metrics, and algorithm availability/metadata fields used
for the gate. Both replays reproduced the archived `collision` termination.

The `orca` row should stay caveated as deterministic through the documented adapter path, not as a
native planner replay. Its algorithm metadata reported `execution_mode: adapter`,
`adapter_name: ORCAPlannerAdapter`, and `projection_policy: heading_safe_velocity_to_unicycle_vw`.

## Unsupported Row

The #1503 recommendation also mentioned the best head-on route. That row remains blocked for this
gate: the compact tracked #1502 report records only this local path:

`output/adversarial/issue_1502/issue1502-two-family-d4a49b26/classic_head_on_corridor/guided_route_search/classic_head_on_corridor_low_20260531_054246_045282/route_overrides.yaml`

The route override YAML itself is ignored local output and was not present in the tracked compact
evidence. Without either a tracked route override fixture or a deterministic regeneration command
for the selected route, this gate cannot honestly classify the head-on row as deterministic.

## Recommendation

Continue #1488 with crossing/TTC representative replay determinism marked good enough for the next
bounded adversarial step. Do not treat head-on route-search replay determinism as covered yet.

Before a larger or paper-facing adversarial campaign, either:

- promote a compact, reviewable head-on `route_overrides.yaml` fixture for the selected route, or
- add a deterministic regeneration command that can reproduce the selected route override from
  tracked inputs.

This remains development-stress evidence only. It does not support paper-facing benchmark claims or
broad mechanism-diversity claims.

## Validation

- `uv sync --all-extras`
- `uv run robot_sf_bench run ...` for two repeats each of `failure_0015` and `failure_0045`
- local comparison of replay JSONL signatures
- `python3 -m json.tool docs/context/evidence/issue_1861_adversarial_replay_2026-05-31/replay_determinism_summary.json`
- `git diff --check`
