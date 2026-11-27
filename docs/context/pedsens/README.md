# Pedestrian sensitivity study (#10190)

The study measures how the named pedestrian settings affect planner outcomes
on the 0.0.8 scenario and planner roster. Full acquisition, the release behaviour
receipt, model adoption and paper evidence admission are pending. The study
starts after the 0.0.8 main and doorway release runs are dispatched.

The [study config](../../../configs/benchmarks/pedestrian_sensitivity_10190.yaml)
consumes the authored 0.0.8 release matrix (48 scenarios), 14 planner bindings and
scenario-specific horizons unchanged. It replaces the release seed policy with
1001–1030, and refuses every other seed, including retired 111–140 and the sealed
evaluation seed band. Repeats, booleans, floats and missing seeds are also refused.

## Opt-in definitions and sources

| Factor | Override | Source / interpretation |
| --- | --- | --- |
| legacy | no profile overrides | Existing spawn-coupled desired speed (normally 0.65 m/s); collision/placement radius 0.40 m, force radius 0.35 m. Authored single walkers retain their original speeds. |
| radius | `ped_radius: 0.25`, `ped_force_radius: 0.25` | #10190's width-preserving rigid-disc approximation to ANSUR outer shoulder half-width, roughly 0.225–0.255 m. |
| speed | mean 1.29, SD 0.19 m/s | #10190 / #10074 Bordeaux free-walking target, attributed to Moussaïd et al. (2009). |
| spread | mean 0.65, SD 0.19 m/s | Mean-matched control separating dispersion from mean. |
| wall | `obstacle_force_profile: gradient_v3` | The actual opt-in implementation from PR #10073; corrected surface-distance law and its development parameters. |
| combined | radius + speed + wall | Uses all three selectors together. |

Speed profiles use rejection-truncated normals on **[0, 3] m/s**, using the existing
sampler and production speed binding. Ordinary tiers retain legacy clipping and
0.2 SD; defaults are unchanged. The new controls use the existing `InitVar` and
explicit-override serialization pattern, so absent controls preserve the base
settings bytes/hash. The radius profile applies to force geometry, placement,
collision geometry and native clearance metrics. The pair social kernel still
uses centre distance; this study does not invent a contact/body-force model.

These are target inputs, not adopted/calibrated defaults. Reference values were
read with `gh` from #10190, #10074 and the linked
[2026-10-07 research snapshot](https://github.com/ll7/diss/tree/main/docs/context/research/2026-10-07_pedestrian_simulator).
The preparation does not independently revalidate the snapshot's primary-source
numbers. ANSUR outer shoulder width is a geometry approximation, not a measured
human disc radius. Truncation bounds are an explicit modeling choice.

The campaign branch combines the study driver from PR #10191 and the wall
selector from PR #10073. The wall and combined arms explicitly select
`surface_distance_unit_normal_v2`, the law required by `gradient_v3`.
The profile changes the gain and offset along with the law, so the wall factor
measures the effect of the profile as a whole.
The combined source and launch packet require independent review before acquisition.

## Commands and launch boundary

Run from repository root, using a fresh durable output directory:

```bash
uv run python scripts/benchmark/run_pedestrian_sensitivity.py \
  --mode plan --out "$PEDSENS_ARTIFACT_ROOT/plan" -n 2
uv run python scripts/benchmark/run_pedestrian_sensitivity.py \
  --mode smoke --out "$PEDSENS_ARTIFACT_ROOT/smoke" -n 2
```

In `plan` and `study` modes, use `--scenarios` and `--planners` to select
names from the release matrix. Mode `plan` expands the same inputs and writes
a manifest without dispatching any episodes. For the workstation batch, use
all six profiles, five scenarios, two planners and five seeds for 300 episodes.

The full design has **6 × 48 × 14 × 30 = 120,960 episodes**. Each of ten
three-seed shards retains every profile, scenario and planner. Use a fresh
output directory for every attempt and verify the union of complete shard
identities before computing the full-study summary. The launch packet follows
the [Slurm playbook](../../../SLURM/AGENTS.md) for the shared environment and
verified artifact custody.

`--after-sealed-dispatch` records the operator's acknowledgement that both
release dispatches have happened. The operator also verifies the reviewed
source, checkpoint availability and durable artifact destination before starting.

## Outputs, summary and interpretation

Native, schema-validated per-episode JSONL plus native provenance sidecars are
written under `factor/planner/scenario--seed`. `paired_rows.jsonl` binds these paths,
IDs, Bernoulli success/collision/near-miss outcomes, success-only seconds, execution
mode, step count, elapsed slot cost and exact failure/reset events. A near-miss or
collision rate means the fraction of episodes containing at least one native event;
counts are retained in the original rows. Normalized native goal time is converted
to seconds using its bound horizon and timestep, exactly matching the native metric.

`summary.json` reports success-ranking Kendall **tau-b** against legacy and
per-planner variant-minus-baseline deltas, with 95% percentile bootstrap CIs.
Resampling uses paired **seed clusters**, keeping every scenario, planner and
factor together; scenarios are the fixed release matrix, not a population sample.
Time-to-goal deltas condition on both arms succeeding; `paired_n` exposes this
selection. Undefined tie-only correlations or no-common-success draws remain null,
with valid/undefined draw counts. Two-seed smoke CIs cannot support conclusions.
Missing/duplicate paired slots, invalid metrics and fallback/degraded execution
refuse successful completion.

`new_failures` inventories every lost success and new collision with paired event
and raw-trace pointers. Event classes are descriptive; causal mechanism attribution
remains explicit until paired-trace review. Smoke's three new contacts are classified
in the receipt; no tuning follows from failures. The #10000 release-baseline Slurm
sweep, exception check and adversarial review remain unrun; **no passed-gate claim**.

`manipulation_FACTOR.json` records raw desired/realized free speeds; lone disc
passage/contact/clearance for 0.8, 1.0, 1.2, 1.4 m doorways; and head-on minimum
centre and surface distances. Production substrate/behavior controls are reused.
Free walkers are separated by 4 m; speed is averaged over 5–10 s. Doorway cap is
40 s; head-on cap is 20 s, with deterministic ±0.02 m lateral asymmetry. These
probes omit body rotation and source-study context. #10074 V6's onset distance
is not a minimum-passing-distance threshold. No empirical passing-distance
acceptance band is fabricated, and rigid-disc passage is not a replication of V2.

`manifest.json` records source commit, dirtiness, input/lock hashes, exact roster,
scenario inventory, seed/profile config, environment and output checksums. Native
sidecars retain expanded map/runtime identities. Existing output directories refuse
overwrite/resume. Interrupted runs remain `running` or `failed` with partial rows;
use a fresh directory and retain the partial evidence. Only a complete, checksummed
manifest is an end-to-end success receipt.

## Verification

See [test-value notes](TESTS.md), [smoke receipt](smoke_receipt.json) and
[plan](PLAN.md). Raw runs/logs remain in the lane evidence directory, outside
checkout-local output; public receipt and summary are versioned without raw private
paths. Scientific runs, SNQI-v2 calibration/admission, full-roster cost validation,
release-behavior gate and default adoption remain separate work on #10190.
