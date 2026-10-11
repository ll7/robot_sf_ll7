# traces008b doorway worked-example 0.0.8 handoff

Date: 2026-10-11

Robot SF source: `373dbfde4f39667cf9e8732dabe7df5118bdeab1`

Published data: Robot SF 0.0.8, DOI `10.5281/zenodo.23150472`

Hub bundle: `/home/luttkule/external_data_hub/benchmark-results/robot_sf_ll7/traces008b-doorway-release-0.0.8-373dbfde-20261011`

Readback: `sha256sum -c SHA256SUMS` passed on `imech192` after upload.

## Scope

Included dissertation sites:

- Chapter 7 seed-flip contrast: `diss/chapters/04_results.tex:3761` and `:3863`, figure `fig:trace-doorway-seedflip`.
- Chapter 6 doorway worked example: `diss/chapters/social_navigation.tex:440`, `:481`, `:576`, `:600`, figures `fig:trace-doorway-scene`, `fig:trace-doorway-phase-portrait`, `fig:trace-doorway-ego-occupancy`.
- Chapter 8 execution-context observation levels: `diss/chapters/05_discussion.tex:616` and `:638`, table `tab:execution-context-levels`.
- `ll7/diss#3028`, the doorway worked-example simulator-defect follow-up.

Excluded because `traces008` owns them: `fig:trace-headon-compare`, `fig:trace-groupcross-compare`, `fig:trace-dual-panel-interaction`, `fig:trace-speed-clearance-phase`, `fig:trace-ego-occupancy`.

I used retrieval-broker first for `ll7/diss`; it returned no candidates from the backend, so the source audit fell back to bounded `rg` plus direct line reads. The bounded search found no additional in-scope predecessor worked-example figure in `diss/chapters` outside the doorway, seed-flip, and context sites above.

## Selection Results

The 0.0.8 published per-episode rows do not support the old exact doorway split. The old cell, `ppo` on `classic_doorway_medium`, is now 4/30 success and 26/30 collision.

The top published doorway cell by outcome entropy is `socnav_sampling` on `classic_doorway_high`: 12/30 success, 13/30 collision, 5/30 timeout. I reran it from source `373dbfde` on allowed dev seeds `1001..1200` with full trace telemetry under the required experimental adapter profile; that dev sweep produced 0/200 success and 200/200 collision. It therefore cannot render a success-vs-collision seed-flip pair without using forbidden or sealed seeds.

The trace-backed replacement cell is `goal` on `classic_doorway_medium`: published 0.0.8 rates 13/30 success and 17/30 collision, no timeout. The allowed dev-seed trace sweep produced 90/200 success and 110/200 collision, no timeout. This is not an exact 15/15 replacement, but it preserves the reader question with a clean two-outcome same-scenario doorway cell.

The closest published split was `prediction_planner` on `classic_doorway_low`: 14/30 success, 15/30 collision, 1/30 timeout. Its trace sweep was not usable in the isolated Slurm environment because `PredictionPlannerAdapter` required a missing torch dependency.

## Per-Figure Decision Table

| Figure/site | Old message | 0.0.8 message | Selected cell and rates | Trace sha256_16 |
|---|---|---|---|---|
| `fig:trace-doorway-seedflip` | One planner/scenario splits exactly 15/30 success and 15/30 collision; adjacent seeds show success vs collision. | Exact 15/15 no longer holds. Top published doorway entropy cell cannot supply an allowed-dev-seed success trace. Proposed replacement: same-scenario doorway seed sensitivity, not exact half split. | `goal` / `classic_doorway_medium`; published 13/30 success, 17/30 collision; dev sweep 90/200 success, 110/200 collision. Seed 1004 success, seed 1005 collision. | `36dec9cb4d857e48` success, `8870372b3f97d1f8` collision |
| `fig:trace-doorway-scene` | Pre-release `simple_policy` seed 141 illustrates a doorway collision-pressure episode, but the stationary pedestrian and collision flag depend on known wall-law/radius artefacts. | 0.0.8 still carries the doorway wall-law limitation; keep the worked example limitation-aware. | `goal` / `classic_doorway_medium`; same cell rates as above. Suggested single release-backed collision-pressure episode: seed 1104, pedestrian collision, 70 steps. | `174a3307556b3ecb` |
| `fig:trace-doorway-phase-portrait` | Speed-clearance read of the same pre-release doorway episode. | Use the same release-backed seed 1104 episode if the author accepts replacing the old example. | Same as above. | `174a3307556b3ecb` |
| `fig:trace-doorway-ego-occupancy` | Ego-frame proxemic read of the same pre-release doorway episode. | Use the same release-backed seed 1104 episode if the author accepts replacing the old example. | Same as above. | `174a3307556b3ecb` |
| `tab:execution-context-levels` | One predecessor doorway seed differs across contexts: release campaign, single-worker cluster node, login node. | For the selected release-source episode, the tested contexts were identical at the analysis-trace digest level: all collision, 33 steps, 3 near misses, same digest. This no longer supports the old cross-context divergence example. | `goal` / `classic_doorway_medium`, seed 1005; same host workers 8, same host workers 1, different host workers 8. | `a447a31e3c48d99c` for all three context repeats |
| `ll7/diss#3028` Ch6 doorway defect follow-up | The Ch6 doorway example depends on wall-law and pedestrian-radius artefacts. | Not gone in 0.0.8. Release notes still scope classic doorway rows as pedestrians held back by wall force, not doorway counterflow. | Use release-backed traces only with explicit limitation wording; do not turn doorway rows into planner ranking or door-width claims. | See Ch6 rows above |

## Preservation Contents

The hub bundle contains:

- `manifest.json`: source SHA, DOI, selected cells, dev seed range, Slurm job records, context repeats, and figure mapping.
- `selection/published_doorway_cell_rates.json`: published 0.0.8 doorway-cell outcome rates.
- `runs/classic_doorway_medium__goal__workers8/`: 200 allowed-dev-seed full-trace rows for the trace-backed replacement cell.
- `runs/classic_doorway_high__socnav_sampling__workers8/`: 200 allowed-dev-seed full-trace rows for the top-published-entropy doorway cell, documenting why it cannot render a success-vs-collision pair.
- `bundles/goal_medium_seed1004_success/`, `bundles/goal_medium_seed1005_collision/`, `bundles/goal_medium_seed1104_ped_collision/`: `trace_series_adapter.py` output for existing figure generators.
- `context/`: three seed-1005 context repeats, with summaries and raw rows.
- `run_scripts/`: Slurm scripts, telemetry config, and dev seed list.
- `SHA256SUMS`: read back successfully on `imech192`.

## Proposed Evidence Pins YAML

```yaml
- id: trace-doorway-release008-traces008b
  kind: trace-bundle
  repo: ll7/robot_sf_ll7
  commit: 373dbfde4f39667cf9e8732dabe7df5118bdeab1
  citable_paths:
    - external_data_hub/benchmark-results/robot_sf_ll7/traces008b-doorway-release-0.0.8-373dbfde-20261011
  covers_cards: [E-TRACE-DOORWAY-01]
  tier: worked-example
  rebase: rerun
  traces:
    doorway_goal_medium_seed1004_success:
      bundle: traces008b-doorway-release-0.0.8-373dbfde-20261011
      rel_path: bundles/goal_medium_seed1004_success
      sha256_16: "36dec9cb4d857e48"
      label: "Doorway goal success (release 0.0.8 source, dev seed 1004)"
      figures: [fig:trace-doorway-seedflip]
    doorway_goal_medium_seed1005_collision:
      bundle: traces008b-doorway-release-0.0.8-373dbfde-20261011
      rel_path: bundles/goal_medium_seed1005_collision
      sha256_16: "8870372b3f97d1f8"
      label: "Doorway goal collision (release 0.0.8 source, dev seed 1005)"
      figures: [fig:trace-doorway-seedflip, tab:execution-context-levels]
    doorway_goal_medium_seed1104_ped_collision:
      bundle: traces008b-doorway-release-0.0.8-373dbfde-20261011
      rel_path: bundles/goal_medium_seed1104_ped_collision
      sha256_16: "174a3307556b3ecb"
      label: "Doorway goal pedestrian collision (release 0.0.8 source, dev seed 1104)"
      figures: [fig:trace-doorway-scene, fig:trace-doorway-phase-portrait, fig:trace-doorway-ego-occupancy]
    doorway_goal_medium_seed1005_context_same_host_workers8:
      bundle: traces008b-doorway-release-0.0.8-373dbfde-20261011
      rel_path: context/classic_doorway_medium__goal__seed1005__workers8__auxme-imech091
      sha256_16: "a447a31e3c48d99c"
      label: "Doorway goal context repeat, same host, workers=8"
      figures: [tab:execution-context-levels]
    doorway_goal_medium_seed1005_context_same_host_workers1:
      bundle: traces008b-doorway-release-0.0.8-373dbfde-20261011
      rel_path: context/classic_doorway_medium__goal__seed1005__workers1__auxme-imech091
      sha256_16: "a447a31e3c48d99c"
      label: "Doorway goal context repeat, same host, workers=1"
      figures: [tab:execution-context-levels]
    doorway_goal_medium_seed1005_context_different_host_workers8:
      bundle: traces008b-doorway-release-0.0.8-373dbfde-20261011
      rel_path: context/classic_doorway_medium__goal__seed1005__workers8__auxme-imech093
      sha256_16: "a447a31e3c48d99c"
      label: "Doorway goal context repeat, different host, workers=8"
      figures: [tab:execution-context-levels]
  notes: >
    Release-0.0.8 re-execution at source 373dbfde4f39667cf9e8732dabe7df5118bdeab1
    on allowed development seeds 1001..1200, with analysis_trace: all and simulation
    step traces where supported. The old exact 15/15 PPO doorway-medium predecessor
    message is not supported by published 0.0.8 rows. The top published doorway
    entropy cell, socnav_sampling/classic_doorway_high, is preserved in the bundle
    but produced no success episodes on allowed dev seeds, so the trace-backed
    seed-flip replacement uses goal/classic_doorway_medium. Classic doorway rows
    remain limitation-scoped by the 0.0.8 wall-law disclosure.
```

## Author Decisions

The evidence supports regenerating release-backed doorway figures, but two manuscript messages need author choice:

- The old "exactly in half" seed-flip message should be replaced, not preserved. The clean trace-backed alternative is a 13/17 published split, not a 15/15 split.
- The execution-context table should either switch to the observed release-source repeatability result for this selected episode or be re-scoped to a separate search for a context-divergent 0.0.8 episode. This lane did not find divergence for the selected release-backed seed.

No code was changed in `robot_sf_ll7`; this is an analysis handoff plus external-data preservation bundle.
