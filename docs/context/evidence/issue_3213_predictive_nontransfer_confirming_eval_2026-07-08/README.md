# Issue #3213 Predictive-Planner Non-Transfer Confirming-Eval Packet (2026-07-08)

Pinned evidence packet for the predictive-planner **non-transfer** confirming-eval bundle behind
the `#3254`-era closed-loop finding. Created by [issue #4879](https://github.com/ll7/robot_sf_ll7/issues/4879)
to durably archive the checkpoint-eval plateau (closed-loop success `0.0667`-`0.1` across four
populated prediction checkpoints) versus the `0.30` success gate, with full config + seed provenance.

This packet follows the campaign-bundle pattern of
[`issue_1554_job_13198_constraints_first_analysis`](../issue_1554_job_13198_constraints_first_analysis/README.md):
a self-contained directory with `packet.json` (machine provenance), a narrative `README.md`, an
`artifact_inventory.json` with checksums, the underlying data, and a claim-decision note.

## What this pins

The closed-loop confirming-eval grid from the `#3213` maneuver-authority sweep (PR
[#3306](https://github.com/ll7/robot_sf_ll7/pull/3306)), which evaluated five planner-authority
variants across five predictive-planner checkpoints on the `predictive_hardcase_portfolio_v1`
crossing-conflict scenarios. Before this packet, the grid lived only as a bare
[`robust_grid.json`](../issue_3213_authority_sweep/robust_grid.json) with no pinned provenance; this
packet pins an identical byte-copy and adds the provenance, claim boundary, and references the
downstream assessment docs need to make stronger statements about the non-transfer result.

## Provenance

- Producing PR: [#3306](https://github.com/ll7/robot_sf_ll7/pull/3306), commit
  `95adcc3ae69620f67b42603bdfdde9504b99d9b1` (2026-06-21).
- Child issue: [#3213](https://github.com/ll7/robot_sf_ll7/issues/3213) (maneuver-authority sweep);
  parent issue: [#3215](https://github.com/ll7/robot_sf_ll7/issues/3215) (hard-case portfolio);
  era issue: [#3254](https://github.com/ll7/robot_sf_ll7/issues/3254) (predictive crossing-conflict
  training line, `predictive_ego_v1`).
- Scenario set: `configs/scenarios/sets/predictive_hardcase_portfolio_v1.yaml`
  (`classic_cross_trap_low/medium/high`, `classic_group_crossing_high`).
- Planner-authority grid: `configs/benchmarks/predictive_hardcase_authority_grid_issue_3213.yaml`.
- Closed-loop seeds: broad robust schedule `200-229` (robust_a `200-214` on `LiCCA test 9832460`,
  robust_b `215-229` on `imech192 a30 12969`), ~120 episodes/cell. This is distinct from the small
  `predictive_hard_seeds_v1.yaml` hard-seed set; both are recorded in `seed_provenance.yaml`.
- Full seed/scenario/checkpoint provenance: [`seed_provenance.yaml`](seed_provenance.yaml).

## Headline answer (plateau vs gate)

Populated closed-loop success-rate cells range `0.0667`-`0.1`:

- Baseline authority across the four populated checkpoints: `0.0667`-`0.0833`.
- Near-field speed cap (`nf_speedcap_only`) and `nearfield_turn`: `0.0917`-`0.1` (only active knobs;
  `+~0.03` absolute / `+~33%` relative, consistent across checkpoints).
- Inert knobs (about baseline): `nf_headings_only`, `nf_horizonboost_only`.
- Required success gate: `0.30`.

Neither checkpoint choice nor planner authority closes the gap → binding constraint is
model/data-side → the predictive-planner **non-transfer** result. See
[`negative_finding.md`](negative_finding.md) for the full decision and claim boundary.

The `#3254` final-eval point `0.08696` (bundle
[`issue_3254_predictive_crossing_conflict_13042_2026-06-23`](../issue_3254_predictive_crossing_conflict_13042_2026-06-23/README.md))
sits inside this plateau, so that single-point run is not an outlier.

## Recoverable-artifact negative (per issue #4879)

One checkpoint, `sweep_h256_mp4_s42`, returned zero episodes for all five authority variants
(`success_rate_mean=null`, `episodes_approx=0`). That genuinely-empty fifth checkpoint is the one
unrecoverable cell in the grid; it is documented here and in `packet.json`
(`source_artifacts.unpopulated_checkpoint`) rather than silently dropped. A second checkpoint,
`sweep_h192_mp3_s7_wd5e5`, has two partial cells (~60 episodes, robust_a half only) for
`nf_horizonboost_only` and `nearfield_turn`.

## Files

- [`packet.json`](packet.json): machine-readable provenance, methodology, plateau summary,
  non-transfer finding, checksums, and forbidden-actions confirmation.
- [`robust_grid.json`](robust_grid.json): pinned byte-identical copy of the source
  confirming-eval grid (sha256 `bde1a7eb…`, matches the source at
  `../issue_3213_authority_sweep/robust_grid.json`).
- [`confirming_eval_plateau.csv`](confirming_eval_plateau.csv): flat checkpoint × authority-variant
  success-rate table with per-seed-half values and populated flags.
- [`seed_provenance.yaml`](seed_provenance.yaml): seed schedule, scenario set, planner grid, hard-seed
  manifest, checkpoint population status, and gate definition.
- [`negative_finding.md`](negative_finding.md): claim decision, plateau vs gate, what the result
  confirms / does not support, and the recommended next action.
- [`artifact_inventory.json`](artifact_inventory.json): file list with sha256, sizes, and kinds.

## Claim Boundary

`diagnostic_only`. This packet confirms the predictive-planner non-transfer plateau across
checkpoints and authority knobs. It is **not** benchmark-strength planner evidence, not a safety
claim, not a model promotion, and not paper-facing evidence.

## Validation

```bash
python -m json.tool docs/context/evidence/issue_3213_predictive_nontransfer_confirming_eval_2026-07-08/packet.json
python -m json.tool docs/context/evidence/issue_3213_predictive_nontransfer_confirming_eval_2026-07-08/artifact_inventory.json
# Confirm the pinned grid is byte-identical to the source:
sha256sum docs/context/evidence/issue_3213_predictive_nontransfer_confirming_eval_2026-07-08/robust_grid.json \
          docs/context/evidence/issue_3213_authority_sweep/robust_grid.json
uv run python scripts/validation/check_docs_proof_consistency.py \
  --path docs/context/evidence/issue_3213_predictive_nontransfer_confirming_eval_2026-07-08/README.md \
  --path docs/context/evidence/issue_3213_predictive_nontransfer_confirming_eval_2026-07-08/packet.json \
  --path docs/context/evidence/issue_3213_predictive_nontransfer_confirming_eval_2026-07-08/artifact_inventory.json \
  --path docs/context/evidence/issue_3213_predictive_nontransfer_confirming_eval_2026-07-08/seed_provenance.yaml \
  --path docs/context/evidence/issue_3213_predictive_nontransfer_confirming_eval_2026-07-08/negative_finding.md \
  --path docs/context/evidence/issue_3213_predictive_nontransfer_confirming_eval_2026-07-08/confirming_eval_plateau.csv \
  --path docs/context/negative_result_register.md \
  --path docs/context/issue_3213_maneuver_authority_setup.md \
  --path docs/context/issue_3254_predictive_crossing_conflict_negative_result.md \
  --check-evidence-catalog
```
