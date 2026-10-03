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

## Before the freeze: #10112 source contract

[#10112](https://github.com/ll7/robot_sf_ll7/issues/10112) **blocks the freeze**.
Its scoring, smoke and ordering changes must land on main before the orchestrator
names the freeze commit. This ruling supersedes the earlier preparation audit's
classification of #10112 as a mint-only blocker. Historical manifests remain
unchanged; regenerate every identity/packet at the newly named clean source.

D-083 now declares `snqi_v2_spec` with the weights, family, pending anchor asset,
and `calibration.dev1001_1002_scheduled_acquisition.yaml`, each hash-bound.
Both v0.2 templates pin those same assets. The source anchor file remains
`pending_calibration`; loading a configuration is permitted for identity and
checkpoint preparation, but executing an unscored bound campaign refuses.
The doorway template selects its v2 campaign successor; its original v1 config
and concrete manifest remain historical bytes.

Acquisition freezes a separate artifact after the freeze, without editing the
tracked pending file or moving the named source. The release runner accepts
`--snqi-v2-anchors <artifact>` and checks the strict frozen loader, dev1001/1002
split, calibration source and exact acquisition configuration identity. Supply
`--snqi-v2-calibration-root <complete raw root>` to additionally rederive and
compare the anchors from every producer row/sidecar. Independent scientific
pins remain mandatory in production; these checks confer no scientific authority.

## Two-phase mint: preparatory rows, smoke, final campaign

1. Land the public and private #10112 successors, validate clean main, then name
   the freeze. The separately authorized DOI operator reserves unpublished
   coordinates before generating the production main and doorway identities.
2. Stage calibration and smoke checkpoints at that source. Use the tracked
   `configs/benchmarks/paper_experiment_matrix_v2_h600_s30_runtime_smoke_v0_6.yaml`
   and its same-named manifest under `configs/benchmarks/releases/`.
   This is 14 current D-083 arms × blind corner × dev seed **1003**, dt=0.1,
   differential drive, authored H400. v0_2 and v0_5 are historical contracts.
   The preparatory mint requires the smoke staging receipt at
   `output/release_identity/runtime_smoke_checkpoint_staging_receipt.json`.
3. Mint **proposed preparation rows**, not a full campaign, using the reviewed
   private successor's existing tool:

   ```bash
   "$OPS/.venv/bin/python" "$OPS/ops/jobs/scripts/mint_snqi_v2_release_rows.py" \
     --public-root "$PUBLIC_ROOT" --public-sha "$FREEZE_SHA" \
     --private-ops-root "$OPS" --private-ops-commit "$OPS_SHA" \
     --date "$MINT_DATE" --out-dir "$ARTIFACT_ROOT/preparation-mint"
   ```

   Inputs are clean exact source checkouts, staged smoke checkpoints, current
   acquisition/smoke contracts, fresh identities and durable destinations.
   Outputs are proposed calibration/smoke rows and hash-bound packets, go=false.
   An independent exact-head review precedes `--admit --review-ref <verdict>`;
   the existing queue writer/readiness gate and canonical driver still own
   promotion/submission. Neither a diagnostic plan nor preparation grants sealed
   evaluation admission. Retain the proposed bytes and regenerate for review.
4. Run the calibration acquisition on **1001/1002**, using the canonical camera
   runner and `analyze_snqi_contract.py --campaign-root <root>
   --freeze-v2-anchors <artifact>`. Preserve all 1,344 nonfallback rows,
   sidecars, input hashes, command-mode census and F/J/K p95; no imputation.
   Complete the independent checklist below before production scientific mint.
5. Run the tracked v0_6 smoke through `run_benchmark_release.py --manifest
   <tracked v0_6 manifest> --checkpoint-receipt <staged receipt>`.
   Obtain separate authentic environment admission, 70-cell stress acceptance,
   checkpoint/staging, preservation and independent cold-readback receipts.
   All must bind the same freeze and remain under 24 hours old where required.
   Successful smoke alone is not environment/stress/scientific admission.
6. Generate/verify both production identities with the real reserved coordinates;
   stage full main and doorway checkpoints. Now run the existing private
   `mint_snqi_v2_full_campaign.py --request <request.json> --out-dir <fresh dir>
   --report <report.json>`. Supply both identities/checkpoints, smoke result,
   environment admission, stress/cold custody, reviewed anchors and sealed-seed
   ruling. Outputs remain proposed/non-dispatchable until independently admitted.
   The two-track executor forwards the exact `snqi_anchors` input to both public
   runners. No diagnostic mint can supply any missing admission.
7. The separately admitted owner runs main then doorway on one node, followed by
   separate acceptance/bundles, the D-062 comparator and final two-bundle review.
   Publication/tag/DOI actions require their separate authorization.

## Independent scientific review required before production mint

Leave these boxes unchecked here. An independent reviewer completes them after
actual acquisition; this integration and a rehearsal cannot fill the trust set.

- [ ] Review the complete dev1001/1002 14×48×2 acquisition custody, actual producer
  hashes/sidecars, native/adapter/mixed census, absence of fallback/degraded rows,
  force-source decision, schema/zero anchors, T=3, N=0.25 and real positive F/J/K p95.
- [ ] Bind the calibration source to the named freeze. Verify zero metric/runtime,
  planner/model, physics, schema and authored-budget drift between acquisition and
  campaign; keep the sealed seed commitment and the D-084 overtaking H600.
- [ ] Independently review anchor applicability to the fixed doorway width slice;
  retain its separate scientific and publication boundary (CHAIN-4 G12).
- [ ] Review the sealed evaluation ruling binding the freeze, concrete main and
  companion manifests/configs, canonical acquired anchor digest, exact sealed
  tuple and review reference. No request-created receipt can authenticate it.
- [ ] In a separate reviewed private code change, add the actual
  `REVIEWED_SCIENTIFIC_SOURCES` entry: `freeze_sha`, `review_ref`, and both
  `snqi_anchors` / `evaluation_seed_admission` sources, each with repository,
  relative path, full source commit and SHA-256. Test positive controls on the
  actual pinned Git blobs and negative controls on changed bytes/source/splits.
- [ ] Re-run production mint's strict loader and all scientific/source/custody
  gates at the exact admitted public/private revisions. The trust set is empty
  until that independently reviewed change lands; retain the refusal meanwhile.

## Rehearsing acquisition and scored packaging (D-086)

At a clean rehearsal source, use the same acquisition file on dev1001/1002,
freeze anchors into ignored output, then generate the D-086 seed-1001 preparatory
smoke and seed-1001/1002/1003 campaign identities described above. The preparatory
D-086 smoke may remain unscored and is permanently diagnostic. Before the scored
campaign wrapper, set `ROBOT_SF_SNQI_V2_ANCHORS=<artifact>` and
`ROBOT_SF_SNQI_V2_CALIBRATION_ROOT=<complete acquisition root>`; the wrapper forwards
both as explicit runner inputs. The runner revalidates acquisition custody,
computes `snqi_v2`/terms and exports the shared diagnostic bundle. Inspect the
actual 2,016 scored rows and bundle, retaining `release_eligible: false`.
Development anchors/receipts never admit a later source or a sealed campaign.
