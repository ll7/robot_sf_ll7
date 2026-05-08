# Issue 1051 Camera-Ready Evidence Provenance Audit

Date: 2026-05-07

Related issues:

* `ll7/robot_sf_ll7#1051`
* Follow-up already filed: `ll7/robot_sf_ll7#1053`

## Goal

Audit whether the camera-ready paper evidence trail is complete and durable for PPO provenance,
benchmark command/configs, seed/bootstrap evidence, SNQI diagnostics, and release inputs. This is a
documentation/provenance audit only; it does not rerun benchmarks or reinterpret metrics.

## Current Verdict

The release contract is mostly reconstructable from committed configs and docs, but the evidence
trail is not yet publication-archive complete.

Strong surfaces:

* release manifest, campaign config, scenario matrix, seed policy, and SNQI v3 assets are tracked;
* the release manifest hash-pins the canonical campaign config, scenario matrix, and SNQI assets;
* the canonical PPO baseline points at a W&B artifact path through `model/registry.yaml`;
* seed/bootstrap policy is explicit in `configs/benchmarks/paper_experiment_matrix_v1.yaml`.

Gaps:

* the current tracked May 4 evidence bundle is partial and explicitly not publication-ready;
* `docs/context/evidence/camera_ready_all_planners_2026-05-04/` does not include
  `reports/snqi_diagnostics.json`;
* several context notes still cite publication bundles only as local `output/` paths;
* the canonical PPO registry entry still has `commit: null`, a local `local_path`, and one note
  pointing at a local `output/model_cache/.../best_summary.json` file.

Use issue `#1053` to repair durable artifact references and archive pointers before treating this
as a complete paper evidence trail.

## Evidence Checklist

| Evidence item | Current source | Status | Audit note |
|---|---|---|---|
| Release manifest | `configs/benchmarks/releases/paper_experiment_matrix_v1_release_v0_1.yaml` | present | Hash-pins release config, scenario matrix, SNQI weights, and SNQI baseline. |
| Canonical benchmark command | `docs/benchmark_release_protocol.md` and `docs/benchmark_release_reproducibility.md` | present | Both point at `scripts/tools/run_benchmark_release.py --manifest configs/benchmarks/releases/paper_experiment_matrix_v1_release_v0_1.yaml`. |
| Campaign config | `configs/benchmarks/paper_experiment_matrix_v1.yaml` | present | Paper profile, seed set, bootstrap settings, SNQI assets, and planner matrix are explicit. |
| Scenario matrix | `configs/scenarios/classic_interactions_francis2023.yaml` | present | SHA-256 matches the release manifest. |
| SNQI weights | `configs/benchmarks/snqi_weights_camera_ready_v3.json` | present | SHA-256 matches the release manifest. |
| SNQI baseline | `configs/benchmarks/snqi_baseline_camera_ready_v3.json` | present | SHA-256 matches the release manifest. |
| Seed policy | `configs/benchmarks/seed_sets_v1.yaml` and `configs/benchmarks/paper_experiment_matrix_v1.yaml` | present | Canonical S3 `eval` seed set and bootstrap settings are explicit. |
| Seed/bootstrap evidence | `docs/context/issue_832_paper_matrix_extended_seed_schedule.md` | present with caveat | Records S3/S5 execution and comparison, but output paths are worktree-local. |
| PPO baseline config | `configs/baselines/ppo_15m_grid_socnav.yaml` | present | Points at the issue-791 model id and includes paper-claim caveats. |
| PPO model registry | `model/registry.yaml` | partial | Has W&B run/artifact pointer, but `commit: null`, `local_path` points at worktree-local `output/model_cache/...`, and the `Best summary` note points at local `output/model_cache/.../best_summary.json`. Both non-portable registry fields should be folded into the #1053 follow-up. |
| SNQI v3 contract note | `docs/context/issue_635_snqi_v3_paper_contract.md` | present with caveat | Provides the contract, but its canonical publication bundle paths are local `output/` paths. |
| Publication bundle workflow | `docs/benchmark_camera_ready_release.md` | present | Defines GitHub release upload, checksums, and manifest validation. |
| Tracked compact evidence | `docs/context/evidence/camera_ready_all_planners_2026-05-04/` | partial | Useful internal evidence, but `benchmark_success=false` and publication bundle was skipped. |
| Tracked SNQI diagnostics for May 4 bundle | `docs/context/evidence/camera_ready_all_planners_2026-05-04/reports/snqi_diagnostics.json` | missing | The source run reported SNQI pass, but this compact bundle does not carry the diagnostics JSON required by the release manifest. |
| Durable publication archive URL | release asset / DOI URL in current paper evidence | missing | Existing docs mostly cite local bundle paths or placeholders; #1053 should replace these with durable archive pointers. |

## Hash Check

The tracked release-manifest hashes match the current files:

| File | SHA-256 |
|---|---|
| `configs/benchmarks/paper_experiment_matrix_v1.yaml` | `a91bc8f3903a25939538270c0bf5fc29c3216ec40ded370a601616c2e8ed5a2d` |
| `configs/scenarios/classic_interactions_francis2023.yaml` | `d9e148e4b544b4c7e2b6ba98e599aef47046d114e0e25645f021946674cb9dc5` |
| `configs/benchmarks/snqi_weights_camera_ready_v3.json` | `71a67c3c02faff166f8c96bef8bcf898533981ca2b2c4493829988520fb1aeb2` |
| `configs/benchmarks/snqi_baseline_camera_ready_v3.json` | `329ca5766491e1587979d0a435c7ba676e148ccdff97040a36546bbb9414035a` |

Command:

```bash
rtk sha256sum \
  configs/benchmarks/paper_experiment_matrix_v1.yaml \
  configs/scenarios/classic_interactions_francis2023.yaml \
  configs/benchmarks/snqi_weights_camera_ready_v3.json \
  configs/benchmarks/snqi_baseline_camera_ready_v3.json
```

## Path Check

Checked required source files and the compact May 4 evidence bundle. All source docs/configs were
present, but the compact bundle was missing the tracked SNQI diagnostics JSON:

```text
missing docs/context/evidence/camera_ready_all_planners_2026-05-04/reports/snqi_diagnostics.json
```

The missing file matters because the release manifest lists `reports/snqi_diagnostics.json` as a
required campaign artifact and because SNQI status alone is weaker evidence than the diagnostic
payload with weights, baseline, rank alignment, outcome separation, and dominance fields.

## Follow-Up Boundary

No benchmark rerun is required by this audit alone. The next paper-critical action is issue `#1053`:
replace or supplement local `output/` publication-bundle references with durable archive pointers,
fold the `model/registry.yaml` `local_path` and `Best summary` note fields into the same
durable-reference repair so the PPO checkpoint carries no worktree-local fallback, and include the
required SNQI diagnostics payload when promoting any compact evidence bundle as paper support.
