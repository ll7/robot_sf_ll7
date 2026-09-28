# Issue #9431 corrected 0.0.7 worked-example traces

Plain-language summary: these six tracked trace bundles are issue #9431's
bounded worked-example slice regenerated from a commit that contains the exact
0.0.7 trace config. They are not a population estimate, planner ranking, or
causal ablation.

- source commit: `2e87f9de0523ef977bce19b40189ea0a74c61e13`
- aggregate benchmark release tag (not the trace source):
  `paper-matrix-v2-h600-s30-2026-09-07f7e8d43084de748915e1b1eb8b2a1603357c6e`
- trace job: Slurm `15724`, terminal `COMPLETED (0:0)`
- trace campaign: `issue9431_trace_18ep_0_0_7_2e87f9de0523_20260922`
- exact input config: `configs/benchmarks/issue_9431_trace_worked_examples_v007.yaml`
- input config SHA-256: `9d0ddf20842e1d959a1f7428a4efb998ccab48e33d992b51bceeabb5781d90b5`
- effective config hash: `b63b47930c13da87`
- scenario-matrix SHA-256:
  `03fc83302f707dd1b27c0fa81c4e45e36e8354a4413171d09365926f62bb5c2c`
- scenario/seed slice: `classic_head_on_corridor_medium` and
  `classic_group_crossing_medium`, seeds `22`, `23`, `24`
- producer `SHA256SUMS` SHA-256:
  `c29f6571eb3a14a76c28d6f4ffb21b9145135b698f8503194326c7d44c46512d`
- durable artifact:
  `wandb://ll7/robot_sf/campaign-issue9431_trace_18ep_0_0_7_2e87f9de0523_20260922:v0`
- preservation manifest digest:
  `sha256:6827dc8527528e8e81055d4ccb152fa67b3481e2be950d01dbf5b1856a8d68e5`
- independent full cold-restore report SHA-256:
  `f9529356b7018aa1dc2e5e5e6ff6e6aceca2232be4043446b9414391b0bca7ec`
- checked-in campaign provenance sidecar SHA-256:
  `c765b4839e1fca6a5c9589d12293ee2e190b884f8e599fe159437bd7944dd8b6`

The head-on figure uses ORCA seed 24 (completion) and social-force seed 23
(timeout): they are two requested worked-example seeds, not a matched-seed
counterfactual. The group-crossing figure uses goal and social-force seed 22.
Every bundle retains the upstream exporter claim boundary and its own checksums.
All 18 source rows contain nonempty simulation-step traces with selected and
applied actions plus force vectors. Planner-internal decision arrays are empty
by contract for these baseline planners and are not claimed. The earlier job
15716 package is superseded for public worked-example provenance; its preserved
bytes are not relabeled as job 15724 output. The CPU-only wrapper did not emit
startup/admission JSON inside the result tree. The separate admission,
submission-intent, watcher, terminal, run metadata, and execution-context
records bind the run; this receipt gap is why job 15724 remains diagnostic-only.
