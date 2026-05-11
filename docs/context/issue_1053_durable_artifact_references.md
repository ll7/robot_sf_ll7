# Issue 1053 Durable Artifact Reference Audit

Date: 2026-05-07

Related issues:

* `ll7/robot_sf_ll7#1053`
* Follow-up blocker: `ll7/robot_sf_ll7#1062`

## Goal

Check whether claim-bearing model, benchmark, SNQI, and release artifacts can be recovered from
committed pointers and durable artifact stores rather than from this worktree's local `output/`
tree.

## Current Verdict

> 2026-05-09 update: issue `#1062` records a durable publication archive pointer for the scoped
> seven-planner release `0.0.2`. Keep the original audit below as the May 7 state before that
> pointer was verified.

The model side has a durable pointer; the scoped `0.0.2` publication-bundle side now has a durable
GitHub release and DOI pointer. The all-planners May 4 compact evidence remains partial.

* The current paper-facing PPO checkpoint has a W&B artifact pointer in `model/registry.yaml`.
* The benchmark release inputs are tracked and hash-pinned in
  `configs/benchmarks/releases/paper_experiment_matrix_v1_release_v0_1.yaml`.
* SNQI v3 weights and baseline assets are tracked in `configs/benchmarks/`.
* Local `output/model_cache/` and `output/benchmarks/publication/` paths remain common in context
  notes and should be treated as cache/history unless paired with a W&B artifact, release asset,
  DOI, or other durable manifest.
* No bulky generated outputs were promoted during this audit.
* Issue `#1062` later recorded the durable scoped release archive, embedded manifest, embedded
  checksums, and SNQI diagnostics in
  `docs/context/issue_1062_paper_evidence_archive.md` and
  `docs/experiments/publication/20260414_benchmark_release_0_0_2/`.

## Artifact Inventory

| Artifact class | Example source | Durable state | Action |
|---|---|---|---|
| PPO checkpoint | `model/registry.yaml` entry `ppo_expert_issue_791_reward_curriculum_eval_aligned_large_capacity_20260417` | durable model artifact present via W&B `wandb_artifact_path` | keep W&B artifact as source; treat `output/model_cache/...` as cache only. |
| PPO source commit | same registry entry | partial | `commit: null`; leave visible as a provenance caveat until exact training commit is recovered. |
| Benchmark release config | `configs/benchmarks/paper_experiment_matrix_v1.yaml` | durable | tracked and hash-pinned by the release manifest. |
| Scenario matrix | `configs/scenarios/classic_interactions_francis2023.yaml` | durable | tracked and hash-pinned by the release manifest. |
| SNQI weights/baseline | `configs/benchmarks/snqi_weights_camera_ready_v3.json`, `configs/benchmarks/snqi_baseline_camera_ready_v3.json` | durable | tracked and hash-pinned by the release manifest. |
| SNQI diagnostics | `reports/snqi_diagnostics.{json,md}` inside campaign/publication bundles | durable for scoped `0.0.2` | Recover from the `0.0.2` release archive; see `docs/context/issue_1062_paper_evidence_archive.md`. |
| Publication bundle archive | `output/benchmarks/publication/<bundle_name>.tar.gz` | durable for scoped `0.0.2` | GitHub release asset plus DOI pointer recorded by `#1062`; local output paths remain cache/history. |
| Compact evidence bundles | `docs/context/evidence/` | partial durable summaries | acceptable for small review evidence, not a replacement for the full publication bundle. |

Date suffixes embedded in artifact identifiers and bundle names follow the `YYYYMMDD` format (for
example `20260417` in the PPO checkpoint id). Future-dated entries such as `2026...` are
intentional and reflect planned or future-dated promotion.

## Local Output Check

Command required by the issue:

```bash
rtk git status --ignored --short -uall output
```

Result: the local `output/` tree contains ignored autoresearch logs, coverage output, benchmark
traces, model caches, and other generated artifacts. None were staged or committed. This is expected
for local validation, but these paths must not be the sole source for paper evidence.

## Documentation Changes

This audit updates:

* `model/registry.md` to define durable W&B artifact pointers versus local cache fields;
* `model/registry.yaml` to mark the issue-791 PPO local paths as cache references;
* `docs/benchmark_release_reproducibility.md` to state that local release-output paths must be
  paired with durable release or artifact-store pointers before paper handoff.

## Follow-Up Boundary

Issue `#1062` resolves the archive-publication blocker for the scoped seven-planner release `0.0.2`
by verifying the durable release URL, DOI, archive checksum, embedded publication manifest,
embedded checksums, and embedded SNQI diagnostics. The all-planners surface remains separate because
`socnav_bench` still depends on external licensed assets.
