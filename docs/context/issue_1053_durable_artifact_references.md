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

The model side has a durable pointer; the publication-bundle side is still blocked.

* The current paper-facing PPO checkpoint has a W&B artifact pointer in `model/registry.yaml`.
* The benchmark release inputs are tracked and hash-pinned in
  `configs/benchmarks/releases/paper_experiment_matrix_v1_release_v0_1.yaml`.
* SNQI v3 weights and baseline assets are tracked in `configs/benchmarks/`.
* Local `output/model_cache/` and `output/benchmarks/publication/` paths remain common in context
  notes and should be treated as cache/history unless paired with a W&B artifact, release asset,
  DOI, or other durable manifest.
* No bulky generated outputs were promoted during this audit.

The missing paper-critical source is a durable publication bundle and diagnostics pointer. Issue
`#1062` now tracks publishing or recording that archive, checksums, manifest, and SNQI diagnostics.

## Artifact Inventory

| Artifact class | Example source | Durable state | Action |
|---|---|---|---|
| PPO checkpoint | `model/registry.yaml` entry `ppo_expert_issue_791_reward_curriculum_eval_aligned_large_capacity_20260417` | durable model artifact present via W&B `wandb_artifact_path` | keep W&B artifact as source; treat `output/model_cache/...` as cache only. |
| PPO source commit | same registry entry | partial | `commit: null`; leave visible as a provenance caveat until exact training commit is recovered. |
| Benchmark release config | `configs/benchmarks/paper_experiment_matrix_v1.yaml` | durable | tracked and hash-pinned by the release manifest. |
| Scenario matrix | `configs/scenarios/classic_interactions_francis2023.yaml` | durable | tracked and hash-pinned by the release manifest. |
| SNQI weights/baseline | `configs/benchmarks/snqi_weights_camera_ready_v3.json`, `configs/benchmarks/snqi_baseline_camera_ready_v3.json` | durable | tracked and hash-pinned by the release manifest. |
| SNQI diagnostics | `reports/snqi_diagnostics.{json,md}` inside campaign/publication bundles | blocked for current paper archive | compact evidence bundles may include diagnostics, but the current paper bundle still needs a durable archive pointer. |
| Publication bundle archive | `output/benchmarks/publication/<bundle>.tar.gz` | blocked | local path only unless uploaded; tracked by follow-up `#1062`. |
| Compact evidence bundles | `docs/context/evidence/` | partial durable summaries | acceptable for small review evidence, not a replacement for the full publication bundle. |

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

Issue `#1062` is the blocker for archive publication. It should verify every new durable URL or
artifact pointer and keep raw episode JSONL, videos, large logs, coverage, and model checkpoints out
of git unless a maintainer explicitly chooses a small fixture.

