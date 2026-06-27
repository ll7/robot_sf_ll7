# Issue #3204 — proxy-based checkpoint-selection readiness preflight

**Status:** blocked (inputs not available locally) · **Evidence tier:** diagnostic-only
**Related:** #3307 (merged analyzer), epic #3215, #3213/#3214/#3254 (plateau-breaking lineages)

## What this slice adds

A read-only, fail-closed **preflight** that gates the inputs of the proxy-vs-ADE
checkpoint-selection analysis. The analyzer itself (`scripts/research/analyze_predictive_checkpoint_proxy.py`)
already merged via #3307; the conclusive run is run-gated (`resource:slurm`). This slice does **not**
attempt that run — it operationalizes the manual "checkpoints not available" diagnostic into a
reusable check so the blocked state is machine-checkable and the revival condition is explicit.

- `configs/research/predictive_checkpoint_proxy_v1.yaml` — declarative readiness contract: hard-seed
  fixture, checkpoint selector (`registry_tag: predictive`, `min_resolvable_checkpoints: 6`),
  candidate proxy signals, and the training-summary contract.
- `scripts/research/check_predictive_checkpoint_proxy_readiness.py` — preflight that:
  - resolves each `predictive` registry checkpoint's `local_path` presence (reusing the canonical
    `robot_sf.models.registry.load_registry`; no download), and
  - optionally judges a training summary by reusing the merged analyzer's verdict (an
    `inconclusive`/no-spread summary, e.g. the all-zero `hardcase_proxy_probe_v1` probe, fails
    closed).
  - Exit `0` (`ready`) only when all provided inputs resolve; exit `2` (`blocked`) otherwise.
- `tests/research/test_check_predictive_checkpoint_proxy_readiness.py` — fixture tests for the
  fail-closed boundary plus a pin on the live-registry blocked state.

## Current observed state (live registry)

All 8 `predictive_*` registry entries resolve to absent, non-durable `output/tmp/...` paths in a
clean checkout, so the preflight reports `blocked` with `resolvable_count = 0 < 6`. This matches the
prior manual finding on the issue.

## Revival condition

Promote/hydrate ≥ 6 predictive checkpoints to a durable, locally-resolvable store (e.g. W&B artifacts
pinned in `model/registry.yaml`), **and** make a training summary available whose `proxy.history`
shows non-degenerate hard-seed `success_rate` spread (from a plateau-breaking lineage). When the
preflight flips to `ready`, the merged analyzer can produce the Spearman(proxy, success)-vs-ADE
comparison and the A/B result the issue tracks.

## Out of scope (unchanged here)

No checkpoint selection, no training, no Slurm/GPU submission, no evidence promotion, and no
paper/dissertation claim edits.

## Reproduce

```bash
uv run python scripts/research/check_predictive_checkpoint_proxy_readiness.py --json
uv run pytest tests/research/test_check_predictive_checkpoint_proxy_readiness.py -q
```
