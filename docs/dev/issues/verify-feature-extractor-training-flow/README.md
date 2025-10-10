# Verify Feature Extractor Training Flow

## Specification linkage
- Source spec: `specs/141-check-that-the/spec.md`
- Design artifacts: `plan.md`, `research.md`, `data-model.md`, `contracts/`, `quickstart.md`
- Constitution alignment: reproducible summaries, helper reuse in `robot_sf/`, Loguru-first logging

## Objectives
- Guarantee `scripts/multi_extractor_training.py` runs end-to-end on macOS (single-thread spawn) and Ubuntu RTX (vectorized workers).
- Persist each run under `./tmp/multi_extractor_training/<timestamp>-<run-id>/` with per-extractor subdirectories, JSON, and Markdown summaries.
- Capture reusable helpers for run metadata, hardware probing, and summary rendering inside `robot_sf/training/`.
- Document workflows and configs so researchers can repeat macOS and GPU quickstarts without modifications.

## Key deliverables
- Dataclasses and helpers under `robot_sf/training/` covering configurations, run records, hardware probes, paths, and summary writers.
- Refactored training script that records run metadata, handles skips/failures gracefully, and invokes the summary pipeline.
- Contract-driven tests (JSON schema, markdown structure, integration smoke) plus hardware probe unit tests.
- Updated configs, docs, and changelog entries referencing the new workflow.

## Validation checkpoints
- Contract/unit suites green: JSON + Markdown summary validation and hardware probe tests.
- Integration passes on single-thread (macOS) and vectorized (GPU) modes or skip with explicit reasons in CI.
- Quickstart instructions verified manually with archived sample outputs.
- Repository lint/tests clean prior to merge.

## CLI usage
Run the comparison script with the provided configs to generate timestamped summaries:

```bash
uv run python scripts/multi_extractor_training.py \
  --config configs/scenarios/multi_extractor_default.yaml \
  --run-id mac-smoke

# GPU/vectorized flow (skips automatically when CUDA is unavailable)
uv run python scripts/multi_extractor_training.py \
  --config configs/scenarios/multi_extractor_gpu.yaml \
  --run-id gpu-batch
```

Each execution writes `summary.json`, `summary.md`, and a compatibility `complete_results.json` alongside per-extractor artifacts under `tmp/multi_extractor_training/<timestamp>-<run-id>/`.
