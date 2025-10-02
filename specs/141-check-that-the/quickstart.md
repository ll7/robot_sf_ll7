# Quickstart — Verify Feature Extractor Training Flow

## Prerequisites
1. Activate the repository virtual environment (`uv sync && source .venv/bin/activate`).
2. Ensure submodules are initialized (`git submodule update --init --recursive`).
3. (macOS) Export `OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` before launching Python.
4. (Ubuntu RTX) Verify CUDA drivers are available (`nvidia-smi`).

## Default macOS Single-Thread Run
1. From the repository root, execute:
   ```bash
   uv run python scripts/multi_extractor_training.py --config configs/scenarios/multi_extractor_default.yaml
   ```
2. Confirm output under `./tmp/multi_extractor_training/<timestamp>-<run-id>/` with subdirectories per extractor.
3. Inspect `summary.json` for machine-readable metadata and `summary.md` for the Markdown synopsis.
4. Review logs to ensure each extractor finished with `status="success"` unless a validation warning explains a skip.

## Ubuntu RTX High-Performance Run
1. Duplicate the config and enable GPU workers (example YAML snippet):
   ```yaml
   worker_mode: vectorized
   num_envs: 4
   device: cuda
   ```
2. Launch the script:
   ```bash
   uv run python scripts/multi_extractor_training.py --config configs/scenarios/multi_extractor_gpu.yaml
   ```
3. Verify `hardware_profile` entries in `summary.json` list GPU model, CUDA version, and worker count.
4. Ensure the aggregated metrics include all extractors; failed runs should appear with `status="failed"` and explanatory notes.

## Validation Checklist
- [ ] Timestamped directory contains per-extractor checkpoints and logs.
- [ ] `summary.json` matches the schema in `contracts/training_summary.schema.json`.
- [ ] `summary.md` lists hardware differences and extractor outcomes.
- [ ] Skipped or failed extractors log actionable guidance in the console and Markdown summary.
