# Quickstart: Performance Tracking & Telemetry

Follow these steps to try the new tracker end-to-end. Commands assume you already ran `uv sync && source .venv/bin/activate`.

## 1. Enable the tracker on the imitation pipeline

```bash
uv run python examples/advanced/16_imitation_learning_pipeline.py \
  --enable-tracker \
  --tracker-output output/run-tracker/demo_run
```

What happens:
- Tracker prints "Step X/Y" with ETA updates to stdout.
- A manifest directory is created under `output/run-tracker/demo_run/` containing `manifest.jsonl`, `steps.json`, and `telemetry.jsonl`.

## 2. Tail the live progress

```bash
uv run python scripts/tools/run_tracker_cli.py status --run output/run-tracker/demo_run
```
Outputs the latest per-step durations, ETA accuracy, and rule-based recommendations.

## 3. (Optional) Stream to TensorBoard

```bash
uv run python scripts/tools/run_tracker_cli.py enable-tensorboard \
  --run output/run-tracker/demo_run \
  --logdir output/run-tracker/tb/demo_run
uv run tensorboard --logdir output/run-tracker/tb
```

The adapter mirrors metrics to TensorBoard without altering the canonical JSON artifacts.

## 4. Run performance smoke tests

```bash
uv run python scripts/telemetry/run_perf_tests.py \
  --scenario configs/validation/minimal.yaml \
  --output output/run-tracker/perf-tests/latest
```

The command executes the existing performance smoke script, records measured throughput, compares it to baselines, and writes `perf_test_results.json` plus any recommendations.

## 5. Review historical runs

```bash
uv run python scripts/tools/run_tracker_cli.py list --limit 10
```

You can filter by status, date, or scenario to answer "what happened last night" quickly.
