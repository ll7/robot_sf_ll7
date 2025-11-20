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

## 2. Tail the live progress (status/watch)

```bash
uv run python scripts/tools/run_tracker_cli.py status output/run-tracker/demo_run
uv run python scripts/tools/run_tracker_cli.py watch output/run-tracker/demo_run --interval 1.0
```
`status` prints the latest per-step durations, ETA accuracy, and rule-based recommendations. `watch` refreshes every second so you can monitor long steps without re-running the command.

## 3. (Optional) Stream to TensorBoard

```bash
uv run python scripts/tools/run_tracker_cli.py enable-tensorboard \
  output/run-tracker/demo_run \
  --logdir output/run-tracker/tb/demo_run
uv run tensorboard --logdir output/run-tracker/tb
```

The adapter mirrors metrics to TensorBoard without altering the canonical JSON artifacts.

## 4. Run performance smoke tests

```bash
uv run python scripts/tools/run_tracker_cli.py perf-tests \
  --scenario configs/validation/minimal.yaml \
  --output output/run-tracker/perf-tests/latest \
  --num-resets 5
```

The CLI wraps `scripts/validation/performance_smoke_test.py`, records throughput + pass/soft-breach/fail status, and stores `perf_test_results.json` plus any recommendations next to the tracker manifest.

## 5. Review historical runs

```bash
uv run python scripts/tools/run_tracker_cli.py list --limit 10
```

You can filter by status, date, or scenario to answer "what happened last night" quickly.

## 6. Summarize or export run reports

```bash
uv run python scripts/tools/run_tracker_cli.py summary output/run-tracker/demo_run --format markdown
uv run python scripts/tools/run_tracker_cli.py export output/run-tracker/demo_run --format markdown --output output/run-tracker/demo_run/summary.md
```

`summary` (alias `show`) prints recommendations, telemetry aggregates, and artifact links inline. `export` writes the same data to JSON or Markdown so you can attach the report to docs, issues, or release notes.
