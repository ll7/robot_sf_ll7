# Benchmark Resume & Caching Layer

## Problem statement
Running benchmark batches repeatedly wastes time by recomputing episodes that already exist in the output JSONL. We need a deterministic resume mechanism to skip completed episodes and an optional caching layer for reuse across runs/configs.

## Goals
- Deterministic episode identity from inputs (scenario, seed, algo/config, sim params, SNQI weights, map, etc.).
- Resume: scan existing JSONL, build an index of completed episode_ids, and skip them before submission.
- Parallel-safe: compatible with `--workers > 1` (parent filters jobs; parent does all writing).
- Optional manifest for provenance and quick status checks.

Non-goals (initial):
- Cross-run global cache directory (can be added later).
- On-worker disk writes (we keep the parent as the single writer to simplify concurrency).

## Design overview

Episode identity:
- Define `episode_id = sha256(json.dumps(norm_metadata, sort_keys=True))[:16]` where `norm_metadata` captures the minimal, stable set of inputs that affect outcomes:
  - scenario spec (map id/path + scenario json with agents, goals)
  - seed
  - algo id + algo_config_path (and resolved hash of config content)
  - sim params: horizon, dt
  - SNQI: weights hash and baseline id
  - repo version or optional `run_tag`

Resume flow:
1) If `resume=True` and `out_path` exists, call `index_existing(out_path)` which reads JSONL line-by-line and collects `episode_id` from each valid record. Log malformed lines; continue.
2) Before submitting jobs to the pool, compute each job's `episode_id` and filter out those already present.
3) Proceed with existing parent-only JSONL append path. The parent still validates and writes per-record.

Manifest:
- Optional `out_path + ".manifest.json"` with:
  - `run_hash` (hash of run-level fixed params)
  - `created_at`, `updated_at`
  - `completed_count`, `failed_count`
  - `completed_ids_sample` (few ids for quick sanity)
  - `schema_version`

CLI & API:
- Python: `run_batch(..., resume: bool = True)`
- CLI: `--resume/--no-resume` (default: resume=True)

## API changes
- runner.run_batch signature adds `resume: bool = True`.
- CLI `run` and `baseline` subcommands gain `--resume/--no-resume`.
- No schema changes to per-episode JSONL; each record should include `episode_id` and the core metadata used to derive it (most likely already present; if not, minimally add `episode_id`).

## Edge cases
- Corrupted JSONL lines: ignore with a warning; do not crash resume.
- Output path from a different config: detect via optional `run_hash` mismatch and warn; still resume best-effort unless `--strict-resume` is set.
- Seed collisions: covered by hashing full metadata, not just seed.
- Parallel: parent filters jobs; no contention since parent remains the only writer.

## Implementation plan
1) Add `compute_episode_id(metadata_dict) -> str` in `robot_sf/benchmark/runner.py` (or a small `ids.py`).
2) Add `index_existing(out_path: Path) -> set[str]` that tolerantly scans JSONL and returns existing ids.
3) Thread `resume` into `run_batch`; if True, filter `jobs` by existing ids before sequential/parallel path.
4) CLI: add `--resume/--no-resume` default True to `run` and `baseline`; thread into code paths.
5) Tests:
   - Unit: build two fake episode records; verify `index_existing` and `compute_episode_id` stability.
   - Integration: run a tiny batch twice with `--resume`; second run appends zero lines.
   - Parallel: same as above with `--workers 2`.

## Future extensions (optional)
- Global `cache_dir` with per-episode blobs under `<cache_dir>/<run_hash>/<episode_id>.json` and a `merge` utility.
- `--strict-resume` to abort if `run_hash` mismatches.
- Periodic manifest updates and richer stats for resuming mid-run.

## Success criteria
- Re-running a completed batch with `--resume` performs no episode re-computation, both with 1 and N workers.
- No change to JSONL schema beyond adding an `episode_id` field if missing.
- No regressions in existing CLI/API behavior; tests remain green.
