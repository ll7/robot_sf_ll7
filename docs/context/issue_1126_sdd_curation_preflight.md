# Issue #1126 SDD Curation Readiness Preflight (2026-06-27)

Status: Current. Scope: the *curation-step* readiness gate only. This note does **not** record a
real-data curation run; issue #1126 stays `state:blocked-external-input` until licensed SDD
annotations are staged (issue #1497, validated by #2413).

## Why this exists

Issue #1126 curates the first real SDD-derived benchmark scenario set *after* a licensed annotation
source is staged. While external data is blocked, the only safe work is a **manifest/schema
preflight** that decides whether a curation run may be promoted as benchmark evidence, and that fails
closed otherwise. This preflight is the curation-step gate; it is distinct from:

- the SDD **staging** gate — owned by `scripts/tools/manage_external_data.py`
  (`load_sdd_staging_spec`, `validate_sdd_staging`, `resolve_sdd_scenario_prior_mode`), manifest
  `configs/data/sdd_staging_manifest.yaml` (issues #1497 / #2413 / #3473);
- the SDD **importer** — `scripts/tools/import_sdd_scenarios.py` (issue #1091).

## What landed

- `scripts/tools/sdd_curation_preflight.py` — a thin per-issue runner that *composes* the canonical
  owners (it does not re-derive staging or parsing):
  - `resolve_sdd_scenario_prior_mode(...)` decides `dataset_backed_prior` vs `proxy_schema_smoke`;
  - `import_sdd_scenarios.load_sdd_points(...)` probes a candidate annotation file against the
    deterministic curation selection rule (enough usable tracks after `lost`/label filtering) with
    **no** scenario/map output written.
- `tests/tools/test_sdd_curation_preflight.py` — fixture-backed unit tests for the fail-closed
  contract.

## Hard contract (fail-closed)

- `benchmark_promotion_allowed` is True **only** when SDD is staged AND checksum-validated
  (`dataset_backed_prior`) AND the candidate annotation satisfies the selection rule.
- A parseable **fixture** or an unpinned/unvalidated staged copy can be *probed* (`curation_runnable`
  may be True for a schema smoke), but its evidence stays `proxy_schema_smoke` and
  `output_classification` never reaches `benchmark_ready_candidate`. Synthetic/fixture rows are never
  promoted as benchmark evidence.
- `proxy_schema_smoke` and `dataset_backed_prior` remain strictly distinct in the report.

## Validation (this checkout, no SDD staged)

```bash
uv run python scripts/tools/import_sdd_scenarios.py --help
uv run python scripts/tools/sdd_curation_preflight.py            # -> proxy_schema_smoke, promotion False
uv run python -m pytest tests/tools/test_sdd_curation_preflight.py -q   # 8 passed
uv run python -m pytest tests/ -k "sdd or import_scenarios" -q          # 53 passed
```

The CLI on this checkout reports `staging mode: proxy_schema_smoke`, `dataset_backed: False`,
`benchmark promotion: False` — i.e. it correctly refuses to treat the blocked state as
benchmark-ready. `--require-benchmark-ready` exits non-zero (3) so callers fail closed.

## Real-data path (when #1497 stages licensed annotations)

1. Stage licensed SDD annotations and validate via `manage_external_data.py` (`sdd-validate`) so
   `resolve_sdd_scenario_prior_mode` reports `dataset_backed_prior`.
2. Run the preflight against the selected staged annotation file:
   `uv run python scripts/tools/sdd_curation_preflight.py --annotation <staged>/annotations.txt
   --label Pedestrian --min-track-points 8 --require-benchmark-ready` (probe must be satisfiable).
3. Only then run `scripts/tools/import_sdd_scenarios.py` and decide `benchmark_ready` vs
   `exploratory_only` after the smoke run, recording source identity, checksums, license/URL, scale
   assumptions, and command (per the issue acceptance criteria).

## Out of scope here

No SDD download/ingestion, no real curation run, no benchmark campaign, no Slurm/GPU submission, and
no paper/dissertation claim edits.
