# Hybrid ORCA Sampler Smoke Validation (2026-04-29)

## Scope

Validate the newly added policy-search workflow end to end on a cheap local stage and capture the result in markdown plus structured artifacts.

## Commands

```bash
source .venv/bin/activate && uv run pytest tests/validation/test_policy_search_common.py
source .venv/bin/activate && uv run pytest tests/validation/test_run_policy_search_candidate.py
source .venv/bin/activate && uv run pytest tests/planner/test_risk_dwa_mppi_hybrid.py
source .venv/bin/activate && uv run python scripts/validation/run_policy_search_candidate.py --candidate hybrid_orca_sampler_v1 --stage smoke
source .venv/bin/activate && uv run python scripts/tools/build_policy_search_failure_report.py --summary-json output/policy_search/hybrid_orca_sampler_v1/smoke/latest/summary.json --output docs/context/policy_search/validation/hybrid_orca_sampler_v1_smoke_failure_report
```

## Test Results

- `tests/validation/test_policy_search_common.py`: `5 passed`
- `tests/validation/test_run_policy_search_candidate.py`: `4 passed`
- `tests/planner/test_risk_dwa_mppi_hybrid.py`: `16 passed`

## Smoke Result

- Candidate: `hybrid_orca_sampler_v1`
- Stage: `smoke`
- Episodes: `3`
- Success rate: `0.0000`
- Collision rate: `0.0000`
- Near-miss rate: `0.0000`
- Failure taxonomy: `timeout_low_progress=3`
- Decision emitted by the runner: `pass`

## Artifact Paths

- Summary JSON: `output/policy_search/hybrid_orca_sampler_v1/smoke/latest/summary.json`
- Runner report: `docs/context/policy_search/reports/2026-04-29_hybrid_orca_sampler_v1_smoke.md`
- Failure report JSON: `docs/context/policy_search/validation/hybrid_orca_sampler_v1_smoke_failure_report/failure_report.json`
- Failure report markdown: `docs/context/policy_search/validation/hybrid_orca_sampler_v1_smoke_failure_report/failure_report.md`

## Interpretation

The important result is infrastructure-level, not quality-level: the candidate registry, path resolution, stage runner, benchmark integration, and report generation all worked on a real benchmark execution path.

The planner itself is still too conservative for promotion. On the sanity slice it stayed collision-free, but it failed to complete the route on every seed and was classified as `timeout_low_progress` each time. That means the next local iteration should tune ORCA-to-sampler switching and progress thresholds before any nominal-sanity or stress-slice stage is meaningful.
