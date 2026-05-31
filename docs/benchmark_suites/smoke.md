# Smoke Suite

```yaml
suite_id: policy_search_smoke
benchmark_track: policy_search_smoke
status: runnable_local_diagnostic
```

## Purpose

Run one policy-search candidate on the smallest planner sanity scenario. This is the first check
that a candidate config loads, produces commands, and writes policy-search diagnostics.

## Scenarios And Seeds

- Scenario config: `configs/scenarios/single/planner_sanity_simple.yaml`
- Scenario ID: `planner_sanity_simple`
- Seed set: fixed seed `111` from `configs/policy_search/funnel.yaml`
- Horizon: `80`
- Workers: `1`

## Eligible Planners

Any candidate in `docs/context/policy_search/candidate_registry.yaml` with a concrete
`candidate_config_path` and no missing local prerequisites. Learned or external candidates must
still satisfy their adapter/checkpoint availability contracts.

## Metrics

Success rate, collision rate, near-miss rate, termination reason counts, scenario exclusions,
scenario-family metrics, mean minimum distance when recorded, and mean speed.

## Canonical Command

```bash
uv run python scripts/validation/run_policy_search_candidate.py \
  --candidate <candidate_id> \
  --stage smoke \
  --output-dir output/policy_search/<candidate_id>/smoke/manual \
  --workers 1
```

## Expected Runtime

Usually under a few minutes on a warmed local checkout for lightweight classical candidates.
Learned checkpoints may take longer if artifact hydration is required.

## Claim Boundary

Smoke success is runnable-wiring evidence only. It is not benchmark-strength ranking evidence,
planner promotion, safety evidence, or paper-facing evidence.

## Caveats

Fallback, degraded, missing-checkpoint, unavailable, or failed rows must be reported as caveats or
exclusions. Do not count fallback-only execution as a successful smoke unless the suite is explicitly
measuring fallback behavior.
