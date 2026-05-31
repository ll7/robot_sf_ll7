# Issue #1878 Head-On Route Replay Determinism

Date: 2026-05-31

Related issues:

- <https://github.com/ll7/robot_sf_ll7/issues/1878>
- <https://github.com/ll7/robot_sf_ll7/issues/1488>
- <https://github.com/ll7/robot_sf_ll7/issues/1502>
- <https://github.com/ll7/robot_sf_ll7/issues/1861>

## Scope

This note records a bounded follow-up for the #1502 `classic_head_on_corridor` guided-route row.
The selected route was already present in compact #1502 evidence, but the generated
`route_overrides.yaml` path was local-only. This slice promotes that route payload to a tracked
fixture and runs a two-pass replay determinism check from tracked inputs.

This is development stress-test evidence only. It is not paper-facing benchmark evidence and does
not broaden the Issue #1502 or Issue #1488 campaign claims.

## Durable Inputs

- Route fixture:
  `configs/scenarios/route_overrides/issue_1878/classic_head_on_corridor_low_guided_1502.yaml`
- Replay matrix:
  `configs/scenarios/sets/issue_1878_head_on_replay.yaml`
- Source compact summary:
  `docs/context/evidence/issue_1502_adversarial_two_family_2026-05-31/classic_head_on_corridor_guided_summary.json`
- Determinism summary:
  `docs/context/evidence/issue_1878_head_on_replay_2026-05-31/head_on_replay_determinism_summary.json`

The fixture's `route_payload` matches the #1502 summary exactly. The replay matrix uses the current
benchmark CLI `goal` policy as the local replay mapping for the frozen
`classic_global_theta_star` row and records native execution metadata. Treat that as a replay
mechanism check, not as a broad planner-quality claim.

## Result

The two replay runs matched on the stable replay signature:

- scenario: `issue_1878_classic_head_on_corridor_low_guided_1502`
- seed: `123`
- horizon: `500`
- algo: `goal`
- status: `collision`
- termination: `collision`
- steps: `335`
- stable signature SHA-256: `360057895f1e6f5a04fa045141ae26b961184367a8a3d854e7437f52f48dc78a`

The raw JSONL files have different raw SHA-256 values because volatile run metadata is present, so
they remain ignored under `output/`. The compact tracked summary records the raw hashes and the
stable deterministic fields.

## Validation

Commands:

```bash
uv run pytest tests/test_scenario_loader_route_overrides.py::test_issue_1878_head_on_route_override_matches_issue_1502_summary tests/test_scenario_loader_route_overrides.py::test_issue_1878_head_on_replay_matrix_resolves_tracked_override -q
TF_CPP_MIN_LOG_LEVEL=2 LOGURU_LEVEL=WARNING uv run robot_sf_bench validate-config --matrix configs/scenarios/sets/issue_1878_head_on_replay.yaml
uv run ruff check tests/test_scenario_loader_route_overrides.py
TF_CPP_MIN_LOG_LEVEL=2 LOGURU_LEVEL=WARNING uv run robot_sf_bench --quiet run --matrix configs/scenarios/sets/issue_1878_head_on_replay.yaml --algo goal --out output/adversarial/issue_1878/head_on_replay_a.jsonl --horizon 500 --workers 1 --no-resume --no-video --structured-output json
TF_CPP_MIN_LOG_LEVEL=2 LOGURU_LEVEL=WARNING uv run robot_sf_bench --quiet run --matrix configs/scenarios/sets/issue_1878_head_on_replay.yaml --algo goal --out output/adversarial/issue_1878/head_on_replay_b.jsonl --horizon 500 --workers 1 --no-resume --no-video --structured-output json
python -m json.tool docs/context/evidence/issue_1878_head_on_replay_2026-05-31/head_on_replay_determinism_summary.json
```

Observed results:

- focused fixture tests: `2 passed`
- config validation: exit `0`, with an intentional one-seed warning
- replay A/B: both exit `0`, native/available, `1` written episode each
- stable signature comparison: matched

## Limits

- The fixed replay matrix intentionally has one seed; it proves deterministic replay for this row,
  not coverage across seeds.
- Force-based metrics were excluded from the stable signature because the replay emitted missing
  pedestrian force data warnings.
- Issue #1488 still needs a remote issue update or PR text after this branch is reviewed, because this
  worker did not push or open a PR.
