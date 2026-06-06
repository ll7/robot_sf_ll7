# Issue #2477 ScenarioBelief Contract Bridge (2026-06-06)

Status: contract bridge to existing MVP, not benchmark evidence.

Related surfaces:

- GitHub issue: https://github.com/ll7/robot_sf_ll7/issues/2477
- Parent roadmap issue: https://github.com/ll7/robot_sf_ll7/issues/2469
- Predecessor design: https://github.com/ll7/robot_sf_ll7/issues/1966
- Contract manifest: `configs/representation/scenario_belief_contract_issue_2477.yaml`
- Implementation: `robot_sf/representation/scenario_belief.py`
- Existing tests: `tests/representation/test_scenario_belief.py`
- Manifest tests: `tests/representation/test_scenario_belief_contract_manifest.py`

## Result

Issue #2477 asked for a sensor-independent scenario belief representation direction. The main
representation already exists from the #1966/#2105/#2157 line as `ScenarioBelief`. This pass adds
a #2477-specific manifest that makes the existing contract discoverable from the current research
roadmap and pins the issue checklist to concrete repository surfaces.

The manifest records:

- required top-level belief, entity, and estimate fields;
- producer ownership for simulator-oracle and visibility-limited simulator adapters;
- consumer ownership for `SOCNAV_STRUCT`, debug JSON, and diagnostic summary projections;
- sensor-independent assumptions and non-goals;
- the first fixture/smoke path that demonstrates shared schema and stable policy projection keys.

## Claim Boundary

This is interface and diagnostic evidence only. It does not prove real-sensor calibration, generality
across all sensors, planner improvement, or benchmark performance. The manifest should guide the
next representation work without allowing ScenarioBelief rows to be treated as benchmark-strength
results.

## Validation

Targeted validation:

```bash
uv run pytest tests/representation/test_scenario_belief.py \
  tests/representation/test_scenario_belief_contract_manifest.py -q
uv run ruff check robot_sf/representation tests/representation/test_scenario_belief_contract_manifest.py
uv run ruff format --check robot_sf/representation tests/representation/test_scenario_belief_contract_manifest.py
uv run python scripts/validation/check_docs_proof_consistency.py --path docs/context/catalog.yaml
git diff --check
```

## Follow-Up Boundary

Issue #2478 should extend this contract toward explicit uncertainty-aware scenario belief fields,
especially class, pose, velocity, and covariance semantics. New sensor adapters or planner
consumers should wait until those semantics are pinned or explicitly documented as diagnostic-only.
