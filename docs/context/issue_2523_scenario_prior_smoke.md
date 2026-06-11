# Issue #2523 Scenario-Prior Smoke

Issue: [#2523](https://github.com/ll7/robot_sf_ll7/issues/2523)
Status: proxy smoke evidence; not benchmark evidence.

Related: [#2501](https://github.com/ll7/robot_sf_ll7/pull/2501),
[#2479](https://github.com/ll7/robot_sf_ll7/issues/2479),
[issue_2479_real_trajectory_scenario_prior.md](issue_2479_real_trajectory_scenario_prior.md).

## Result

Issue #2523 asked for the first `scenario_prior.v1` smoke artifact from staged data if available,
or from a schema-compatible proxy fixture if real data remained blocked. Local SDD staging was not
available: `python scripts/tools/manage_external_data.py list` reported `sdd` as missing under
`output/external_data/sdd`.

This pass therefore promotes a compact proxy artifact:

- [scenario_prior.v1.json](evidence/issue_2523_scenario_prior_smoke/scenario_prior.v1.json)
- [summary.json](evidence/issue_2523_scenario_prior_smoke/summary.json)

## Classification

The representation is `proxy_schema_adequate_for_smoke`: it exercises the #2479 feature groups
for scene identity, normalization, agent population, kinematics, interaction geometry, and prior
weights. It is not adequate as real-data evidence, training input, benchmark evidence, or a
realism claim.

## Claim Boundary

Do not cite this artifact as learned-prior realism, cross-dataset generalization, benchmark
usefulness, planner improvement, or license-safe redistribution of raw data. The next valid step is
to stage one license-approved SDD annotations tree and regenerate the same artifact shape from
importer outputs.

## Validation

```bash
uv run pytest tests/research/test_scenario_distribution_prior_manifest.py tests/research/test_issue_2523_scenario_prior_smoke.py -q
uv run ruff check tests/research/test_issue_2523_scenario_prior_smoke.py
bash scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check
```
