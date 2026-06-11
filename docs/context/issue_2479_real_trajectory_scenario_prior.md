# Issue #2479 Real-Trajectory Scenario Prior Scope (2026-06-06)

Status: scoped manifest contract, not benchmark evidence.

Update 2026-06-11 (#2523): SDD was not staged locally for the first smoke pass, so #2523 produced
a proxy `scenario_prior.v1` artifact instead of real-data prior evidence. See
[issue_2523_scenario_prior_smoke.md](issue_2523_scenario_prior_smoke.md).

Related surfaces:

- GitHub issue: https://github.com/ll7/robot_sf_ll7/issues/2479
- Parent roadmap issue: https://github.com/ll7/robot_sf_ll7/issues/2469
- Scenario-prior manifest: `configs/research/scenario_distribution_prior_issue_2479.yaml`
- Existing SDD importer: `scripts/tools/import_sdd_scenarios.py`
- External-data assistant: `scripts/tools/manage_external_data.py`
- SDD importer note: `docs/context/issue_1091_sdd_importer.md`
- Real-world trajectory import docs: `docs/real_world_trajectory_import.md`
- First proxy smoke: `docs/context/issue_2523_scenario_prior_smoke.md`

## Result

Issue #2479 asks whether Robot SF can learn a realistic scenario prior from real trajectory
datasets. This pass stops at the scoping boundary: it records candidate datasets, provenance gates,
the scenario-prior representation fields, and the first non-training spike without staging raw data
or making a realism claim.

The manifest prioritizes Stanford Drone Dataset (SDD) because the repository already has a
license-aware local staging contract and a narrow importer that writes map, scenario, and
provenance outputs. ETH/UCY-style, TrajNet-style, and road-user interaction datasets remain
follow-up candidates until source, license, coordinate-frame, and domain-fit checks are recorded.

## Claim Boundary

This is proposal and interface evidence only. It does not prove that learned priors are realistic,
benchmark-useful, cross-dataset general, or performance-improving. Raw external datasets remain
local-only unless a dedicated artifact/provenance decision says otherwise.

## First Spike

The next useful implementation issue should stage one local SDD annotation tree through
`scripts/tools/manage_external_data.py`, run `scripts/tools/import_sdd_scenarios.py`, and extract a
compact `scenario_distribution_prior_manifest.v1` from generated scenario/provenance outputs. The
compact feature manifest should live under `docs/context/evidence/`; raw annotations should stay
ignored and local.

Suggested fields:

- dataset identity, source file/provenance pointer, license/access note, and checksum;
- normalization metadata such as meters-per-pixel, frame rate, axis policy, and map extent;
- agent-population, kinematic, and interaction-geometry summaries;
- scenario-prior bins for encounter type, density, route endpoints, and temporal windows.

## Validation

Targeted validation:

```bash
uv run pytest tests/research/test_scenario_distribution_prior_manifest.py -q
uv run ruff check tests/research/test_scenario_distribution_prior_manifest.py
uv run ruff format --check tests/research/test_scenario_distribution_prior_manifest.py
uv run python scripts/validation/check_docs_proof_consistency.py --path docs/context/catalog.yaml
git diff --check
```

## Follow-Up Boundary

Do not extend importer statistics or train a prior in this scoping PR. The recommended follow-up is
a small SDD prior-manifest spike that enriches importer output or adds a tiny reader over generated
SDD scenario/provenance files, then proves the manifest shape on staged local data.
