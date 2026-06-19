# Issue #2917 ScenarioPrior.v1 Provenance Cards (2026-06-19)

Status: partial initial registry, not benchmark evidence.

Related surfaces:

- GitHub issue: https://github.com/ll7/robot_sf_ll7/issues/2917
- Card registry: `configs/research/scenario_prior_cards_issue_2917.yaml`
- Scenario-prior scoping manifest: `configs/research/scenario_distribution_prior_issue_2479.yaml`
- Proxy smoke artifact: `docs/context/evidence/issue_2523_scenario_prior_smoke/scenario_prior.v1.json`
- Scenario contract docs: `docs/scenario_contracts.md`

## Result

This pass adds the first `scenario-prior-card-registry.v1` surface for the known prior families
already visible in the repository:

- authored scenario contracts and perturbation pilots;
- the #2523 proxy `scenario_prior.v1` smoke fixture;
- the #2479 SDD scenario-distribution candidate;
- repository-trace-derived adversarial/search-generated scenario prior family;
- repository-trace-derived counterfactual scenario-pair mechanism prior family.

The registry records source type, license/provenance status, source traces, extraction method,
parameter-bound source, excluded populations, unsupported claims, and ODD conditions for each
family. The two new repository-trace-derived families are diagnostic or smoke-level only;
they do not constitute benchmark evidence, causality evidence, or real-world representativeness.

## Claim Boundary

This is proposal/interface evidence only. It does not prove learned-prior realism,
benchmark usefulness, cross-dataset generalization, planner performance improvement, or
license-safe redistribution of raw data.

The SDD card is deliberately a candidate gate: the repo has a license-aware staging/import path, but
raw SDD annotations were not staged for this pass. The proxy card remains proxy-only and is not a
training input or benchmark denominator.

## Remaining Work

Future #2917 slices should add cards for repository-trace-derived priors once an extraction tool
exists, and external-dataset-derived cards only after source, license, checksum, coordinate frame,
and excluded-population provenance are tracked.

## Validation

Targeted validation for this slice:

```bash
uv run pytest tests/research/test_issue_2917_scenario_prior_cards.py -q
git diff --check origin/main...HEAD
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
```
