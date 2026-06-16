# Issue #2659 Manifest Lineage Schema Unification

Issue: #2659

## Purpose

`scenario_prior.v1`, `adversarial_scenario_manifest.v1`, and
`counterfactual_scenario_pair.v1` now expose a common lineage/evidence-boundary contract through
`robot_sf.benchmark.manifest_lineage.ManifestLineageContract`.

This is schema and reviewability work only. It does not convert any artifact into benchmark,
planner-promotion, or paper-grade evidence.

## Mandatory Fields Before Execution

The shared contract requires these fields before a manifest-like artifact can be treated as ready
for downstream execution or review:

- `source`: input artifact/config provenance as a mapping.
- `generator_id`: stable producer identifier.
- `validator_version`: validator or fixture-validator contract version.
- `schema_version`: manifest schema identifier.
- `claim_boundary`: explicit claim or non-claim boundary.
- `evidence_tier`: current evidence status, such as `diagnostic-only` or `smoke evidence`.
- `denominator_policy`: whether rows may enter benchmark denominators.
- `execution_gate`: the condition that must hold before execution or downstream use.

## Family Mapping

`scenario_prior.v1` remains a JSON-only proxy smoke artifact. Its tracked fixture declares:

- `generator_id`: `issue_2523_proxy_scenario_prior_fixture`
- `validator_version`: `scenario_prior_proxy_fixture_validator.v1`
- `evidence_tier`: `smoke evidence`
- `denominator_policy`: `proxy_fixture_not_benchmark_denominator`
- `execution_gate`: `not_executable_until_real_data_staged`

`adversarial_scenario_manifest.v1` now serializes the shared fields from its existing source,
generator, execution status, and evidence boundary. Its denominator policy is
`generated_candidates_not_benchmark_denominator`.

`counterfactual_scenario_pair.v1` now serializes source provenance, generator/validator IDs,
evidence tier, denominator policy, and preflight-based execution gate before returning the pair
payload.

## Compatibility Notes

Existing family-specific fields remain in place. The shared contract is intentionally additive so
older consumers can continue reading `source`, `generator`, `claim_boundary`, `evidence_boundary`,
`perturbation_manifest`, and `preflight` fields where they already existed.

The shared helper validates field presence and basic types; it does not prove a manifest is
executable, benchmark-valid, or complete for paper-facing claims. Downstream benchmark claims still
need executable evidence, denominator review, and fail-closed handling.

## Issue #2906 Backfill Proxy Metadata Update 2026-06-16

Issue #2906 extends manifest-lineage backfill entries with structured proxy metadata:
`candidate_sources`, `conflicting_sources`, and `blocked_by`. The graph builder uses these fields
for proxy-source edges instead of parsing human-readable `reason` text, while retaining the reason
string for reader context and legacy fallback.

This remains reviewability and artifact-quality work only. It hardens graph topology against wording
changes, but it does not upgrade any manifest, graph row, or artifact candidate into benchmark or
paper-facing evidence.

## Issue #2905 Node ID Collision Safety 2026-06-16

Issue #2905 refactors the lineage graph builder into smaller typed construction stages and makes
manifest, lineage-field, proxy-source, and artifact-candidate node IDs collision-safe. Distinct
inputs that previously sanitized to the same base string (for example ``a/b`` and ``a__b``) now
receive distinct node IDs through a compact deterministic hash suffix.

The public graph schema is unchanged; only internal ``id`` values differ. Lineage-field node IDs
keep the field name as the final segment so that adjacency and label lookups remain readable.
