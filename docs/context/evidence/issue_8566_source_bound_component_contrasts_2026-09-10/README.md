<!-- AI-GENERATED (robot_sf#8566, 2026-09-10) - NEEDS-REVIEW -->

# Issue #8566 source-bound component-contrast fixture packet

This packet is a deterministic, fixture-only implementation of the author-authorized
Issue #8566 preparation slice. It checks source identity, preserves release lineages,
computes paired scenario-block uncertainty for a six-cell synthetic fixture, and records
source-complete inputs that are unavailable. It does not run a campaign, hydrate an
external artifact, create an episode, or promote an empirical, benchmark, safety,
causal, planner-ranking, release, manuscript, or dissertation claim.

## Status and scope

- Packet status: `diagnostic_only`.
- Evidence status: `not_benchmark_evidence`.
- Fixture: three scenarios × two seeds, with `baseline` and `component_probe` arms.
- Components remain separate: terminal outcome, safety-wrapper rate, and Social Navigation Quality Index (SNQI) score.
- Missing, inactive, fallback, or degraded inputs are not imputed, pooled, or counted as success.

The numeric component rows are synthetic contract fixtures. Their effects and intervals are
useful only for testing the adapter's pairing, denominator, uncertainty, and multiplicity
fields; they are not observations about a planner, release, or corpus.

## Source and lineage binding

The generator validates the current bytes and records SHA-256 (Secure Hash Algorithm 256-bit)
digests for the config, fixture, generator, report, and release metadata. Release metadata is
also checked against the declared tracked commit.

| Source | Role | Identity | Fixture-slice status |
| --- | --- | --- | --- |
| `issue_8566_fixture_rows` | synthetic paired rows | tracked fixture digest | available for diagnostic contract checks |
| `august_b1d5_release_metadata` | August release anchor | source commit `b1d5ab6...`, release metadata tracked at the packet base | metadata only; not pooled with September |
| `september_59577_erratum_metadata` | September erratum metadata | source commit `59577bad...` | metadata only; source rows unavailable |
| `september_59577_recovery_metadata` | September recovery metadata | source commit `59577bad...` | metadata only; source rows unavailable |
| `issue_7980_exact_24_contrasts` | future exact #7980 source packet | immutable artifact identity recorded in config | unavailable; no external bytes consumed |

August `b1d5` and September `59577` are retained as separate lineages. The report emits no
cross-release pooled estimate. The exact #7980 contrasts and source-complete September rows
remain explicit unavailable coverage, with no substitute rows or zero denominators.

## Analysis contract

The frozen pairing key is:

```text
corpus_id, release_id, contrast_id, scenario_id, seed
```

The estimand is `comparison_minus_reference_mean`. Uncertainty uses a seeded,
paired scenario-block percentile bootstrap with 95% confidence and 2,000 replicates;
the scenario is the resampling cluster and the denominator is the declared pair-cell
count. The three available fixture components share the declared Holm step-down family;
the adjustment does not turn the fixture into benchmark evidence.

The config rejects campaign or compute authorization, paper-facing claims, and
fallback/degraded success. The generator fails closed on missing files, digest mismatches,
release-byte mismatches, invalid pair identities, missing component values, and unavailable
cells carrying numeric values.

## Durable files

- `configs/analysis/issue_8566_source_bound_uncertainty_component_contrasts.yaml`: frozen config.
- `tests/fixtures/issue_8566_source_bound_component_contrasts/paired_rows.json`: six synthetic paired rows.
- `robot_sf/benchmark/source_bound_component_contrasts.py`: canonical generator and validator.
- `docs/context/evidence/issue_8566_source_bound_component_contrasts_2026-09-10/report.json`: deterministic diagnostic report with provenance and claim boundary.
- `docs/context/evidence/issue_8566_source_bound_component_contrasts_2026-09-10/SHA256SUMS`: input/report receipt.
- `tests/analysis/test_issue_8566_source_bound_component_contrasts.py`: focused determinism and fail-closed tests.

## Reproduction

From the repository root:

```bash
uv run python scripts/analysis/build_issue_8566_source_bound_component_contrasts.py \
  --config configs/analysis/issue_8566_source_bound_uncertainty_component_contrasts.yaml \
  --output docs/context/evidence/issue_8566_source_bound_component_contrasts_2026-09-10/report.json \
  --receipt docs/context/evidence/issue_8566_source_bound_component_contrasts_2026-09-10/SHA256SUMS
sha256sum -c docs/context/evidence/issue_8566_source_bound_component_contrasts_2026-09-10/SHA256SUMS
uv run pytest -q tests/analysis/test_issue_8566_source_bound_component_contrasts.py
```

The report's `source_coverage` entries are expected to remain `unavailable` for #7980 and
the September source rows until independently supplied, authenticated, and reviewed inputs
exist. That future source-complete work is outside this fixture-only packet and remains gated
by the live decisions on Issues #7980, #6102, and the required independent review.
