<!-- AI-GENERATED (robot_sf#7032) - NEEDS-REVIEW -->
# Issue #7032 Mechanism-Boundary Atlas

Plain-language summary: this atlas records what six negative-result case families can and cannot
support, without turning diagnostic evidence into benchmark or dissertation claims.

## Claim Boundary

This packet is a diagnostic planning atlas only. It is not benchmark evidence, paper evidence,
dissertation admission, a visual-rendering packet, or packet-backed caption evidence. The builder
does not infer evidence from issue prose, filenames, plots, or ignored `output/` artifacts. Source
references are either tracked repository files with verified SHA-256 digests or explicit
`blocked`/`unavailable` placeholders.

Each card keeps the question and hypothesis, structured code/config identity, mechanism activation
evidence, observed result, scope and estimand, contradicted versus still-viable hypotheses, and
dissertation admission status explicit. Path identities carry repository-relative SHA-256 records;
commit identities carry full commit IDs and repository scope; non-file digests must be present in a
verified source manifest. Result state uses the issue-controlled vocabulary (`supported_positive`,
`supported_negative`, `not_supported`, `inconclusive`, `invalid_evidence_contract`, or
`unavailable`) and is validated separately from the mechanism/evidence-boundary dimension. Each
card also carries one or more exact #7032 `boundary_labels`; labels may be combined when a case
has both a mechanism and an evidence-durability boundary.

Visual rendering, PDF/PNG production, packet-backed captions, and dissertation-facing admission are
deferred or blocked in the generated cards.

## Files

- `atlas_input.v1.json`: checked-in source manifest for the first six #7032 case cards.
- `atlas.v1.json`: deterministic generated atlas emitted by the builder.

## Validation

```bash
uv run python scripts/analysis/build_negative_result_mechanism_atlas.py
uv run pytest -q tests/benchmark/test_mechanism_boundary_atlas.py
uv run ruff check robot_sf/benchmark/mechanism_boundary_atlas.py scripts/analysis/build_negative_result_mechanism_atlas.py tests/benchmark/test_mechanism_boundary_atlas.py
git diff --check
```
