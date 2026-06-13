# Dissertation Gap Report

Issue: [#2784](https://github.com/ll7/robot_sf_ll7/issues/2784)
Status: synthesis/planning aid; not new benchmark, paper, dissertation, or safety evidence.

## Purpose

This gap report merges the dissertation evidence ledger (#2760) and the negative-result
register (#2762) into a single four-bucket classification. It is a **synthesis/planning aid**,
not new benchmark evidence, paper-facing results, or safety claims.

The report classifies every ledger row and register entry into one of four buckets:
- **supported**: release-backed and current (only observation_robustness qualifies today)
- **blocked**: has a promotion path but is not yet release-backed
- **negative_revise_only**: diagnostic, revise, or inconclusive findings
- **remove_weaken**: stale, non-claimable, or failed entries

## Reading Guide

- **Bucket**: where the source row sits today. A bucket does not upgrade the source.
- **Promotion step or reason**: the concrete next step (from ledger or register) or `None`
  if no credible path exists. A non-null path does not upgrade the row.
- **Allowed wording / boundary**: verbatim from the source. No new wording is introduced.
- **Caveat**: verbatim from the source. Must appear alongside any use.
- **Claim gap / reason**: what is still missing, verbatim from the source.

## Sources

| Source | Issue | Artifact |
|---|---|---|
| Dissertation evidence ledger | [#2760](https://github.com/ll7/robot_sf_ll7/issues/2760) | `docs/context/evidence/issue_2760_dissertation_evidence_ledger/ledger.json` |
| Negative-result register | [#2762](https://github.com/ll7/robot_sf_ll7/issues/2762) | `docs/context/evidence/issue_2762_negative_result_register/register.json` |

## Output Artifacts

| Artifact | Path |
|---|---|
| Machine-readable gap report | `docs/context/evidence/issue_2784_dissertation_gap_report/gap_report.json` |
| Human-readable gap report | `docs/context/evidence/issue_2784_dissertation_gap_report/gap_report.md` |
| Generator script | `scripts/tools/generate_dissertation_gap_report.py` |
| Validation tests | `tests/docs/test_dissertation_gap_report.py` |

## Claim Boundaries

- This gap report is a **synthesis/planning aid**. It does not produce new benchmark
  evidence, paper-facing results, or safety claims.
- All `allowed_wording` and `caveat` fields are copied verbatim from source rows.
  No new wording is introduced.
- A non-null `promotion_step_or_reason` does not upgrade a row to stronger evidence.
  Promotion requires completing the path and reclassifying the evidence tier.
- Fallback behavior is not acceptable as a successful benchmark outcome unless the task
  explicitly measures fallback mode.
- Every gap classification preserves the source evidence tier and classification without
  upgrade.

## Validation

```bash
# Generate report
uv run python scripts/tools/generate_dissertation_gap_report.py \
  --json-output docs/context/evidence/issue_2784_dissertation_gap_report/gap_report.json \
  --markdown-output docs/context/evidence/issue_2784_dissertation_gap_report/gap_report.md

# Run gap report tests
uv run pytest tests/docs/test_dissertation_gap_report.py -q

# Cross-validate with existing tests
uv run pytest tests/docs/test_dissertation_evidence_ledger.py tests/docs/test_negative_result_register.py -q

# Lint
uv run ruff check scripts/tools/generate_dissertation_gap_report.py tests/docs/test_dissertation_gap_report.py
uv run ruff format scripts/tools/generate_dissertation_gap_report.py tests/docs/test_dissertation_gap_report.py
```
