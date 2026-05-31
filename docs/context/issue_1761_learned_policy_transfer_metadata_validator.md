# Issue #1761 Learned-Policy Transfer Metadata Validator - 2026-05-30

Date: 2026-05-30

Related issue:

- Issue #1761: <https://github.com/ll7/robot_sf_ll7/issues/1761>

Related anchors:

- Issue #1627 learned-policy transfer benchmark design:
  <https://github.com/ll7/robot_sf_ll7/issues/1627>
- Benchmark fallback policy:
  `docs/context/issue_691_benchmark_fallback_policy.md`
- Learned-policy adapter interface:
  `docs/context/issue_1618_learned_policy_adapter_interface.md`

## Decision

The first machine-checkable learned-policy transfer benchmark surface is a metadata-only validator
for objects attached at:

```yaml
algorithm_metadata:
  transfer_benchmark:
    schema_version: learned_policy_transfer_benchmark.v1
```

The schema lives at:

- `robot_sf/benchmark/schemas/learned-policy-transfer-metadata.schema.v1.json`

The Python validator lives at:

- `robot_sf/benchmark/learned_policy_transfer_metadata.py`

This validator checks structure and fail-closed status semantics only. It does not hydrate model
artifacts, import external policies, run source repositories, add adapters, or change benchmark
runner/report-writer behavior.

## Required Contract

The v1 object requires:

- candidate identity: `transfer_candidate_id`, `policy_family`, `source_kind`,
  `paper_or_source`, and `local_assessment_note`,
- artifact provenance status under `artifact_provenance.artifact_manifest_status`,
- observation and action contracts,
- `transfer_stage`,
- `execution_mode`,
- `readiness_status`,
- `availability_status`,
- `availability_reason`,
- `benchmark_success`,
- at least one `evidence_pointers` entry.

`benchmark_success=true` is accepted only when the metadata is success-capable:

- `execution_mode` is `native` or `adapter`,
- `readiness_status` is `native` or `adapter`,
- `availability_status` is `available`,
- `transfer_stage` is `robot_sf_smoke` or `transfer_benchmark`,
- `artifact_provenance.artifact_manifest_status` is `complete` or `not_required`.

Fallback, degraded, failed, partial-failure, and not-available metadata can validate as records, but
they must keep `benchmark_success=false`.

## Fixtures

The initial fixtures are:

- `tests/fixtures/learned_policy_transfer_benchmark/v1/ppo_issue791_best_v1.json`:
  native Robot SF PPO metadata with success-capable statuses. The fixture points to the candidate
  and baseline configs but does not hydrate the model artifact.
- `tests/fixtures/learned_policy_transfer_benchmark/v1/crowdnav_height_blocked_source.json`:
  source-first CrowdNav HEIGHT metadata using the Issue #1394 blocker evidence. It validates as
  `availability_status=not_available` and `benchmark_success=false`.

## Validation

Validation commands:

```bash
uv run pytest tests/benchmark/test_learned_policy_transfer_metadata.py -q
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check origin/main...HEAD
```
