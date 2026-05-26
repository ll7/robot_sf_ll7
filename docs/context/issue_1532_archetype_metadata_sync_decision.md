# Issue #1532 Archetype Metadata Sync Decision

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1532>

Date: 2026-05-26

## Decision

Issue-body archetype metadata remains the source of truth.

No immediate broad synchronization is recommended. A future sync path, if implemented, should:

- default to dry-run/report mode,
- stay issue-scoped or explicitly batched,
- mirror only low-risk labels that already exist or are explicitly provisioned,
- keep Project #5 writes opt-in and separate from issue-body or label cleanup,
- avoid backlog-wide mutation by default.

This keeps the canonical metadata block from
[`issue_1512_issue_archetypes.md`](issue_1512_issue_archetypes.md) authoritative while avoiding
stale, quota-expensive, or semantically ambiguous GitHub-side mirrors.

## Why Body Metadata Stays Canonical

The issue body already carries the full contract:

- exactly one canonical `archetype`,
- exactly one canonical `evidence_tier`,
- short repository-relative `linked_policy` paths.

That representation is more precise than the current repository label set, and current Project #5
fields are defined for prioritization rather than archetype/evidence-tier classification. Mirroring
everything into labels or Project fields would either collapse meaning or require new schema
provisioning that this issue does not justify yet.

## Recommended Sync Boundary

### Labels

Low-risk label mirroring is acceptable only when the target labels already exist or are explicitly
provisioned first. The current repository exposes both broad labels such as `workflow` and
`benchmark` and typed labels such as `type:workflow`, `type:docs`, and `evidence:proposal`. Future
automation should treat the typed families as the only candidate programmatic mirrors. Broad labels
may remain human-facing discovery or triage labels, but they are not canonical metadata mirrors.

Any metadata-audit finding must fail closed for label mirroring. If the
`## Archetype Metadata` block is malformed, has invalid canonical values, or is
missing required keys such as `linked_policy`, the sync report/apply path
should not propose typed-label mirrors from that block even when
`archetype` or `evidence_tier` individually look valid. The full metadata block
stays authoritative until a human repairs it.

Recommended conservative archetype mappings:

| Body metadata | Existing/provisioned typed label | Recommendation |
| --- | --- | --- |
| `archetype: workflow` | `type:workflow` | Safe optional mirror |
| `archetype: docs` | `type:docs` | Safe optional mirror |
| `archetype: synthesis` | `type:synthesis` | Safe optional mirror |
| `archetype: benchmark-campaign` | `type:benchmark` | Safe optional mirror |
| `archetype: analysis` | `type:analysis` | Safe optional mirror |
| `archetype: training-campaign` | `type:training` | Safe optional mirror |

Everything else should remain body-only unless maintainers explicitly add and document a matching
typed label taxonomy first. In particular, do **not** infer new labels for `preflight`,
`slurm-execution`, `blocked-asset`, or any other archetype that lacks an exact typed label.

`evidence_tier` should remain body-only by default, including when an incomplete
metadata block otherwise contains a valid-looking evidence tier. If maintainers
later want evidence-tier label mirrors, they should document exact
`evidence:*` mappings separately and keep them optional. Do not infer
evidence-tier labels from partial name similarity, and do not treat labels such
as `evidence:proposal` as changing the canonical body value.

### Project #5

No immediate Project #5 archetype/evidence-tier sync is recommended.

Reasons:

1. `docs/project_prioritization.md` defines Project #5 around prioritization fields, not issue
   classification.
2. Projects v2 writes are GraphQL-only and higher-cost than body/label cleanup.
3. Archetype/evidence-tier mirrors would be convenience metadata, not stronger evidence.

If a future maintainer explicitly provisions Project fields for archetype metadata, writes should
remain opt-in, batched, and dry-run-first. This note does **not** assume any existing field IDs,
field names, or option IDs beyond the prioritization fields already documented elsewhere.

## Future Command Contract

If follow-up implementation is desired, prefer a small local/manual helper with report-first
behavior. Suggested shape:

```bash
uv run python scripts/tools/issue_archetype_sync.py report \
  --issue-number 1532 \
  --dry-run
```

Optional write gates should be explicit and independent:

- `--apply-labels`
- `--apply-project`
- `--issue-number <n>` or `--issue-list-file <path>`
- `--search-query <query>` only with an explicit batch-confirmation flag

The default invocation should perform no remote mutation.

## Dry-Run Report Shape

The dry-run output should make skipped writes obvious and explain why. A JSON summary or stable
Markdown table is sufficient if it includes at least:

| Field | Meaning |
| --- | --- |
| `issue_number` | Target issue |
| `body_metadata` | Parsed `archetype`, `evidence_tier`, `linked_policy`, or parse/validation error |
| `existing_labels` | Current label names |
| `proposed_label_additions` | Labels that would be added |
| `proposed_label_skips` | Candidate mirrors skipped with reasons (`label_missing`, `ambiguous`, `not_low_risk`) |
| `project_sync_mode` | `not_requested`, `dry_run_only`, `apply_requested`, or `schema_missing` |
| `project_field_candidates` | Only symbolic field/value names, never invented IDs |
| `rate_limit_snapshot` | Optional REST/GraphQL remaining counts when the command actually queried GitHub |
| `mutation_plan` | Ordered operations that would run if write gates were enabled |

Report summaries should clearly separate:

- parse errors in the body metadata block,
- safe label mirrors,
- project writes blocked by missing schema or missing opt-in.

## Write Gating And Batch Discipline

Any future implementation should follow
[`issue_713_batch_first_issue_workflow.md`](issue_713_batch_first_issue_workflow.md):

1. Parse and validate issue-body metadata first.
2. Report proposed label changes second.
3. Apply label/body cleanup before any Project #5 writes.
4. Treat Project writes as a separate opt-in pass.
5. Never mix broad issue cleanup with hidden project mutation in one default command.

Recommended safety defaults:

- refuse implicit backlog sweeps,
- require explicit issue targeting for writes,
- require `--apply-project` separately from `--apply-labels`,
- no auto-creation of labels or Project fields,
- skip ambiguous mappings instead of guessing.

## Rate-Limit And API Guidance

Future automation should stay rate-limit-conscious:

- use REST for ordinary issue and label reads/writes,
- reserve GraphQL for Project #5 reads/writes only when explicitly requested,
- check rate limits before larger batches,
- stop Project writes when GraphQL quota is low instead of retry-looping,
- cache project/field IDs only within a bounded local session or local ignored cache when needed.

This follows the repository's batch-first GitHub workflow and avoids spending GraphQL budget on
classification metadata that is already present in issue bodies.

## Future Test Expectations

If a follow-up implementation lands, add targeted tests rather than broad workflow fixtures:

1. body-metadata parser tests for valid, missing, and malformed YAML blocks,
2. mapping tests that confirm only allowed low-risk labels are proposed,
3. dry-run summary tests that confirm no writes occur by default,
4. explicit gating tests for `--apply-labels` and `--apply-project`,
5. rate-limit/degraded-path tests that confirm Project writes are skipped cleanly when quota or
   schema prerequisites are missing.

Those tests should stay narrow and avoid live GitHub mutation.

## Non-Evidence Caveats

Mirrored labels or Project fields are workflow convenience only. They do not:

- strengthen the evidence tier,
- replace the issue-body metadata block,
- prove current project schema support,
- justify broad backlog rewrites.

If a mirror disagrees with the issue body, the body remains authoritative until a human intentionally
repairs the mismatch.

## Validation Path

- repo-relative link sanity for links in this note and `docs/context/README.md`
- `BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh`
- `git diff --check`
