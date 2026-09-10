# Cleanup Eligibility Guard

Plain-language summary: before deleting one artifact or output identity, this check-only guard
decides whether the durable bytes behind it are safe to lose. It composes existing lifecycle
records -- output ownership, preservation receipts, active writers/leases, and consumer
references -- and fails closed when required state is missing or contradictory. It changes no
retention class, consumer decision, release policy, or deletion procedure.

- Motivating issue [#8906](https://github.com/ll7/robot_sf_ll7/issues/8906) (parent [#8819](https://github.com/ll7/robot_sf_ll7/issues/8819)); guard `scripts/validation/check_cleanup_eligibility.py`; fixtures `tests/validation/fixtures/cleanup_eligibility/cases.json`

## Outcomes

One explicit `artifact_id` returns exactly one outcome:

| Outcome | Meaning |
| --- | --- |
| `eligible` | Every gate passed; deletion would not lose durable-required bytes. |
| `protected_active` | An output owner or writer/lease is still active. |
| `protected_only_copy` | No verified durable copy, or fewer independent durable failure domains than the retention class requires. |
| `protected_unverified` | Durable copies exist but destination bytes, digest, or verification basis do not qualify. |
| `protected_referenced` | An unresolved consumer or active retention hold depends on the artifact. |
| `unknown` | Lifecycle, digest, failure-domain, writer, or consumer state cannot be read. |
| `conflict` | Owner records contradict each other. |

Only `eligible` permits cleanup; `unknown` and `conflict` fail closed, and absence of a record
never proves disposal.

## Gates before `eligible`

- the exact `semantic_digest` is recorded and every counted copy matches it;
- at least one verified copy is in an approved failure domain (`public_release`, `cloud_durable`,
  `personal_durable`); `durable_required`/`release_facing` need two independent failure domains;
- every counted copy has a checksum-receipt or manifest-verification basis;
- no active output owner, writer, or lease, and no two writers claiming the same artifact;
- no active or `unknown` consumer, and no retention hold;
- `private_projection` is `public_safe` so no private locator can leak.

A copy on an expiring host (`institutional_durable`, `institutional_cache`) is non-durable unless
another approved failure domain is verified. Directory name, file age, Git-ignore status, and
scheduler completion are rejected as deletion evidence and keep the artifact `protected_unverified`.

## Reason codes

| Reason code | Gate |
| --- | --- |
| `active_output_owner`, `active_writer_or_lease` | Active ownership. |
| `unresolved_consumer`, `retention_hold` | Consumer or retention rule. |
| `no_verified_durable_copy`, `only_expiring_host_copy`, `insufficient_failure_domains` | Durable custody or independence. |
| `destination_bytes_unverified`, `destination_digest_mismatch`, `non_evidence_basis` | Copy evidence. |
| `stale_pointer`, `concurrent_writer`, `duplicate_copy`, `duplicate_artifact_record` | Conflicting records. |
| `private_projection_unresolved`, `missing_failure_domain`, `missing_digest`, `owner_state_unknown`, `writer_state_unknown`, `consumer_state_unknown`, `record_schema_error`, `artifact_not_recorded`, `no_copies_recorded` | State cannot be read. |
| `private_locator_rejected` | Value is not a public-safe logical ID; never echoed. |

Blocking owner identities use the public-safe form `<kind>:<logical-id>` (`owner`, `writer`,
`consumer`, `retention`, `pointer`, `projection`, `artifact`, `copy`); codes are sorted and stable
in JSON and text output.

## Check-only usage

```bash
uv run python scripts/validation/check_cleanup_eligibility.py \
  --record tests/validation/fixtures/cleanup_eligibility/cases.json \
  --artifact two-copy-verified-result --check --json
```

Exit codes: `0` eligible; `2` protected/unknown/conflict or usage error; `1` record missing,
unreadable, or not valid JSON. The existing cleanup command can require the same result:

```bash
uv run python scripts/dev/clean_generated_output.py \
  --eligibility-record <RECORD.json> --artifact <ARTIFACT_ID> --check
```

Without `--check` it deletes only after `eligible`; no automatic deletion is added here. Record
producers compose the canonical owners: output ownership and categories
(`robot_sf/common/artifact_paths.py`), result-tree manifests (`scripts/tools/chunk_manifest.py`),
private-safe custody projections (`scripts/tools/locator_snapshot.py`), worktree leases
(`scripts/dev/pr_gate_lease.py`), and the evidence vocabulary
([artifact_evidence_vocabulary.md](artifact_evidence_vocabulary.md)). If any owner cannot supply
lifecycle, digest, failure-domain, writer, or consumer state, record `unknown` inputs explicitly and
stop; do not infer eligibility from absence in one registry. Public reports contain logical IDs and
digests only -- never locators, hostnames, mount paths, or URLs.
