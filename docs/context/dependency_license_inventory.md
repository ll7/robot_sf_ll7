# Dependency license inventory and release boundary

Issue #7298 has three separate facts that must not be collapsed into a green
license verdict:

1. the frozen lock and profile closure identify the packages and artifacts that
   a supported environment would resolve;
2. installed metadata records what the selected distributions actually report;
3. `scripts/validation/dependency_license_policy.v1.json` records the release
   disposition still required for each distribution mode and vendored companion.

The generator is read-only and does not contact a package index:

```bash
python scripts/tools/check_dependency_license_inventory.py \
  --output output/validation/dependency-license-inventory.json
```

This command emits blocked evidence and exits `0` when the report is structurally
valid. A release or archive preflight must use the strict form:

```bash
python scripts/tools/check_dependency_license_inventory.py \
  --output output/validation/dependency-license-inventory.json \
  --fail-on-unresolved
```

Strict mode returns `2` for unknown, proprietary, conflicting, unbound, stale,
or policy-pending rows **that belong to at least one declared profile**. Lock
rows that no declared profile resolves are reported separately under
`unrepresented_lock_packages` and are deliberately outside the strict gate,
because the profile matrix — not the lock file — defines the release surface.
Whether every such row should be pulled into the gate is tracked separately;
until then, do not read a strict pass as a statement about them. Strict mode is
intentionally expected to remain blocked until a reviewer records a disposition;
a user-installed dependency is not silently treated as redistributed, and a
non-redistributed dependency is not silently approved. The report preserves
lock-provided source URLs, artifact filenames, SHA-256 values, and profile
membership, while installed metadata is labelled
`installed_distribution_not_artifact_bound` unless an exact artifact binding is
proved separately.

The committed profile matrix covers the root environment, every declared extra,
the explicit `all` closure, standalone `fast-pysf`, `pyrvo2`, and SocNavBench.
`rllib` remains a standalone profile and is explicitly excluded from `all` by
the current project declaration; the exclusion is reported rather than hidden.
The vendored Python-RVO2 and SocNavBench rows retain their upstream revision,
license facts, notices, local-change/provenance paths, and evidence digests.

The generated report has no wall-clock timestamp and records digests for
`pyproject.toml`, every selected lock/profile/policy/provenance input, schemas,
and the generator. Recheck an existing artifact with:

```bash
python scripts/tools/check_dependency_license_inventory.py \
  --repo-root . \
  --check-freshness output/validation/dependency-license-inventory.json
```

Freshness fails closed when the report was not generated from the canonical
`dependency_license_profiles.v1.json` and `dependency_license_policy.v1.json`,
so a report built against a substitute manifest or a relaxed policy cannot pass
as fresh. Adding `--fail-on-unresolved` to the freshness form re-applies the
strict exit code to the report's recorded `unresolved_count`.

This is release-compliance evidence, not a legal opinion. Closing #7298 still
requires reviewed dispositions for release-relevant blocked rows and a separate
proof that each supported profile was resolved with its pinned lock.
