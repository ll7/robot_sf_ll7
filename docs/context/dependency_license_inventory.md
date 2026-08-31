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
or policy-pending rows that belong to the selected profile surface. Lock rows
that no declared profile resolves remain listed under
`unrepresented_lock_packages`, with one matching record in
`unrepresented_lock_package_dispositions`. Each record is either a
`reviewed_exclusion` (for a declared development/tooling group or a
resolution marker proven inactive for the target) or `unresolved`; the latter
is included in the strict failure count on the default full declared surface.
An explicit `--profile` selection narrows this strict check to that profile
closure while retaining outside rows as visible, non-member context. A row
without a reviewed reason is never silently treated as approved. The profile
matrix — not the lock file alone — still defines the supported release surface,
and a strict pass does not make a legal or redistribution claim. The report preserves
lock-provided source URLs, artifact filenames, SHA-256 values, and profile
membership, while installed metadata is labelled
`installed_distribution_not_artifact_bound` unless an exact artifact binding is
proved separately.

The manifest records the reviewed exclusion rules for root development,
documentation, CARLA, imitation, and standalone fast-pysf development
contexts. Resolver rows for another target are excluded only when their
explicit `resolution-markers` are proven false from the manifest target. Any
remaining row is emitted as `unresolved_membership` and keeps strict mode
blocked until a maintainer reviews its context.

For release checks, narrow the command to the exact declared profile or profile union instead of
using the whole development matrix:

```bash
python scripts/tools/check_dependency_license_inventory.py \
  --profile core \
  --profile viz \
  --fail-on-unresolved
```

The selected profile closure is recorded under `surface.profile_ids`. Other declared profiles,
lock rows, and installed distributions remain visible in the report with an explicit
`outside_selected_profiles` marker; selection never turns an unresolved row into an approval.
Unrepresented rows are scoped by profile membership rather than by lockfile name, so unrelated
rows sharing a lockfile do not silently become release members. On the full declared surface,
unexplained rows retain an `unresolved_membership` marker and continue to block strict mode.

When the immutable software-candidate bundle has already been admitted, bind its exact wheel,
source distribution, provenance, and CycloneDX software bill of materials (SBOM) to the selected
frozen closure:

```bash
python scripts/tools/check_dependency_license_inventory.py \
  --profile core \
  --candidate-bundle output/validation/software-candidate \
  --output output/validation/dependency-license-inventory.json \
  --fail-on-unresolved
```

Candidate binding verifies the closed manifest/provenance contract, member checksums, archive
package identity and metadata, and the SBOM component set against the selected lock closure. It
replaces ambient installed metadata for the selected rows with an `artifact_bound` identity
observation, but it does not invent license facts: a reviewed exact policy disposition is still
required for each dependency. The resulting `candidate_binding` record carries the candidate
identity, member digests, and component digest needed for candidate-bundle admission.

The committed profile matrix covers the root environment, every declared supported extra,
the explicit `all` closure, standalone `fast-pysf`, and SocNavBench.
`rllib` remains a standalone profile and is explicitly excluded from `all` by
the current project declaration; the exclusion is reported rather than hidden.
Per maintainer decision on #8021 (Option 2), ORCA (`pyrvo2`) is documented as an
external optional installation and excluded from the supported release extras and sanitized
package inventory. The local vendored companion remains development/source-checkout
infrastructure, not a shipped or supported PyPI companion.
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

## Exact package dispositions

The policy's `package_dispositions` registry is the only place where a reviewed
package/version exception may override a broad distribution-mode hold. Each row
is bound to its exact license expression, source/index, lock artifact filenames
and SHA-256 values, upstream notice references, frozen profile set, allowed and
blocked surfaces, and a local evidence path. A package row passes this exact
policy only when all of those identities match the lock and observed metadata.

The llvmlite 0.49.0 row records the bounded `A_surface_specific_disposition`
ruling from Issue #7653: `user_installed` and `not_distributed` are allowed;
`bundled_source` and `built_companion` remain blocked, as do mirrored,
vendored, container-bundled, unknown, unavailable, and conflicting surfaces. The exact
`BSD-2-Clause AND Apache-2.0 WITH LLVM-exception` expression is not generalized
to arbitrary SPDX `WITH` expressions. Its durable notice and provenance
references are recorded in
`docs/context/evidence/llvmlite_0.49.0_surface_disposition_2026-08-20.md`.

This is release-compliance evidence, not a legal opinion. Closing #7298 still
requires reviewed dispositions for release-relevant blocked rows and a separate
proof that each supported profile was resolved with its pinned lock.
