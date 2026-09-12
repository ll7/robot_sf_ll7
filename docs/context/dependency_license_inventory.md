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

For release checks, select the exact reviewed profile or profile union instead of using the whole
development matrix. The v0.0.6 software candidate uses the checked-in `all` closure, which covers
the twelve public extras and excludes standalone `rllib`:

```bash
python scripts/tools/check_dependency_license_inventory.py \
  --fail-on-unresolved
```

The development checkout still declares `rllib` for local use. The rights-clean candidate
materializer removes only that stanza from its copied `pyproject.toml`, preserving the `all`
aggregator and the twelve supported extras. Its wheel and source distribution therefore must
advertise exactly thirteen `Provides-Extra` values (`all` plus those twelve); the source checkout
is never modified.

The candidate Git repository is a deterministic standalone root; it does not pretend to have the
source commit as a Git parent. The separate source SHA remains the provenance binding, reflected by
the rights policy's `commit_parent: root_commit` contract.

The selected profile closure is recorded under `surface.profile_ids`. Other declared profiles,
lock rows, and installed distributions remain visible in the report with an explicit
`outside_selected_profiles` marker; selection never turns an unresolved row into an approval.
Unrepresented rows are scoped by profile membership rather than by lockfile name, so unrelated
rows sharing a lockfile do not silently become release members. On the full declared surface,
unexplained rows retain an `unresolved_membership` marker and continue to block strict mode.

When the immutable software-candidate bundle has already been admitted, bind its exact wheel,
source distribution, provenance, and CycloneDX software bill of materials (SBOM) to the reviewed
v0.0.6 supported closure:

```bash
python scripts/tools/check_dependency_license_inventory.py \
  --candidate-bundle output/validation/software-candidate \
  --output output/validation/dependency-license-inventory.json \
  --fail-on-unresolved
```

Candidate binding verifies the closed manifest/provenance contract, member checksums, archive
package identity and metadata, and the SBOM component set against the selected lock closure. It
replaces ambient installed metadata for the selected rows with an `artifact_bound` identity
observation, but it does not invent license facts: a reviewed exact policy disposition is still
required for each dependency. The resulting `candidate_binding` record carries the candidate
identity, materialization commit/tree and policy/inventory identities, member digests, and
component digest needed for candidate-bundle admission. A producer that omits the optional
materialization envelope remains compatible; when present, it is validated and must match the
provenance record exactly.
A candidate invocation without `--profile` selects `all`; it never silently narrows to `core`. The
separate rights-admission consumer rejects a report whose surface is not exactly `["all"]`; the
report's `all` profile must carry the twelve supported extra IDs. It does not invent license facts:
a reviewed exact policy disposition is still required for each dependency.

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


The `sb3-contrib` 2.9.0 row records the exact-row review decision for the
`external_dependency_not_redistributed` surface. It is limited to the current
`all` profile and the exact identity
`sb3-contrib@2.9.0#d0ac0e722c73f77d`. The selected PyPI artifacts are
`sb3_contrib-2.9.0.tar.gz` (SHA-256
`6e839c669552ecb3deb616a42ea98ed5e6c599eb5ac5f86232415b87a3873699`, 90,312
bytes) and `sb3_contrib-2.9.0-py3-none-any.whl` (SHA-256
`b492b2be792f8f8214ff6b984e94812dc877f809ae82302cbe00626442ae316a`, 93,042
bytes). Each selected archive has the 1,078-byte MIT `LICENSE` with SHA-256
`b76bbab0bd3d0182611f19ed6a4593408787084e9b024c7d9dd1d0295c2bf3a8`; the
wheel member is RECORD-verified and the wheel/sdist license bytes are equal.

The upstream `v2.9.0` tag resolves to commit
`075bd5be8d43848b0f0cd3bc8a32f6892d0d58fb`. The immutable upstream references
are [`LICENSE`](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/075bd5be8d43848b0f0cd3bc8a32f6892d0d58fb/LICENSE)
and the [tag tree](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/tree/075bd5be8d43848b0f0cd3bc8a32f6892d0d58fb).
The inspected tree contains `LICENSE` only; no `NOTICE` member is asserted.
The source registry is `https://pypi.org/simple`, with metadata at
`https://pypi.org/pypi/sb3-contrib/2.9.0/json`.

This row is bound to the reviewed diagnostic candidate source
`fb6896576a9240838901f2b0994f288ec510c827`, candidate commit
`f537fa5661b2d3adba3acf9ee7564bcadbcc406e`, candidate tree
`1ca7f58207e7425942498d27368c3fd4ade2047b`, candidate lock SHA-256
`d01cbbf7fb7215d140b3c78f66202e0c48e449601f809036f5e02e9a9bfb79c7`, and
raw candidate manifest SHA-256
`d1ed5ca826eff94123b5e781173df7e67026d3db31e8a334ee10e96aaef992d0`.
The candidate source inventory SHA-256 is
`4360f56472a72fbbd97d46621fff7322331ff4764be4e13cd0fae175151083d8`, and the
candidate provenance SHA-256 is
`68e82d94b44f341db8249d3b568a460d9fd1dad6112da3ae63ff97875b31052a`.
Neither inspected Robot SF candidate payload contains an `sb3-contrib` path.
`candidate_archive_shipped` remains unrecorded; this exact policy row preserves
the reviewed external/non-redistributed boundary and does not claim legal
permission, rights clearance, custody, or whole-release admission. The review
input bytes are identified by SHA-256
`b0c74e18ef4294d17a0dab8d89136125e8da2bcfb93db1064e773b93cfbd5075`. It was
recorded at `2026-09-10T17:49:07Z` by `/root/p00_runtime` on the
`gpt-5.6-luna/max` factual route; no human or legal identity is claimed.
Version, source, artifact, profile, candidate, or packaging-surface changes
reopen this row. `bundled_source`, `built_companion`, mirrored, vendored,
container-bundled, unknown, unavailable, and conflicting surfaces remain
blocked.

This is release-compliance evidence, not a legal opinion. Closing #7298 still
requires reviewed dispositions for release-relevant blocked rows and a separate
proof that each supported profile was resolved with its pinned lock.
