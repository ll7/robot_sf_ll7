# Robot SF Ecosystem Contract

## Contents

- [Purpose](#purpose)
- [Files and ownership](#files-and-ownership)
- [Capability admission](#capability-admission)
- [Canonical bytes and digests](#canonical-bytes-and-digests)
- [Compatibility rules](#compatibility-rules)
- [Change classification](#change-classification)
- [Revision envelope](#revision-envelope)
- [Canonical conformance fixture](#canonical-conformance-fixture)
- [Commands](#commands)
- [Current limits](#current-limits)

## Purpose

The Robot SF ecosystem contract is a machine-readable producer declaration.
It tells downstream tools which explicit Robot SF interfaces they can use. It
does not require downstream tools to import Robot SF Python modules.

The contract covers only reviewed CLI commands, JSON schemas, protocol
identifiers, and artifact identity conventions. It does not scrape prose or
promote an internal symbol to a stable API.

This work implements
[robot_sf_ll7 issue 6710](https://github.com/ll7/robot_sf_ll7/issues/6710).
It is an evidence-governance interface. It does not make or change a scientific
claim.

## Files and ownership

The explicit source registry is
`robot_sf/benchmark/contracts/robot_sf_ecosystem_capabilities.v1.json`.
Maintainers review changes to this file. The registry names each capability and
the narrow source selectors that own its meaning.

The generator is
`scripts/tools/build_robot_sf_ecosystem_contract.py`. It resolves the registry,
validates the selected source objects, and writes these generated files:

- `robot_sf/benchmark/contracts/robot_sf_ecosystem_contract.v1.json`
- `robot_sf/benchmark/contracts/robot_sf_ecosystem_contract.v1.sha256`

The producer payload and revision envelope use separate Draft 2020-12 schemas:

- `robot_sf/benchmark/schemas/robot_sf_ecosystem_contract.v1.json`
- `robot_sf/benchmark/schemas/robot_sf_ecosystem_revision_envelope.v1.json`

Do not edit the generated producer contract or digest sidecar by hand. Change
the registry or an authoritative source, then run the generator.

## Capability admission

A capability must have a unique ID, an interface version, a status, a semantics
ID, a public location, and one or more authoritative input selectors. The
generator rejects unknown or unused selectors.

CLI capabilities must cite all of these inputs:

- The installed project entry point.
- The command handler.
- A test that fixes the command contract.
- An ordered list of documented exit codes.

Schema capabilities must cite exactly one location-matching Draft 2020-12 JSON
Schema. Protocol capabilities must cite a literal protocol identifier. Artifact
identity capabilities must declare a convention identifier and tested source.

The v1 registry cannot mark a capability as `stable`. Stable status needs a
separate reviewed stability owner and policy. This guard prevents accidental
stability promises.

## Canonical bytes and digests

The producer contract uses RFC 8785 JSON Canonicalization Scheme (JCS). The
project pins and tests `rfc8785==0.1.4`. The generator does not use a generic
`sort_keys` fallback.

The committed JSON file contains exact RFC 8785 bytes and has no trailing line
feed. `contract_digest.value` is SHA-256 over the canonical document after the
complete `contract_digest` member is removed. The sidecar is SHA-256 over the
exact committed file bytes.

Each authoritative input also has a SHA-256 digest:

- JSON schemas use their canonical RFC 8785 value.
- Selected TOML fields and Python constants use their canonical JSON value.
- Selected Python symbols and tests use a canonical projection of their AST
  node type, selector, decorators, and source text.

The test vectors in
`tests/fixtures/ecosystem_contract/v1/canonicalization_vectors.json` fix object
ordering, IEEE 754 number formatting, and UTF-16 property ordering. Downstream
implementations can reuse these vectors.

## Compatibility rules

A consumer supplies explicit requirements. The checker matches each required
capability by these fields:

1. `capability_id`
2. Interface major version
3. Accepted status
4. `semantics_id`

The consumer must also support contract schema major 1 and every feature in
`minimum_consumer_features`. A whole-contract digest change is provenance. It
is not, by itself, an incompatibility. Commit, tag, and lock changes are also
provenance only.

The checker fails closed on malformed documents, unsupported schema versions,
duplicate IDs, missing capabilities, unsupported status, changed semantics,
and missing validator features.

A minimal consumer requirements file has this form:

```json
{
  "schema_version": "robot_sf_ecosystem_requirements.v1",
  "supported_contract_schema_majors": [1],
  "supported_consumer_features": [
    "rfc8785.jcs.v1",
    "robot_sf.contract.capability_match.v1",
    "robot_sf.contract.change_classification.v1",
    "sha256.v1"
  ],
  "required_capabilities": [
    {
      "capability_id": "robot_sf.schema.episode.v1",
      "interface_major": 1,
      "accepted_statuses": ["beta"],
      "semantics_id": "robot_sf.schema.episode.record_semantics.v1"
    }
  ]
}
```

Pass the file with `--requirements` during `--validate`. The command returns 3
when the valid producer contract does not satisfy the requirements.

## Change classification

Every non-initial contract must bind the digest of its baseline and declare one
of `additive`, `deprecated`, or `breaking`.

V1 classifies a new capability ID as additive. It classifies an explicit
deprecation record as deprecated. Removal, interface-major change, semantics-ID
change, CLI command or exit-code change, schema/protocol/convention identity
change, changed deprecation semantics, and a newly required consumer-validator
feature are breaking.

V1 treats every changed authoritative selector digest for an existing
capability as breaking. It does this even when the interface has a same-major
version increase. A digest alone cannot prove that a schema change added only
optional fields. It also cannot rule out required-field, unit, denominator,
enum, hash, or other semantic changes. A future reviewed semantic diff format
can add a narrow optional-field exception. V1 does not infer that exception.

A transition to `deprecated` uses the deprecation class. Every other status
transition is breaking because status is an explicit consumer match field.

Breaking changes require a new contract major. Additive, deprecation, and
unchanged records must preserve the contract major. Every non-initial candidate
version must also be strictly greater than its baseline. A breaking change
labeled as additive fails validation.

## Revision envelope

The producer contract is revision-invariant. It contains no source commit,
release tag, or lock-file digest. An unrelated commit therefore does not require
contract regeneration.

Release assembly can create a separate revision envelope. The envelope binds:

- A full source commit SHA.
- Release status and an optional required tag.
- The exact producer contract file digest.
- The exact `uv.lock` file digest.
- The generator path, version, and canonicalization profile.

The envelope is also canonical RFC 8785 JSON. Validation resolves both bound
paths inside the repository and recomputes both digests. Tagged and released
envelopes additionally verify that the source commit exists, the tag resolves
to that commit, and both bound files are the blobs at that commit. An
unreleased envelope is a preparation record and is not release evidence.

## Canonical conformance fixture

The v1 contract declares the stable identity, version, and repository path of
`robot_sf.ecosystem_handoff.v1`. It does not include a content digest for that
packet. The packet records the contract digest, so hashing packet content inside
the producer contract would create a contract/fixture digest cycle. A future
release manifest binds the contract and packet content digests together.

The packet is generated from production serializers and writers at
`tests/fixtures/ecosystem_handoff/v1/`. It contains one deterministic episode,
schema and provenance records, table- and figure-ready inputs, an artifact
manifest, portable `SHA256SUMS`, and five negative variants. It is diagnostic
conformance material only. It is not benchmark output, software release
content, or scientific evidence.

Generate and validate the packet:

```text
uv run python scripts/tools/build_ecosystem_handoff_fixture.py --overwrite
uv run python scripts/tools/build_ecosystem_handoff_fixture.py --check
uv run python scripts/tools/build_ecosystem_handoff_fixture.py --validate
```

The standalone validator at
`scripts/tools/validate_ecosystem_handoff_fixture.py` imports no Robot SF
modules. A downstream consumer can copy a packet and run its validator with
`--packet-dir`, then verify the packet-local checksums with
`sha256sum -c SHA256SUMS`.

## Commands

Generate the contract and sidecar:

```text
uv run python scripts/tools/build_robot_sf_ecosystem_contract.py
```

Check committed output and all authoritative source selectors:

```text
uv run python scripts/tools/build_robot_sf_ecosystem_contract.py --check
```

Validate an external contract and the committed sidecar:

```text
uv run python scripts/tools/build_robot_sf_ecosystem_contract.py \
  --validate robot_sf/benchmark/contracts/robot_sf_ecosystem_contract.v1.json \
  --digest-file robot_sf/benchmark/contracts/robot_sf_ecosystem_contract.v1.sha256 \
  --verify-sources
```

Compare a candidate contract with its baseline:

```text
uv run python scripts/tools/build_robot_sf_ecosystem_contract.py \
  --compare baseline.json candidate.json
```

Create an unreleased revision envelope without changing the invariant producer
contract:

```text
uv run python scripts/tools/build_robot_sf_ecosystem_contract.py \
  --write-revision-envelope build/ecosystem-revision.json \
  --source-commit 0123456789abcdef0123456789abcdef01234567
```

The CLI returns 0 for success, 1 for stale committed generated output, 2 for an
invalid document or invocation, and 3 for an incompatibility or invalid change
declaration.

## Current limits

The producer contract currently declares one canonical conformance fixture as
defined by [robot_sf_ll7 issue
6711](https://github.com/ll7/robot_sf_ll7/issues/6711). This declaration is a
stable identity/version/path reference, not an artifact publication or a
scientific-evidence approval.

The files under `tests/fixtures/ecosystem_contract/v1/` are internal validator
fixtures. They test valid and invalid contract transitions. They are not public
Robot SF simulation fixtures and are not listed in the producer contract.
