# Immutable software candidate

`.github/workflows/software-candidate.yml` is the credential-free producer for a Robot SF
software-package candidate. Its canonical invocation is a direct `workflow_dispatch` at the
reviewed source head; it also remains callable by a pinned reusable caller. One job builds one
wheel and one source distribution once, validates those exact files, creates deterministic
provenance, and uploads the checked bundle once.

This slice does **not** authorize or perform a package-index upload, GitHub Release, Zenodo
deposit, DOI reservation, tag, environment approval, or trusted-publisher/OIDC exchange. It does
not admit benchmark, model, dataset, or scientific evidence. Parent publication issue #8023
remains blocked until its independent metadata, rights, policy, and author gates are satisfied.

## Calling the producer

An authorized caller pins the reusable workflow to an exact reviewed commit:

```yaml
jobs:
  software-candidate:
    permissions:
      contents: read
    uses: ll7/robot_sf_ll7/.github/workflows/software-candidate.yml@<exact-commit-sha>
```

The call requests no secrets. Its outputs are the candidate identity (`artifact-id`,
`artifact-digest`, `artifact-name`, `source-sha`, `candidate-source-sha`, and `candidate-tree-sha`)
plus the separately uploaded rights-receipt identity (`rights-artifact-id`,
`rights-artifact-digest`, and `rights-artifact-name`). A later consumer must bind every returned
identity before downloading or promoting either artifact; a matching name alone is not identity
evidence. Direct dispatch uses the deterministic name
`robot-sf-software-rights-admission-<source-sha>-<run-id>-<attempt>` for the receipt artifact.

## Candidate contents

`candidate-manifest.json` conforms to
`scripts/dev/software_candidate_manifest.v1.schema.json` and binds four payload members:

- the only `robot_sf` wheel;
- the only `robot_sf` source distribution;
- a deterministic CycloneDX 1.5 SBOM; and
- `candidate-provenance.json`.

Every member record includes its exact filename, size, kind, and SHA-256. The envelope also binds
the source commit, the rights-scoped candidate commit and tree, repository, workflow run ID and
attempt, package version, materialization policy and source-inventory hashes, and the complete
validation roster. The manifest itself is the admission envelope, so it is not recursively listed
as one of its own payload members.

## Rights-scoped source materialization

Before building, the producer reads the tracked
`scripts/validation/software_candidate_policy.v1.json` from the exact source commit. The
`materialize-source` command selects the policy's regular files, excludes models, maps, examples,
other non-candidate paths, keeps ORCA/pyrvo2 and SocNavBench source checkouts external, and
requires release-safe evidence for selected asset-like files. It creates a standalone
deterministic Git candidate. The generated `SOFTWARE_CANDIDATE.json` and
candidate-local rights inventory record the source SHA, policy and inventory hashes, selected
members, and explicit exclusions. The external materialization report records the candidate
commit and tree used by the later build.

The build root is then staged from that candidate commit, not directly from the authoritative
checkout. The strict distribution-license gate runs against the staged candidate tree and its
candidate-local inventory. The clean-install smoke still exercises the installed core runtime
and every console entry point; when the candidate has no packaged map files, its runtime check
uses a deterministic programmatic core map. This is runtime smoke evidence only, not benchmark or
scientific evidence.

The producer uses the existing version-alignment, Twine metadata/README, distribution
archive/license, and clean wheel-install/entry-point owners. Assembly fails unless all four
validator identities are present exactly once in canonical order. The helper independently
reconstructs the expected tree from the exact commit through an absolute, configuration-empty
Git carrier, then enumerates the authoritative checkout without consulting repository
configuration, the index, ignore rules, or a `git` executable from `PATH`. Before staging, after
staging, after the only build, and during assembly it hashes raw tracked file bytes and symlink
targets, checks Git executable modes and path types, and rejects tracked changes or removals,
untracked or ignored paths, and unsafe symlink targets.

The authoritative checkout supplies identity, materialization policy, validators, and admission
only. Before the build, `stage-build-source` uses absolute system Git with empty global/system
configuration, an empty template, disabled hooks, and no executable search path to create a new
external `$BUILD_SOURCE`. It verifies that disposable root against the exact materialized
candidate commit/tree and rechecks the candidate source before returning. The sole `uv build` runs
from that disposable exact-commit root; build output and runtime scratch also remain external.
Hatch-VCS may therefore generate ignored `robot_sf/_version.py` only inside the disposable root.
That path is not allowlisted or ignored by the authoritative source gate, and assembly remains
bound to the untouched authoritative checkout. The canonical wheel-install smoke wrapper also
runs from the disposable root, confining its unavoidable `output/validation` scratch directory
there while its report and wheel inputs remain absolute external paths. Provenance records the
disposable-exact-commit build role and the exact one-build command. A materialized Git LFS path is
accepted only when the frozen commit marks that path `filter=lfs` and its bytes match the committed
pointer's SHA-256 and size; no LFS helper is executed. The helper also rejects fuzzy or non-commit
IDs,
multiple/missing/unclassified distributions, duplicate or unsafe archive members, mismatched
package metadata, and malformed SBOM input.
Pinned `uv 0.11.21` also writes a one-byte `*` `.gitignore` marker in a custom output directory;
the helper classifies only that exact tool marker and never copies it into the candidate bundle.
Any changed marker content remains an admission failure.

`uv export` currently includes a fresh UUID and timestamp in each CycloneDX document. The helper
removes only those volatile identity fields, binds the root component to the admitted wheel/sdist
version, and serializes the result canonically. Given identical input bytes and identities, the
SBOM, provenance, and manifest are byte-for-byte deterministic.

The producer then runs the candidate-bound supported-surface dependency-license inventory with
`--candidate-bundle ... --fail-on-unresolved`; candidate binding selects the closed `all` profile.
The reviewed `all` closure is the exact v0.0.6 public
surface: `viz`, `maps`, `benchmark`, `training`, `gpu`, `recurrent`, `progress`, `analytics`,
`browser`, `sacadrl`, `socnav`, and `criticality`; `rllib` and ORCA/pyrvo2 remain outside it. Its
report binds the exact candidate manifest/member bytes, canonical dependency policy/profile
inputs, and the `all` profile into `supported_dependency_gate`, including the explicit twelve-
extra roster. A core-only report is rejected for this candidate even if it happens to have zero
unresolved rows. The separate `rights-admission.json` receipt conforms to
`robot_sf.software_rights_admission.v1` and is emitted only when both strict gates pass with
`unresolved_count: 0`; a blocked dependency report cannot produce an accepted receipt. The
current repository still has unresolved rows on the full supported closure, so a real current-head
run is expected to stop before receipt upload. This is evidence of a fail-closed boundary, not
publication approval.

## Offline revalidation

After obtaining a bundle by exact artifact ID and checking GitHub's artifact digest, revalidate it
from a trusted checkout of the same helper/schema revision:

```bash
python scripts/dev/software_candidate_manifest.py verify \
  --bundle-dir /path/to/downloaded/bundle \
  --expected-source-sha <exact-40-hex-source-sha> \
  --expected-workflow-run-id <exact-decimal-run-id>
```

`verify` has no build-tool or network dependency. It fails on missing, additional, duplicated,
schema-invalid, source-drifted, run-drifted, size-drifted, or hash-drifted content; it also
rechecks wheel/sdist metadata, archive-member safety, SBOM version binding, and provenance. A
consumer or future promotion workflow must call this verifier on the downloaded bytes. It must
not run `uv build`, `python -m build`, or any equivalent package build command.

Schema admission is bound to the reviewed v1 schema bytes (SHA-256
`d7bd1f2d7c4146b85fb23ee2d6462bb363f94c79c749b50336832742caf6bdad`) as well as its stable ID,
version, closed-object shape, required fields, and exact four-member contract. Supplying a
syntactically valid but weakened schema therefore fails before any candidate can be accepted.

Passing this check means only that the software candidate bytes and their build-time validation
evidence are internally consistent. Publication remains an author- and policy-gated operation.
