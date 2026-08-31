# Immutable software candidate

`.github/workflows/software-candidate.yml` is the credential-free producer for a Robot SF
software-package candidate. It is a reusable workflow, not a publication workflow and not a
manual dispatch target. One job builds one wheel and one source distribution once, validates
those exact files, creates deterministic provenance, and uploads the checked bundle once.

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

The call requests no secrets. Its outputs are `artifact-id`, `artifact-digest`, `artifact-name`,
and `source-sha`. A later consumer must bind all four values before downloading or promoting an
artifact; a matching name alone is not identity evidence.

## Candidate contents

`candidate-manifest.json` conforms to
`scripts/dev/software_candidate_manifest.v1.schema.json` and binds four payload members:

- the only `robot_sf` wheel;
- the only `robot_sf` source distribution;
- a deterministic CycloneDX 1.5 SBOM; and
- `candidate-provenance.json`.

Every member record includes its exact filename, size, kind, and SHA-256. The envelope also binds
the source commit, repository, workflow run ID and attempt, package version, and the complete
validation roster. The manifest itself is the admission envelope, so it is not recursively listed
as one of its own payload members.

The producer uses the existing version-alignment, Twine metadata/README, distribution
archive/license, and clean wheel-install/entry-point owners. Assembly fails unless all four
validator identities are present exactly once in canonical order. The helper independently
reconstructs the expected tree from the exact commit through an absolute, configuration-empty
Git carrier, then enumerates the workspace without consulting repository configuration, the
index, ignore rules, or a `git` executable from `PATH`. Before and after the only build it hashes
raw tracked file bytes and symlink targets, checks Git executable modes and path types, and rejects
tracked changes or removals, untracked or ignored paths, and unsafe symlink targets. Build output
and runtime scratch remain outside the checkout. A materialized Git LFS path is accepted only when
the frozen commit marks that path `filter=lfs` and its bytes match the committed pointer's SHA-256
and size; no LFS helper is executed. The helper also rejects fuzzy or non-commit IDs,
multiple/missing/unclassified distributions, duplicate or unsafe archive members, mismatched
package metadata, and malformed SBOM input.
Pinned `uv 0.11.21` also writes a one-byte `*` `.gitignore` marker in a custom output directory;
the helper classifies only that exact tool marker and never copies it into the candidate bundle.
Any changed marker content remains an admission failure.

`uv export` currently includes a fresh UUID and timestamp in each CycloneDX document. The helper
removes only those volatile identity fields, binds the root component to the admitted wheel/sdist
version, and serializes the result canonically. Given identical input bytes and identities, the
SBOM, provenance, and manifest are byte-for-byte deterministic.

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
`ffa6635a7a37e21a36881ff8a89be59ee706c41107b94771ace8ed663d2f6469`) as well as its stable ID,
version, closed-object shape, required fields, and exact four-member contract. Supplying a
syntactically valid but weakened schema therefore fails before any candidate can be accepted.

Passing this check means only that the software candidate bytes and their build-time validation
evidence are internally consistent. Publication remains an author- and policy-gated operation.
