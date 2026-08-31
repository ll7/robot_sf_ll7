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
rejects dirty or drifting source, fuzzy commit IDs, multiple/missing/unclassified distributions,
duplicate or unsafe archive members, mismatched package metadata, and malformed SBOM input.

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

Passing this check means only that the software candidate bytes and their build-time validation
evidence are internally consistent. Publication remains an author- and policy-gated operation.
