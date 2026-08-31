# Protected software-package promotion

`.github/workflows/software-promotion.yml` promotes one already-verified
software candidate from the build-once workflow. It does not rebuild the
package. Every job rechecks the candidate manifest, the GitHub artifact ID,
name, archive digest, source commit, version, wheel/sdist hashes, SBOM, and
provenance before it can use the bytes.

This is the software-package lane, separate from the benchmark-data release
and from Zenodo publication. The workflow does not create a tag or GitHub
Release. It is a manually dispatched, approval-gated workflow so an
unreviewed tag or moving branch cannot publish a package accidentally.

## Maintainer setup

Before the first rehearsal, a repository administrator must create these
GitHub environments in the repository settings:

| Environment | Required deployment protection | Trusted publisher target |
| --- | --- | --- |
| `testpypi` | at least one required reviewer | TestPyPI project `robot-sf`, workflow `.github/workflows/software-promotion.yml`, environment `testpypi` |
| `pypi` | at least one required reviewer | PyPI project `robot-sf`, workflow `.github/workflows/software-promotion.yml`, environment `pypi` |

Register the PyPI and TestPyPI trusted publishers using the exact repository,
workflow filename, and environment names above. The workflow receives a
short-lived OpenID Connect (OIDC) identity token only in the corresponding
upload job. No PyPI API token, password, or repository secret is needed. Do
not put a token in workflow inputs, artifacts, comments, or logs.

The package names must be reserved and the final metadata/rights candidate
must be accepted before running the workflow. In particular, this workflow
does not bypass #8019, #8017, or the sanitized source candidate from #8149.

## First promotion

1. Run the reusable
   [`software-candidate.yml`](../.github/workflows/software-candidate.yml)
   workflow at the reviewed source ref. Record its exact artifact ID, artifact
   name, `sha256:` artifact digest, workflow run ID, source SHA, and package
   version from the run summary.
2. Dispatch `software-promotion.yml` with those six candidate values. The
   candidate artifact is downloaded by immutable artifact ID, and its GitHub
   API metadata is checked before its files are inspected.
3. Approve `testpypi`. The upload job sends only the candidate wheel and source
   distribution to TestPyPI with `skip-existing: false`, then stores a
   credential-free receipt containing all candidate hashes and provenance
   bindings.
4. The cold-install job downloads both formats from the TestPyPI public simple
   index, checks byte identity, installs the exact retrieved wheel in a clean
   environment, and probes every advertised console script from outside the
   checkout. It must produce a passed cold-install receipt.
5. Approve `pypi`. The production job can start only after the cold-install
   job succeeds. It revalidates the candidate and both receipts before sending
   the same staged wheel and source distribution to PyPI. A pre-existing
   version is an error; the workflow never overwrites or silently skips a
   collision.

Receipt artifact IDs and `sha256:` digests are printed as non-secret run
   metadata. Preserve them with the release record so an interrupted run can
   be resumed without uploading again.

## Safe resume

If TestPyPI accepted the package but the workflow stopped afterwards, dispatch
again with the prior TestPyPI receipt artifact's run ID, artifact ID, name,
and digest. The workflow verifies that unexpired artifact and the exact
candidate binding, skips the TestPyPI upload, and reruns the public-index
cold-install gate.

If PyPI accepted the package but the workflow stopped while recording its
receipt, supply the analogous prior PyPI receipt values. The production job
then verifies that receipt and skips a duplicate upload. Missing, expired,
replayed, wrong-version, wrong-source, wrong-candidate, or hash-mismatched
receipts fail closed. Never use `skip-existing: true` as a recovery method.

## Review and evidence boundary

The candidate builder is the only build owner. The promotion workflow is an
index-publication owner, and the receipt artifacts are compact provenance
evidence—not a substitute for independent package-index downloads. A passed
TestPyPI smoke proves package promotion and installability only; it does not
establish benchmark or dissertation claims. Publish the software Zenodo
record and a GitHub Release only through their separate, reviewed workflows
after the package receipts and rights/metadata gates are complete.
