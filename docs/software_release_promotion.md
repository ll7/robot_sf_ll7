# Protected software-package promotion

`.github/workflows/software-promotion.yml` promotes one already-verified
software candidate from the build-once workflow. It does not rebuild the
package. Every job rechecks the candidate manifest, the GitHub artifact ID,
name, archive digest, source commit, version, wheel/sdist hashes, SBOM, and
provenance before it can use the bytes. It also requires an independent,
successful rights-clean admission from the sanctioned candidate workflow; protected-environment setup or a
caller-provided JSON file cannot clear unresolved release rights.

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
does not bypass #8019, #8017, or the sanitized source candidate gate.

## Rights-clean admission is mandatory

The current broad development tree is not eligible for package publication.
The dispatch form therefore requires five additional values for the exact
GitHub artifact produced by the sanctioned candidate workflow:
`rights_admission_run_id`, `rights_admission_run_attempt`, `rights_admission_artifact_id`,
`rights_admission_artifact_name`, and `rights_admission_artifact_digest`.
The producer run must be a successful completed
`.github/workflows/software-candidate.yml` run at the same source SHA, using
the sanctioned direct `workflow_dispatch` event. Every consumer
downloads that artifact by ID, checks its API ID/name/digest/run/head SHA,
repository, workflow identity, event, and run attempt, and verifies the
workflow run conclusion before reading the receipt. The current `main` copy
is reusable-only and has no in-repository caller; the producer must provide the
reviewed sanitized direct-dispatch producer before this promotion can be
dispatched with real inputs. A reusable `workflow_call` producer is not
accepted without a separately bound, reviewed caller identity.

The artifact must contain exactly `rights-admission.json` and
`dependency-license-inventory.json`. The receipt conforms to
`robot_sf.software_rights_admission.v1` and binds the exact
candidate artifact and manifest, including the optional rights-scoped
materialization identity when present. The verifier requires that identity
to use the producer's closed eight-field contract and requires the transported
dependency report to repeat the same values. It declares
`robot_sf.software_sanitized_candidate.v1`, names
`scripts/validation/software_release_rights_policy.v1.json` and its SHA-256,
and records a zero-finding passed
`check_distribution_licenses.py --strict-asset-rights --source-tree-ref ...`
gate. It must also bind the separate #8021 supported-surface dependency
report: schema `robot-sf.dependency-license-inventory.v1`, the exact
candidate manifest and sanitized source-tree digests, the canonical
dependency policy/profile paths and their SHA-256 values, the report SHA-256,
and `unresolved_count: 0` from the exact
`check_dependency_license_inventory.py --fail-on-unresolved` command. The
publisher parses and re-hashes that transported report independently; a
receipt field containing only a claimed report digest is insufficient. A
rights-only receipt without this zero-unresolved dependency admission is not
a software release admission and is rejected. The publisher validates all
bindings in every package-producing job; it never accepts a locally authored
or caller-provided rights assertion. Until the producer and #8021 emit this complete
receipt, the promotion workflow is intentionally technically ineligible and
cannot upload the current candidate.

The sanitized release wheel must advertise exactly the twelve supported extras
plus the supported `all` aggregator. The standalone `rllib` extra (as well as
`orca` or any unknown extra) is outside this release surface and is rejected by
the publisher even when every surrounding receipt hash has been refreshed.

## First promotion

1. Run the rights-clean candidate workflow and retain its exact
   rights-admission artifact identity. Then run the reusable
   [`software-candidate.yml`](../.github/workflows/software-candidate.yml)
   workflow at the same reviewed source ref. Record its exact artifact ID, artifact
   name, `sha256:` artifact digest, workflow run ID and attempt, source SHA,
   and package version from the run summary as well as the five rights-admission
   values.
2. Dispatch `software-promotion.yml` with the candidate and rights-admission
   values. The
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
6. The post-production cold-verification job downloads both PyPI formats from
   the public Simple index and compares their filenames, sizes, and SHA-256
   bytes with the immutable candidate. It stores a separate verification
   receipt; production is not accepted as complete without this cold download.

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
then verifies that receipt and skips a duplicate upload; the downstream
cold-verification job still downloads both formats and rechecks their exact
hashes. Missing, expired,
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
