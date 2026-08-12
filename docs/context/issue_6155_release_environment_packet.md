# Issue #6155 release environment packet

This packet separates the environment declared by the future `0.0.5` tag from the environments that actually executed historical campaigns, so a lock file is not presented as runtime proof.

Status: release-preparation evidence; publication remains withheld pending an explicit maintainer approval in [Issue #6155](https://github.com/ll7/robot_sf_ll7/issues/6155).

The version-alignment exception is explicit and bounded by
`configs/releases/release_0_0_5_preparation.yaml`: it stages `0.0.5` on an
untagged tree only while approval is awaited and `publication_authorized` is
false. The marker does not authorize any tag, GitHub Release, asset, Zenodo
version, or DOI.

## Tag-side record

The candidate tree at the start of this preparation has these immutable inputs:

| Input | Recorded value |
| --- | --- |
| Candidate source commit | `dc78f373a28fd9bbb6b2444cfd5a74e698dfe48a` |
| `pyproject.toml` SHA-256 | `8659faf78035d219289bbd10cd6b0b10e83d517bd870386b4bf224a28e553d68` |
| `uv.lock` SHA-256 | `e7948867d64128ec81347bac64bd7b046904809051b98085eddeb422cb62ebc9` |
| Declared Python constraint | `>=3.11` |
| Canonical project license | `GPL-3.0-only`, matching `pyproject.toml` and the root `LICENSE` |
| Core dependency declarations | 16 entries in `[project].dependencies` |
| Optional dependency sets | 15 extras in `[project.optional-dependencies]` |
| Development dependency sets | 4 groups in `[dependency-groups]` |
| Resolved lock packages | 323 `[[package]]` entries in `uv.lock` |

The exact dependency expressions remain in the hashed `pyproject.toml`; the exact resolved versions remain in the hashed `uv.lock`. The future tag-target commit and its final hashes must be recorded again after the release-preparation PR is merged. This note intentionally does not modify the frozen evidence bundle or its checksum manifests.

Capture a machine-readable packet from the clean, exact candidate commit with:

```bash
uv run python scripts/repro/capture_release_environment.py \
  --release-tag 0.0.5 \
  --output /tmp/robot_sf_ll7-0.0.5-release-environment.json \
  --require-clean
```

The command records the commit, worktree state, project constraints, lock-resolved package versions, and the package inventory of the environment used for the capture. Its environment inventory is tag-side verification evidence; it is not a historical campaign record.

## Historical campaign-runtime boundary

Issue #6155 requires exact Python and package versions captured during every final dissertation-facing campaign. No such complete runtime package inventory is committed in this repository at this preparation point.

| Surface | What is available | Release use |
| --- | --- | --- |
| Issue #5034 | Compact evidence identifies source commit `484d3fd05a0e29da9e267fa18f817a1fe101de70`; its claim is targeted-smoke, diagnostic-only metric evidence. | Not a dissertation-facing campaign runtime record. |
| Issue #5305 | The compact archive records its source commit and campaign identity; local raw provenance contains Python `3.13.1`, but no durable complete package inventory. | Not a paper/dissertation ranking claim and not a substitute for the required final-campaign record. |
| Issue #5592 | The preregistration explicitly records `campaign_execution_allowed=false`. | No campaign runtime record should exist. |
| Dissertation-facing final campaign | The downstream re-base and environment handoff are tracked by [diss #526](https://github.com/ll7/diss/issues/526) and [diss #142](https://github.com/ll7/diss/issues/142). | Still missing from this repository’s durable release packet; publication remains blocked until the exact runtime record is attached. |

The absence is recorded as a blocker rather than reconstructed from `uv.lock`, current local output, or a different release. A later approval comment must attach the runtime records and bind them, together with the tag-side packet, to one immutable target SHA.

## Required exact-head evidence

At the frozen candidate target, attach the complete output of the commands in Issue #6155, including:

```bash
git rev-parse HEAD
git status --porcelain=v1
(
  cd docs/context/evidence/issue_6154_release_0_0_5_evidence_bundle
  sha256sum -c SHA256SUMS
)
uv run python scripts/tools/release_preflight_check.py \
  --checklist configs/benchmarks/releases/release_0_0_5_preflight_checklist.yaml \
  --fail-on-blocked
uv run python scripts/repro/verify_release_checksums.py \
  --manifest configs/releases/release_0_0_5_checksum_manifest.yaml \
  --no-download
uv run pytest tests/repro/test_release_checksum_verification.py -q
sha256sum pyproject.toml uv.lock
uv run python -VV
```

The expected structural results remain `10/10` bundle checksums, `13/13` preflight items with zero blocked, `25/25` outer-manifest entries, a passing frozen-candidate contract, zero focused-test failures, and a clean worktree before and after verification.
