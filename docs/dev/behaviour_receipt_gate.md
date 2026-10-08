# Behaviour receipt gate integration

Behaviour PRs carry a compact `<!-- behaviour-change-receipt:v2 ... -->` JSON
header in their GitHub body. Rows and detailed classifications live in a committed
`receipts/behaviour/*.json` file, read from the event's exact PR source SHA.
The payload has `schema_version: behaviour-change-rows.v1`, `rows` and
`classifications`. It does **not** embed its enclosing commit SHA: committing a
file containing that SHA would create a circular identity requirement.

The header uses `schema_version: behaviour-change-receipt-header.v2`. It carries
`head_sha`, `scheduler` (Slurm job and source SHA), `scope_sha256`, external
`artifact` URI/SHA-256, `baseline`, `vehicle`, `exceptions`, `totals`,
`interaction_audit` and `refute_review` URI/verdict/head. `rows_artifact` gives
the committed `path` and SHA-256 of its exact bytes. `classifications` is a
compact `{count, sha256}` summary: the SHA-256 hashes the full classification
array serialized with sorted keys and `separators=(",", ":")`. Moving detailed
classifications to the payload keeps the header small even if many episodes
regress. The schema definitions for both transport objects live alongside the
full receipt schema in `scripts/ci/behaviour_receipt.schema.json`.

The gate rejects inline all-row v1 bodies, oversized bodies (GitHub's 65,536
character limit), payloads over 32 MiB, uncommitted files, symlinks, paths outside
`receipts/behaviour/`, byte-digest mismatches and inconsistent classification
summaries. It reconstructs the full receipt and applies the existing coverage,
source, baseline, execution, totals, exception and classification checks.
Changes to dirty worktree payload bytes do not replace the source Git blob.
Commit the rows/classifications first, then put the resulting source SHA in the
PR header and source/job/review identities. Committing the header is unnecessary.

## Scope and dependency rule

Production triggers cover planner/adapters, simulator/dynamics, maps/scenarios,
benchmark configuration and writers/metrics, algorithm/baseline/planner/robot
configs, model weights/registry, sensors, pedestrian logic, shared runtime code,
training, prediction, feature extractors, all `fast-pysf/` code and benchmark
runners under `scripts/`. Markdown files are exempt under every prefix, as are
docs-only changes and the owner inventory itself. Mixed planner/inventory PRs
remain in scope.

A separate base-owned `DEPENDENCY_RECEIPTS_ENABLED` rule examines `uv.lock`
and `pyproject.toml` (root and `fast-pysf/`) through the existing Dependabot policy parsers. A change to
`torch`, `stable-baselines3`, `numpy` or `gymnasium` requires a receipt; other
package bumps, comments and metadata edits stay exempt. Added/removed packages
and changed dependency-group requirements count. Comparison uses the source's
merge base, so runtime dependency updates made only on main do not taint a stale
branch. Ambiguous/malformed dependency inputs fail closed. An author ruling can
change this one constant without switching off production path triggers.

## Trusted policy and source checkout

CI keeps GitHub's default synthetic merge checkout with complete history/tags.
Existing seed-holdout and added-file guards therefore compare the combined tree
against the current base. `BEHAVIOUR_PR_HEAD_SHA` separately binds the receipt
to `github.event.pull_request.head.sha`; source and merge identities are distinct.
The published baseline resolves to the latest published software release tag,
excluding model and other artifact releases.

The workflow creates a detached worktree at the immutable event base SHA and
executes **that base's** validator, schema, owner inventory, algorithm registry
and dependency parser. It independently collects the source diff from Git rather
than trusting a head-owned changed-file list. Head policy edits cannot weaken
admission in the same PR. The head contract checker does not import receipt
policy; the base validator is a separate blocking workflow step. Before the gate
is first installed on base, CI reports it inactive and never substitutes head
policy. Gate changes take effect on subsequent PRs after landing on base.

## Owner inventory and evidence limits

The release integration inventory at
`configs/benchmarks/releases/behaviour_gate_0_1_0.json` binds 14 named arms and
51 scenario/map slots from the empty-world sweep's `main` and `width` inputs.
Scenario names identify slots, including distinct configurations sharing SVGs.
It includes arm-to-registry mappings, repository-relative map paths and
campaign/matrix/map digests. Receipts bind its canonical JSON SHA-256.

The vehicle is `differential_drive_r1m`, the current differential-drive body with
1.0 m radius. The existing 2.0 m narrow doorway remains an infeasible-by-design
exception, linked to its declaration at the integration base. Release integration
owns inventory maintenance; missing inventory blocks behaviour admission.

Each row binds its algorithm and command execution mode. Native-capable
algorithms require native commands; adapter-only algorithms require supported
adapter commands. `controller_executed` must be true and `fallback`/`degraded`
false. Every new failure/collision must be classified with evidence. No success
rate is required, and this gate does not test pedestrian interaction.

Digests establish byte identity, not scientific truth. A fabricated receipt can
be internally consistent: independent review must verify its linked job,
execution artifacts, body/config/source provenance and classifications. Review
plus the digest/binding checks remain the safeguards against receipt forgery.
Synthetic test rows prove validator behaviour only.
