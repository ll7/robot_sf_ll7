# Dependency update risk lanes

The repository separates automated dependency updates by compatibility risk so a tooling refresh cannot hide a runtime or data-interchange regression.

## Canonical sources

- .github/dependabot.yml defines the root update lanes.
- scripts/validation/dependabot_update_policy.v1.json maps direct packages to risk, lane, rollback, and existing CI evidence.
- scripts/dev/check_dependabot_update_policy.py validates the three surfaces together and reuses
  the coherence helper's frozen supported-profile comparison for lock changes.
- The same checker owns the workflow action-pin guard: it compares full-SHA external action refs in
  changed `.github/workflows/*.yml` or `.yaml` files against the exact base and searches the tracked
  HEAD tree for stale old refs.
- scripts/validation/dependency_coherence.v1.json maps each declaration to its required lock and supported profile owners.
- scripts/dev/check_dependency_coherence.py checks that map against the exact pull-request base.
- .github/workflows/pr-contract-check.yml runs the checker for every pull request.
- .github/workflows/ci.yml owns the required compatibility evidence; the policy does not create a second dependency test suite.

The checker also covers the standalone fast-pysf project files. A new direct package must be added to the manifest with a reviewed class before it can pass the policy check. Unknown transitive lock rows remain visible and route through the conservative compatibility jobs.

## Workflow action-pin coupling

For each changed workflow YAML file, the checker compares `uses: owner/action@<40-hex-SHA>` entries
in the exact base tree with `HEAD`. When the same external action moves from one full SHA to another,
the checker searches every tracked `HEAD` file for the old exact `action@SHA` reference. A match
blocks the pull request and names the workflow, old and new refs, and matching paths so coupled
contract tests or other active references can be updated together. New or unchanged pins and
workflow edits that do not replace a full-SHA action are not scanned as replacements.

This is a coupling check, not a replacement for the existing action-pin audit: it does not select
action versions, validate release comments, or allow tag-based refs. The PR-wide entry point is
already `.github/workflows/pr-contract-check.yml`; run it locally with:

```bash
uv run python scripts/dev/check_dependabot_update_policy.py --base-ref origin/main --json
```

The guard was added after the CodeQL drift repaired by #7903/#7910; #7616/#7618 is the earlier
immediate repair with the same class of mismatch.

## Update lanes

High-impact numerical and runtime packages, including NumPy, SciPy, Numba, PyArrow, PyTorch, Gymnasium, Stable-Baselines, TensorFlow, and Ray, remain individual updates. They require fast feedback, the supported Python and operating-system compatibility matrix, and the wheel installation smoke check.

Serialization and data-interchange packages use the individual serialization-data lane. The current examples are orjson, JSON Schema, PyYAML, RFC 8785, and TOML. They route through the same compatibility and installation surfaces because a serialization change can alter stored or exchanged data without changing the simulator API.

External experiment integrations use the individual experiment-integrations lane. The current examples are Weights & Biases, TensorBoard, and Optuna. These updates share compatibility coverage but are not grouped with runtime or developer tooling.

Developer-only tools use the bounded developer-tooling group. This includes Ruff, Mypy, Pylint, Pre-commit, Pytest, and related test helpers. The group is intentionally limited to tools whose review and rollback surface is the development workflow.

Security updates remain independently actionable. Every root group explicitly applies only to version updates, and no root group may opt into the Dependabot security-update scope, so urgent security work is not delayed behind a normal grouped-update cadence.

## Evidence and merge boundary

The checker identifies changed lock rows against the exact pull-request base and reports both the
configured direct risk class and the effective material-resolution lane. A lock row whose
before/after normalized resolution is identical remains visible as `lock_normalization` but does
not add a second direct risk class. The report retains per-profile before/after resolution and
closure digests, environment predicates, selected package identities, and configured class
evidence. An unavailable or unsupported profile fails closed. A mixed material direct update, an
unknown direct package, a missing required CI job, or a compatibility command that loses its
declared focused paths fails closed in the existing PR contract check.

The checker does not claim a benchmark result, a performance change, or a research result. Passing it means only that the dependency update is routed to the declared existing CI surfaces. Those CI jobs still need to pass before the pull request can be treated as merge-ready.

## Cross-lock coherence

The cross-lock checker answers a narrower question: did every changed dependency declaration receive the lock and profile validation owned by its project? The root `robot_sf` project and the standalone `fast-pysf` project resolve independently, even when they declare a shared package such as NumPy. A shared declaration therefore requires both independently resolved locks only when both declarations change. The map records this ownership instead of assuming that a subtree is a uv workspace member.

The report uses these fail-closed states:

| State | Meaning |
| --- | --- |
| `coherent` | Required lock owners are present and the checked supported profiles agree with the declarations. |
| `missing_lock_update` | A declaration changed without its mapped lockfile changing. |
| `declaration_lock_mismatch` | The lock root or pinned lock check does not agree with the declaration. |
| `profile_unavailable` | A required Python profile or the pinned uv resolver could not be evaluated. |
| `conflict` | Shared requirements or changed owner paths have an incompatible coupling. |
| `invalid` | The manifest, declaration, lock, or exact-base input is malformed. |

The scope classification is reported separately. It distinguishes root-only, fast-pysf-only, shared declarations with independent locks, workspace/member coupling, transitive-only lock changes, and lock normalization with no material supported-profile resolution change. A normalization report keeps the textual lock delta visible; it is not a dependency approval. The marker-only admission rule remains governed by the maintainer ruling in issue #7654, and cheap workflow admission remains gated by issue #7647.

## Lock-generation contract

When Dependabot or a maintainer changes a project declaration, regenerate only the mapped lock owner with the pinned resolver (`uv 0.11.21`), then run the coherence report and the existing lock checks:

```bash
# Root owner: uv lock --upgrade-package PACKAGE
# Standalone owner: uv lock --directory fast-pysf --upgrade-package PACKAGE
uv lock --check
uv lock --check --directory fast-pysf
uv run python scripts/dev/check_dependency_coherence.py --base-ref origin/main --json
```

The first command is the root-project check; the second is the standalone fast-pysf check. A maintainer update that touches only one declaration should not rewrite the other lock. If both declarations intentionally change a shared package, retain both lock updates and the report's independent-owner classification. The report is dependency-routing and continuous-integration integrity evidence only; it does not approve a package version, license, release, benchmark, or merge.

## Rollback

Close or revert one update lane and regenerate only the affected lockfile rows. The exact dependency diff and the policy report identify the smallest lane, so unrelated updates do not need to be discarded. If a new package needs a broader compatibility family, update the manifest and policy together with focused proof before grouping it.
