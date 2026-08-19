# Dependency update risk lanes

The repository separates automated dependency updates by compatibility risk so a tooling refresh cannot hide a runtime or data-interchange regression.

## Canonical sources

- .github/dependabot.yml defines the root update lanes.
- scripts/validation/dependabot_update_policy.v1.json maps direct packages to risk, lane, rollback, and existing CI evidence.
- scripts/dev/check_dependabot_update_policy.py validates the three surfaces together.
- .github/workflows/pr-contract-check.yml runs the checker for every pull request.
- .github/workflows/ci.yml owns the required compatibility evidence; the policy does not create a second dependency test suite.

The checker also covers the standalone fast-pysf project files. A new direct package must be added to the manifest with a reviewed class before it can pass the policy check. Unknown transitive lock rows remain visible and route through the conservative compatibility jobs.

## Update lanes

High-impact numerical and runtime packages, including NumPy, SciPy, Numba, PyArrow, PyTorch, Gymnasium, Stable-Baselines, TensorFlow, and Ray, remain individual updates. They require fast feedback, the supported Python and operating-system compatibility matrix, and the wheel installation smoke check.

Serialization and data-interchange packages use the individual serialization-data lane. The current examples are orjson, JSON Schema, PyYAML, RFC 8785, and TOML. They route through the same compatibility and installation surfaces because a serialization change can alter stored or exchanged data without changing the simulator API.

External experiment integrations use the individual experiment-integrations lane. The current examples are Weights & Biases, TensorBoard, and Optuna. These updates share compatibility coverage but are not grouped with runtime or developer tooling.

Developer-only tools use the bounded developer-tooling group. This includes Ruff, Mypy, Pylint, Pre-commit, Pytest, and related test helpers. The group is intentionally limited to tools whose review and rollback surface is the development workflow.

Security updates remain independently actionable. Every root group explicitly applies only to version updates, and no root group may opt into the Dependabot security-update scope, so urgent security work is not delayed behind a normal grouped-update cadence.

## Evidence and merge boundary

The checker identifies changed lock rows against the exact pull-request base and reports the direct risk class. A mixed direct update, an unknown direct package, a missing required CI job, or a compatibility command that loses its declared focused paths fails closed in the existing PR contract check.

The checker does not claim a benchmark result, a performance change, or a research result. Passing it means only that the dependency update is routed to the declared existing CI surfaces. Those CI jobs still need to pass before the pull request can be treated as merge-ready.

## Rollback

Close or revert one update lane and regenerate only the affected lockfile rows. The exact dependency diff and the policy report identify the smallest lane, so unrelated updates do not need to be discarded. If a new package needs a broader compatibility family, update the manifest and policy together with focused proof before grouping it.
