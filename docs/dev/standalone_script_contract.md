# Standalone Script Contract

Plain-language summary: two helper scripts must run with plain hosted Python
(no installed repository packages), so they keep local copies of small helpers
instead of importing them. This page lists those exemptions so future reuse
passes do not rediscover them through failed continuous integration (CI) runs.

## Sanctioned pattern (option c)

The repository uses a documented exemption list for scripts that run under
plain-Python invocation. The alternatives were considered and not chosen:

- (a) Migrating the workflows and the lease guard to `uv run` would change the
  runtime contract of lightweight CI jobs and the executable shebang script.
- (b) A vendored shim with a sync check adds machinery without removing the
  underlying runtime constraint.

Reuse authors must consult this list before delegating duplicated code out of
an exempted script into `scripts/dev/*.py` or `robot_sf.*` helpers.

## Exemption list

| Script | Plain-Python invokers | Reason | Duplicated helpers |
| --- | --- | --- | --- |
| `scripts/dev/check_docs_evidence_integrity.py` | `.github/workflows/docs-link-integrity.yml`, `.github/workflows/docs-evidence-integrity.yml` (hosted Python + only PyYAML installed) | Importing `scripts.*` raises `ModuleNotFoundError` without an editable install (regression #4926/#4929) | Local `_sha256` instead of `robot_sf.benchmark.identity.hash_utils` |
| `scripts/dev/pr_gate_lease.py` | Direct shebang execution; `test_script_is_executable_and_standalone` enforces the standalone contract | Must run via `./scripts/dev/pr_gate_lease.py --help` with stdlib only | Local `_git_common_dir`, `_repo_root` instead of `scripts/dev/git_common.py` |

## Rules for reuse authors

1. Do not add module-level `import scripts.*`, `from scripts.*`, `import robot_sf*`, or
   `from robot_sf*` to either exempted script. The regression test
   `tests/dev/test_standalone_script_contract.py` fails closed on such imports.
   Function-level dual-mode guards (package import under `if __package__` with a
   sibling-file fallback for direct execution, as in `pr_gate_lease.py`) remain
   allowed because direct execution never executes the package branch.
2. Do not change the workflow invocation form from plain `python ...` to
   `uv run ...` for these two scripts without removing the script from this
   list and updating the regression fixture in the same pull request (PR).
3. If sharing becomes valuable, prefer duplicating the small helper with a
   comment pointing here, or propose a new exemption entry. A shim or
   `uv run` migration is a separate scoped PR, not a drive-by refactor.
4. The workflow fixture parses the two workflow files and asserts the
   sanctioned plain-`python` invocation form.

## Adding or retiring an exemption

- Add a row above, state the plain-Python invoker and the runtime constraint,
  and extend the regression test's script and workflow lists.
- Retire a row only when the invoker no longer requires plain Python; change
  the invocation, remove the row, and update the test together.
