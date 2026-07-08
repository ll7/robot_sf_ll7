# Security Triage Guidance

Robot SF is a research simulator and benchmark repository. Security review should
make risks visible without turning legacy research-script findings into noisy
blocking gates.

## Triage Priority

Treat findings as higher priority when they affect:

- GitHub Actions, dependency manifests, lockfiles, release packaging, or artifact
  hydration;
- library code under `robot_sf/` or `robot_sf_carla_bridge/` that can be reused
  by downstream users;
- parsers or loaders for maps, models, traces, videos, benchmark manifests, or
  other files that may come from outside a trusted local checkout;
- commands that execute subprocesses, shell snippets, or dynamically imported
  code.

Treat findings as lower priority when they are limited to:

- offline one-off research or analysis scripts with trusted local inputs;
- tests, examples, notebooks, archived experiments, or smoke fixtures;
- vendored or `third_party/` code where upstream provenance and local patch policy
  should be reviewed before making sweeping edits.

Lower priority does not mean ignored. It means the report should identify the
trust boundary and decide whether to fix the issue, suppress it with a narrow
rationale, or record an accepted risk.

## Vendored Third-Party Code

For vendored code, first identify whether the issue belongs upstream, in local
integration glue, or in a local patch. Prefer the smallest local mitigation that
does not make future upstream synchronization harder. Do not rewrite vendored
code solely to satisfy a scanner unless the finding affects a supported runtime
path.

## False Positives And Accepted Risks

False positives should be suppressed narrowly with a specific rule code and a
short reason near the code or in `pyproject.toml` per-file ignores. Accepted
risks should name:

- affected path, rule, dependency, or advisory;
- why the current trust boundary makes the risk acceptable;
- what would change the decision, such as exposing the path to untrusted input,
  publishing supported library behavior, or using it in CI or releases.

Do not use broad category ignores for new security findings. Ruff `S` findings are
ratcheted via explicit per-code ignores (the issue #3477 baseline) rather than a broad
category suppression; triage new findings into fixes or scoped per-file ignores.

## Ruff Security Baseline

`pyproject.toml` currently selects Ruff Bandit rules (`S`) and ratchets the existing
backlog via explicit per-code ignores (S101, S105, S108, ... under issue #3477) rather
than a global `S` ignore, so any new Bandit finding outside that baseline fails lint. The
`security-baseline` GitHub Actions workflow runs Ruff with ignores overridden:

```bash
uv run ruff check --config 'lint.ignore=[]' --select S .
```

That advisory job uploads the raw finding log. Use it to ratchet the baseline by
fixing findings and replacing broad suppression with scoped per-file ignores.
Keep ratcheting the per-code baseline down: fix findings and replace each code-level
ignore with justified scoped per-file exceptions.

## Dependency Visibility

Dependabot is configured for the root `uv.lock`, the `fast-pysf/uv.lock`
lockfile, and GitHub Actions. Dependency review runs on pull requests that
change dependency manifests or lockfiles. Tool findings should be triaged under
the same rules: supported runtime and CI paths first, vendored or research-only
paths with documented rationale.
