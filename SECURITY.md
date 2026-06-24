# Security Policy

## Supported Versions

Robot SF is a research codebase. Security fixes are handled on the active `main`
branch and on release branches only when a maintainer explicitly marks that branch
as supported.

| Version or branch | Security support |
| --- | --- |
| `main` | Supported |
| Active maintainer-designated release branches | Supported when named in release notes |
| Historical tags, archived experiments, abandoned branches | Not supported |

## Reporting A Vulnerability

Please report suspected vulnerabilities privately through GitHub's private
vulnerability reporting when it is available for this repository: open the
repository **Security** tab and choose **Report vulnerability**. If that option
is not available to you, email the package maintainer address listed in
`pyproject.toml` with the subject prefix `[Robot SF Security]` and ask for a
secure reporting channel before sharing exploit details in an issue, pull
request, or public discussion.

Include:

- affected commit, branch, or release tag;
- dependency, script, workflow, or runtime path involved;
- reproduction steps or a minimal proof of concept when safe to share privately;
- expected impact, including whether the finding requires network access,
  untrusted input, model artifacts, maps, or generated benchmark outputs.

Maintainers will acknowledge private reports when practical, triage the affected
scope, and either fix the issue, document an accepted risk, or close the report
as not applicable to supported code. Do not publish details until maintainers
have had a reasonable chance to respond and coordinate disclosure.

## Research Scope Notes

Many scripts in this repository are offline research, benchmark, or artifact
generation tools. Security findings are still reviewed, but triage considers the
actual trust boundary: public package code, GitHub Actions, dependency manifests,
artifact hydration, and code paths that process untrusted inputs are higher risk
than local one-off analysis scripts with trusted inputs.

See [docs/security_triage.md](docs/security_triage.md) for triage rules,
accepted-risk handling, and the current Ruff security baseline policy.
