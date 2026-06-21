# External Repository Audit Template

Use this template before registering an external code repository for Robot SF staging. Keep paths
repository-relative. Do not commit upstream source code unless the license and a maintainer decision
explicitly choose a vendored/subtree inclusion model.

## Summary

- Repository name:
- Upstream URL:
- Optional fork or private mirror URL:
- Access date:
- Pinned commit SHA:
- Related issue or PR: none / `Issue #<id>` / `PR #<id>`
- Intended Robot SF use:
- Inclusion model: vendor/subtree / gitignored pinned clone / reference-only / reject

## License And Redistribution

- Observed license:
- License URL or file:
- Citation or notice requirement:
- Commercial, research-only, or non-redistribution limits:
- Public `ll7` fork decision: allowed / blocked / unclear
- License compatibility decision: compatible / blocked / unclear
- Redistribution decision: public fork allowed / private mirror only / no redistribution / unclear
- Decision rationale:

## Source Contract

- Canonical upstream files that define the contract:
- Expected entrypoint, package, or executable:
- Required runtime or environment:
- Expected inputs:
- Expected outputs:
- Known API, kinematics, checkpoint, or data assumptions:
- Verdict: integrate next / prototype only / assessment only / reject

## Staging And Manifest

- Staging path: `third_party/external_repos/<name>/`
- Manifest path: `output/external_repos/manifests/<name>.provenance.json`
- Checksum algorithm: SHA-256
- Expected manifest fields:
  - upstream URL
  - optional fork URL
  - pinned SHA
  - staged commit
  - license note
  - license compatibility decision
  - redistribution decision
  - intended Robot SF use
  - validation command
  - aggregate tree checksum
- Verification command:

## Smoke Test Convention

- Staged-check command:
- Robot SF wrapper or adapter smoke command:
- Skip-if-not-staged behavior:
- Fail-closed condition for missing dependencies:
- Difference between source-harness proof and Robot SF wrapper proof:

## Follow-Up

- Required follow-up issue: none / `Issue #<id>`
- Remaining provenance gaps:
- Remaining runtime or adapter gaps:
- Context note, registry, or benchmark report to update:

## Validation Checklist

- [ ] Upstream URL, access date, and pinned commit SHA are recorded.
- [ ] License, citation, fork, compatibility, and redistribution decisions are recorded.
- [ ] Inclusion model is explicit and matches the license decision.
- [ ] Staging path is under `third_party/external_repos/<name>/` and gitignored.
- [ ] Manifest path is local-only under `output/external_repos/manifests/`.
- [ ] Pinned SHA is reachable from the declared source.
- [ ] Validation command or skip-gated smoke test is recorded.
- [ ] No restricted source code is committed unless a maintainer explicitly chose vendoring.
