# External Repository Setup Assistant

Robot SF does not silently vendor optional research/reference repositories. Use the local setup
assistant to discover supported external repos, clone a pinned commit under a gitignored staging
path, and write compact provenance manifests without committing upstream code.

```bash
uv run python scripts/tools/manage_external_repos.py list
uv run python scripts/tools/manage_external_repos.py explain <name>
uv run python scripts/tools/manage_external_repos.py check <name>
uv run python scripts/tools/manage_external_repos.py stage <name>
```

Staged clones live under `third_party/external_repos/<name>/`, which is gitignored. Generated
manifests default to `output/external_repos/manifests/`, which is local-only through the canonical
`output/` ignore rule. A manifest records upstream URL, optional fork URL, pinned SHA, staged
commit, license and compatibility decisions, redistribution decision, intended Robot SF use,
validation command, file count, size, aggregate tree checksum, and sample file hashes.

The first registered external repository is `sicnav`, the MIT-licensed Safe Interactive Crowd
Navigation reference implementation used by the existing SICNav wrapper. The registry pins an
explicit upstream commit and stages from upstream until an `ll7` fork is created. The tracked
SICNav defaults use `third_party/external_repos/sicnav`, not the legacy
`third_party/external_mpc_repos/sicnav` path.

DR-MPC is intentionally not registered in this pipeline. GitHub metadata reported no license for
`https://github.com/James-R-Han/DR-MPC` on 2026-06-22, so public fork, redistribution, and pinned
local staging remain blocked. Keep DR-MPC as reference-only/source-side-first until an explicit
license decision and source-harness proof exist. The issue #3366 audit is tracked in
`docs/context/issue_3366_external_mpc_staging_audit.md`.

## License-Gated Fork Procedure

Use a fork only when the license decision allows redistribution through a public `ll7` fork.
Permissive and common copyleft licenses such as MIT, BSD, Apache, and GPL may be fork candidates
after the license text is verified. Research-only, private, no-redistribution, unclear, or
credential-gated repositories must not be publicly forked; pin the upstream commit directly or use a
private mirror approved for the license.

Before adding a `RepoSpec`:

1. Record the upstream URL, observed license, license URL or file, and access date in
   `docs/templates/external_repo_audit.md`.
2. Decide the inclusion model: core runtime dependencies belong in vendor/subtree workflows;
   optional research or reference repos belong in this gitignored pinned-clone pipeline.
3. If a public fork is allowed, fork to `https://github.com/ll7/<name>` for durability. The pinned
   SHA is still mandatory and must refer to the exact commit staged for Robot SF work.
4. If a public fork is not allowed, leave `fork_url` unset or point to an approved private mirror.
   Do not publish restricted code as part of the staging process.
5. Add the `RepoSpec` with license compatibility, redistribution decision, intended use, and a
   validation command that proves the clone works for its Robot SF use case.
6. Run `stage <name>`, then run the validation command from the staged checkout or the
   repo-specific smoke test. Treat missing clones as skip-gated in tests, not as fallback success.

The assistant fails closed when the staging destination is not gitignored or when the pinned SHA
cannot be fetched from the declared source.
