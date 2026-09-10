# `robot-sf doctor` troubleshooting

[Back to Documentation Index](../README.md)

Plain-language summary: `robot-sf doctor` checks that your computer can run
Robot SF (a social-navigation simulator for robots moving among pedestrians)
and tells you exactly what to fix when something is missing. This page maps
every stable check identifier the command can report to its meaning, the
smallest safe remedy, and how to verify the fix.

Run it first whenever setup behaves unexpectedly:

```bash
uv run robot-sf doctor
uv run robot-sf doctor --format json --skip-env-smoke --skip-quickstart-smoke
```

The shell examples are cross-platform `uv` commands. Where a later example
uses POSIX-shell syntax such as `VAR=value`, `touch`, `rm`, or a pipeline, use
the equivalent command for PowerShell or another platform shell.

## Statuses and exit code

Each check reports one status: `ok` (PASS), `skipped` (SKIP), `missing_optional`
(WARN), or `failed` (FAIL). Checks marked **required** fail the whole report
when they fail. The command exits `1` only when the overall status is `failed`;
a report containing only optional warnings still exits `0`. The `gymnasium` and
`numpy` import checks are required because they are core project dependencies;
the other import and binary probes remain warnings when absent.

The JSON report keeps the stable schema `robot_sf_bench.doctor.v1`. Every check
object carries a stable `name` identifier that this page binds to; new fields
are only ever added, never renamed. Unknown future identifiers are
an undocumented contract failure until this page and its parity test are updated.

## Required checks

| Check `name` | Meaning | If it fails | Minimal remedy and verification |
| --- | --- | --- | --- |
| `python` | Active interpreter is Python 3.11 or newer. | Setup cannot run; exit `1`. | Install Python ≥ 3.11 (for example `uv python install 3.11`) and re-run with it. Verify: `python --version`. |
| `uv_bootstrap` | The `uv` package manager is on `PATH`. | Dependency install and most documented commands fail; exit `1`. | Install `uv` (see the hint printed by the check), then reopen the shell. Verify: `uv --version`. |
| `binary:git` | `git` is on `PATH`. | Checkout, worktree, and provenance tooling fail; exit `1`. | Install `git` with the OS package manager. Verify: `git --version`. |
| `binary:uv` | `uv` is on `PATH` (same binary as `uv_bootstrap`, probed separately). | Same as `uv_bootstrap`; exit `1`. | Same as `uv_bootstrap`. Verify: `uv --version`. |
| `import:gymnasium` | Core environment API dependency is importable. | Environment construction cannot run; exit `1`. | Run `uv sync`. Verify: `uv run python -c "import gymnasium"`. |
| `import:numpy` | Core numerical dependency is importable. | Numerical runtime paths cannot run; exit `1`. | Run `uv sync`. Verify: `uv run python -c "import numpy"`. |
| `quickstart` | Manifest-declared quickstart examples exist and, with smoke enabled, run headlessly. | Onboarding path is broken; exit `1`. | Missing manifest: restore `examples/examples_manifest.yaml` from the checkout. Missing files: restore the listed paths. Smoke failure: fix the listed example first (run it directly for the full traceback). Verify: re-run `uv run robot-sf doctor`. |

## Optional capabilities (WARN at most, never exit `1`)

| Check `name` | Unlocks | If missing | Minimal remedy and verification |
| --- | --- | --- | --- |
| `binary:ffmpeg` | Video export from recordings. | Video writing is unavailable; everything else works. | Install `ffmpeg` with the OS package manager. Verify: `ffmpeg -version`. |
| `binary:gh` | GitHub CLI workflows (issue/PR helpers). | Local work is unaffected; automation scripts that call `gh` fail. | Install `gh` and authenticate (`gh auth login`) only if you need those workflows. Verify: `gh --version`. |
| `binary:docker` | Container-based reproduction. | Only container workflows are unavailable. | Install Docker only if you need container repro. Verify: `docker --version`. |
| `binary:jq` | Shell JSON parsing in helper scripts. | Scripts fall back or fail only where they shell out to `jq`. | Install `jq` with the OS package manager. Verify: `jq --version`. |
| `import:pygame` | On-screen visualization and interactive playback. | Headless flows still work; windows cannot open. | `uv sync --extra viz`. Verify: `uv run python -c "import pygame"`. |
| `import:matplotlib` | Plotting and figure examples. | Plots cannot render. | `uv sync --extra viz`. Verify: `uv run python -c "import matplotlib"`. |
| `model_artifacts` | Bundled PPO checkpoints under `model/` used by the pre-trained demo. | Pre-trained demo cannot run; training from scratch is unaffected. | Restore `model/` from the release bundle, or run flows that do not need the checkpoint. Verify: re-run doctor and check `present`. |
| `optional_extras` | Importable probes for the declared public extras `viz`, `maps`, `benchmark`, `gpu`, `training`, `recurrent`, `rllib`, `progress`, `analytics`, `browser`, `sacadrl`, `socnav`, and `criticality`. The declared `all` extra is an aggregate selector, not a separate probe. | Only the feature slice behind the missing group is unavailable. | Run `uv sync --extra <group>` for the named slice. `orca` and `analysis` are not declared extras: `rvo2` is supplied by `uv sync --group dev`, while `seaborn` is part of `uv sync --extra viz`. Verify: re-run doctor and check the group row. |

## Informational and smoke checks (no action unless failed)

| Check `name` | Meaning | If it fails |
| --- | --- | --- |
| `robot_sf_package` | Installed `robot-sf` version and import source. Always `ok`; informational, no action. | Cannot fail; if the source path looks wrong, reinstall with `uv sync`. |
| `environment` | Reports `MPLBACKEND`, `SDL_VIDEODRIVER`, `DISPLAY` for headless triage. Always `ok`; informational, no action. | Cannot fail. For headless runs set `MPLBACKEND=Agg SDL_VIDEODRIVER=dummy DISPLAY=`. |
| `workspace` | Reports working directory, workspace root, `.git` presence, and `local.machine.md` context. Always `ok`; informational, no action. | Cannot fail; run from the repository root for the most useful paths. |
| `artifact_root` | Temporary-write probe of the artifact root (default `output/`). | Fix permissions or free disk space, then re-run. On a POSIX shell, verify with `touch output/.doctor-write && rm output/.doctor-write`; use the equivalent temporary-file check in other shells. |
| `env_smoke` | One reset/step through the public environment factory (`--skip-env-smoke` reports `skipped`). | The environment itself is broken: check the `error` field, confirm `gymnasium` and the physics backend are installed (`uv sync`), and run the failing call directly for the traceback. |

## Machine-readable note

Automation should key on the stable `name` field and the `robot_sf_bench.doctor.v1`
schema, not on human-readable text. The friendly footer points here
(`docs/troubleshooting/doctor.md`) using a repository-relative path so the
reference works offline.
