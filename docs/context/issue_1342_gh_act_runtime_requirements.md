# Issue #1342 gh-act Runtime Requirements

Issue: [#1342](https://github.com/ll7/robot_sf_ll7/issues/1342)
Related: [#1308](https://github.com/ll7/robot_sf_ll7/issues/1308),
[PR #1339](https://github.com/ll7/robot_sf_ll7/pull/1339),
[`docs/dev_runtime_requirements.md`](../dev_runtime_requirements.md)

## Outcome

The maintainer machine can install and invoke `gh-act` through the GitHub CLI, and `gh act` can
parse the repository pull-request workflow graph when an explicit runner image is supplied. This is
useful enough to keep evaluating, but it is not yet a supported PR-readiness path because no real
non-dry-run workflow job has been proven.

## Environment

Observed on 2026-05-19:

* `gh extension install https://github.com/nektos/gh-act` completed successfully.
* `gh act --version` reported `act version 0.2.88`.
* Docker daemon was available with server version `29.4.2`.
* `scripts/dev/check_runtime_requirements.sh` reported `ffmpeg` missing on this host, so full
  CI parity still needs a host package follow-up before relying on rendering/video paths locally.

## Commands And Results

List pull-request jobs:

```bash
gh act -l pull_request
```

Result: listed `docker-repro-smoke`, `fast-feedback`, `smoke-artifacts`,
`promoted-planner-smoke`, and `ci`.

Non-interactive caveat:

```bash
gh act pull_request -j ci --dryrun
```

Result: failed with an interactive default-image prompt. Automation must either configure
`~/.config/act/actrc` or pass an explicit platform image.

Non-interactive CI graph dry-run:

```bash
gh act pull_request -j ci --dryrun -P ubuntu-latest=catthehacker/ubuntu:act-latest
```

Result: succeeded. The dry run walked `fast-feedback`, `smoke-artifacts`, and the final `ci` gate.

Promoted planner graph dry-run:

```bash
gh act pull_request -j promoted-planner-smoke --dryrun \
  -P ubuntu-latest=catthehacker/ubuntu:act-latest
```

Result: succeeded. The dry run walked the promoted-planner smoke job steps.

## Current Boundary

`gh act` is experimental. It is useful for workflow graph validation and maybe future local
workflow debugging, but it does not replace:

```bash
scripts/dev/run_ci_local.sh
BASE_REF=origin/main scripts/dev/pr_ready_check.sh
```

Before promoting `gh act` to supported docs, run a real narrow job without `--dryrun`, record disk
usage and runtime, and verify that local differences from GitHub-hosted runners are acceptable.
