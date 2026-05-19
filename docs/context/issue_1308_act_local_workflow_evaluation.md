# Issue #1308 Act Local Workflow Evaluation 2026-05-18

Date: 2026-05-18

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1308>

## Decision

Do not adopt repository-supported `act` commands yet.

`scripts/dev/run_ci_local.sh`, `scripts/dev/ci_driver.sh`, and
`BASE_REF=origin/main scripts/dev/pr_ready_check.sh` remain the supported local validation paths.

## Evidence

Local environment check on 2026-05-18:

```bash
command -v act
```

Result: no `act` executable was found.

```bash
command -v docker
docker version --format '{{.Server.Version}}'
```

Result: Docker was present at `/usr/bin/docker`, server version `29.4.2`.

Because `act` itself was not available, no repository workflow job was executed through `act`.
Adding `.actrc`, `.secrets.example`, or a wrapper script without a real successful local `act` run
would make the workflow look supported before the key dependency has been proven on this machine.

## Supported Local Path

Use these commands instead:

```bash
scripts/dev/run_ci_local.sh lint
scripts/dev/ci_driver.sh lint
BASE_REF=origin/main scripts/dev/pr_ready_check.sh
```

The first two commands exercise the same canonical lint phase that `.github/workflows/ci.yml`
delegates to `scripts/dev/ci_driver.sh`. The full PR readiness gate remains the pre-PR proof bar.

## Follow-Up Boundary

Revisit `act` support only after a contributor machine can run a narrow target successfully, for
example a lint-only job from `.github/workflows/ci.yml`. A future adoption PR should include:

- the exact `act` command and event/job target,
- Docker image and architecture notes,
- how cache and artifact-upload steps behave under `act`,
- a clear statement that `act` is workflow-wiring smoke coverage, not a replacement for
  `scripts/dev/run_ci_local.sh` or `scripts/dev/pr_ready_check.sh`.
