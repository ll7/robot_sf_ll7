# Disposable CI runners

This directory contains the **proposed** runner setup for issue #9905. Do not
activate it until the PR has a Claude `MERGE` verdict and the repository owner
has checked the settings below. The PR does not register a runner or change
GitHub settings.

## Trust and access prerequisites

- Keep outside collaborators' fork PRs subject to approval before workflows run.
- Limit the runner group to `ll7/robot_sf_ll7`. The script uses the repository
  registration endpoint, so its runners are scoped to this repository; confirm
  the applicable organization runner-group policy before activation.
- Leave the repository variable `ROBOT_SF_SELF_HOSTED_CI_ENABLED` unset during
  review and installation. It defaults to hosted CI. Set it to the exact string
  `true` only after the reviewed runners are online; unset it to return all
  routed jobs to `ubuntu-latest` immediately. Changing this variable is a
  separate repository-settings action.
- The workflow sends only an `ll7`-started push or same-repository PR, with
  `ll7` also starting any rerun, to `robot-sf-ci-ephemeral`. Forks, bots,
  `pull_request_target`, `workflow_run`, comments, and manual dispatch stay on
  `ubuntu-latest`. Do not add untrusted triggers to this route.
- Routed jobs have `permissions: contents: read`, use only the default
  `GITHUB_TOKEN`, and must not receive repository secrets, SSH keys, cluster
  credentials, or private-ops files.

## Image and isolation

`setup.sh build` derives `robot-sf-ci-runner:2.336.0` from the pinned official
`ghcr.io/actions/actions-runner` image digest in the script. It installs the
headless CI packages before runtime, removes the image's passwordless sudo
policy and supplemental groups, and stores the runner program under
`/opt/robot-sf-runner`. Refresh and review the digest and package set when the
runner version changes; `--disableupdate` makes image updates explicit.

Each job gets a new container with a read-only root, a tmpfs home/work directory
and `/tmp`, UID/GID 1001, no Linux capabilities, no privilege escalation, no
host bind mounts, no Docker socket, and limits of 4 CPUs, 8 GiB RAM, and 512
processes. The runner executes at `nice 10`. Its ephemeral registration handles
one job; `docker run --rm` destroys the container, and the supervisor creates
the next one. The container has ordinary outbound network access for GitHub and
package installation; review host network policy before activation.

## Host commands (after review only)

Install Docker, `gh`, and `flock` on the host using the host's normal administration
process. Authenticate `gh` as an account with repository administration access.
From a reviewed checkout, run:

```bash
bash scripts/ci/self_hosted/setup.sh build
bash scripts/ci/self_hosted/setup.sh start 1
bash scripts/ci/self_hosted/setup.sh start 2
# On imech156-u only:
bash scripts/ci/self_hosted/setup.sh start 3
```

Slots 1–2 are allowed on `imech036` and `imech039`; slots 1–3 on
`imech156-u`. Use a separate command for each desired slot. To stop one slot or
check it:

```bash
bash scripts/ci/self_hosted/setup.sh status 1
bash scripts/ci/self_hosted/setup.sh stop 1
```

`stop` terminates that slot's supervisor and its active container. Logs and PID
files are under `${XDG_STATE_HOME:-$HOME/.local/state}/robot-sf-ci-runners` on
the host. The registration token is fetched by `gh api` inside the supervisor
and piped directly to the container's registration step; it is never written to
a token file, Docker argument, or log. Do not enable shell tracing or capture
the pipe. Preserve runner diagnostic logs before removing a failed container
when troubleshooting.

GitHub's [self-hosted runner reference](https://docs.github.com/en/actions/reference/runners/self-hosted-runners)
documents ephemeral registration and version updates. Its
[secure-use guidance](https://docs.github.com/en/actions/reference/security/secure-use)
explains why a public repository's self-hosted runner needs a strict trust
boundary. GitHub's [runner-group guidance](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/manage-access)
describes repository access restrictions.
