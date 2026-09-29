# Disposable CI runners

This directory contains the **proposed** runner setup for issue #9905. Do not
activate it until the PR has a Claude `MERGE` verdict and the repository owner
has checked the settings below. The PR does not register a runner or change
GitHub settings.

## Trust and access prerequisites

- Set the repository Actions setting **Require approval for all external
  contributors** before registration. Keep that setting enabled. Personal
  repositories have no runner groups; the script uses repository-scoped
  registration, and the runner's job-started hook is the execution boundary.
- Leave the repository variable `ROBOT_SF_SELF_HOSTED_CI_ENABLED` unset during
  review and installation. It defaults to hosted CI. Set it to the exact string
  `true` only after the reviewed runners are online; unset it to return all
  routed jobs to `ubuntu-latest` immediately. Changing this variable is a
  separate repository-settings action.
- The workflow sends only an `ll7`-started push or same-repository PR, with
  `ll7` also starting any rerun, to `robot-sf-ci-ephemeral`. Forks, bots,
  `pull_request_target`, `workflow_run`, comments, and manual dispatch stay on
  `ubuntu-latest`. Do not add untrusted triggers to this route.
- A job-started hook baked into the image checks the repository, actor,
  triggering actor, event name, and event JSON before any job step. It rejects
  fork heads and PRs authored by anyone other than `ll7`, even if a fork edits
  `runs-on` to name the runner directly. A missing or malformed event fails
  closed. The workflow expression remains the routing layer.
- Routed jobs have `permissions: contents: read`, use only the default
  `GITHUB_TOKEN`, and must not receive repository secrets, SSH keys, cluster
  credentials, or private-ops files.

## Image and isolation

`setup.sh build` derives `robot-sf-ci-runner:2.336.0` from the pinned official
`ghcr.io/actions/actions-runner` image digest in the script. It installs the
headless CI packages before runtime, removes the image's passwordless sudo
policy and supplemental groups, and stores the runner program under
`/opt/robot-sf-runner`. The hook lives outside the writable runner directory;
the image sets `ACTIONS_RUNNER_HOOK_JOB_STARTED` to its path. Refresh and review
the digest and package set when the runner version changes; `--disableupdate`
makes image updates explicit.
The image pins Node.js 22.23.2 from the official archive and checks its SHA-256;
Ubuntu's Node 18 package cannot parse the browser-runtime test modules.

Each job gets a new container with a read-only root, a 2 GiB tmpfs at
`/home/runner` for runner binaries and the Python tool cache, a 512 MiB tmpfs
at `/tmp`, and a 512 MiB tmpfs at `/home/runner/_work/_temp` for `RUNNER_TEMP`.
The nested tmpfs covers the runner's temporary directory even though its
parent, `/home/runner/_work`, is a fresh anonymous Docker volume. The checkout,
virtual environment, uv and pip caches, and build scratch use that volume.
`RUNNER_TOOL_CACHE` points into `/home/runner`; `TMPDIR`, `PIP_CACHE_DIR`, and
`UV_CACHE_DIR` point into the per-job volume so wheel builds and pip installs do
not fill the 512 MiB `/tmp` tmpfs. Tmpfs usage counts against the container's
8 GiB memory limit. The [pinned checkout action](https://github.com/actions/checkout/blob/3d3c42e5aac5ba805825da76410c181273ba90b1/src/git-auth-helper.ts)
briefly writes its token under `RUNNER_TEMP` even when credential persistence is
disabled; a host crash cannot leave that file in the anonymous volume. There
is a read-only image symlink from `/opt/hostedtoolcache` to `RUNNER_TOOL_CACHE`:
the setup-python binary embeds the former path in its ELF RUNPATH, and tests
that clear their environment still need to load its adjacent `libpython`.
There is no volume source or host bind mount, so jobs never share a workspace. The
container uses UID/GID 1001, no Linux capabilities, no privilege escalation,
no Docker socket, and limits of 4 CPUs, 8 GiB RAM, and 512 processes. The
runner executes at `nice 10`. Pytest uses at most two workers; OpenBLAS and OpenMP use one
thread per worker so they fit under the process limit. Its ephemeral
registration handles one job; `docker run --rm`
destroys the container and its anonymous volume, and the supervisor creates
the next one. Compare `docker volume ls` before and after a job to confirm
cleanup. If a host crash leaves an unused volume, `docker volume prune` is the
cleanup command. Both runner and probe use the dedicated IPv4 Docker bridge
`robot-sf-ci-egress` (`172.30.244.0/24`, gateway `172.30.244.1`, bridge
`br-robot-sf-ci`). IPv6 is disabled. Before starting a slot, `setup.sh start`
checks that a probe container cannot ping the host gateway or a University of
Augsburg address, and can reach GitHub and PyPI over HTTPS. The supervisor
repeats the network configuration check and isolation probe before each
replacement container. If either check fails, it logs the failure and retries
after 60 seconds without requesting a registration token or starting a runner.
Before fetching each registration token, the supervisor holds a host-local
disk-admission lock and requires 20 GiB free per running slot plus the new slot
on Docker's root filesystem (`docker info -f '{{.DockerRootDir}}'` and `df`).
If the disk check fails or cannot read capacity, it logs the failure and retries
after 60 seconds. The probe does not replace the host firewall
rules below.

## One-time host firewall setup (author action)

After building the image, create the dedicated network with `setup.sh network`.
On each runner host, the author then installs persistent firewall rules for
that bridge. The `DOCKER-USER` chain blocks forwarded traffic to private,
link-local, loopback, and University of Augsburg IPv4 destinations. Traffic to
the host itself traverses `INPUT`, so a separate rule blocks every host IPv4
address, including the bridge gateway. Apply these rules with host
administration access; the setup script does not change iptables:

```bash
bash scripts/ci/self_hosted/setup.sh network
for destination in 10.0.0.0/8 172.16.0.0/12 192.168.0.0/16 \
  169.254.0.0/16 127.0.0.0/8 100.64.0.0/10 137.250.0.0/16; do
  sudo iptables -I DOCKER-USER 1 -i br-robot-sf-ci \
    -s 172.30.244.0/24 -d "$destination" -j DROP
done
sudo iptables -I INPUT 1 -i br-robot-sf-ci -s 172.30.244.0/24 -j DROP
```

Persist the rules with the host's firewall manager and check them after Docker
or host restarts. If Docker cannot create the chosen subnet because it overlaps
an existing route, choose a non-overlapping subnet and update the script,
rules, and probe together before proceeding. The university's
[network documentation](https://www.uni-augsburg.de/de/organisation/bibliothek/nutzen-leihen/online-medien/)
identifies `137.250.*` as its address prefix; the block above covers
`137.250.0.0/16`.

## Host commands (after review only)

Install Docker, `gh`, and `flock` on the host using the host's normal administration
process. Authenticate `gh` as an account with repository administration access.
From a reviewed checkout, run:

```bash
bash scripts/ci/self_hosted/setup.sh build
# Create the network and apply the firewall rules above before starting slots.
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

Accepted residual: `config.sh --token` places the short-lived registration
token in the argument list inside the container's own PID namespace. The token
expires within one hour and only registers a runner. It is not passed in the
host's Docker command line; this residual is accepted for this design.

GitHub's [self-hosted runner reference](https://docs.github.com/en/actions/reference/runners/self-hosted-runners)
documents ephemeral registration and version updates. Its
[secure-use guidance](https://docs.github.com/en/actions/reference/security/secure-use)
explains why a public repository's self-hosted runner needs a strict trust
boundary. GitHub's [runner-group guidance](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/manage-access)
describes repository access restrictions.
