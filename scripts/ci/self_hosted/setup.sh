#!/usr/bin/env bash
# Build and supervise disposable, repository-scoped GitHub Actions runners.
set -euo pipefail
set +x  # A registration token must never appear in a shell trace.

repo="ll7/robot_sf_ll7"
label="robot-sf-ci-ephemeral"
image="robot-sf-ci-runner:2.336.0"
network="robot-sf-ci-egress"
network_subnet="172.30.244.0/24"
network_gateway="172.30.244.1"
network_bridge="br-robot-sf-ci"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
host="$(hostname -s)"
# imech036 and imech039 report the short hostname with an "auxme-" prefix.
host="${host#auxme-}"
state_dir="${XDG_STATE_HOME:-$HOME/.local/state}/robot-sf-ci-runners"

usage() {
  cat <<'USAGE'
Usage: setup.sh build | network | start SLOT | stop SLOT | status SLOT

Build the pinned container image locally, then start one supervisor per slot.
Slots 1-2 are allowed on imech036 and imech039; slots 1-3 on imech156-u.
The supervisor registers one ephemeral runner per container and replaces it
after its single job. Stop a slot with one command: setup.sh stop SLOT.
USAGE
}

require_slot() {
  local limit
  case "$host" in
    imech036|imech039) limit=2 ;;
    imech156-u) limit=3 ;;
    *) echo "Unsupported host: $host" >&2; exit 2 ;;
  esac
  if [[ ! "${1:-}" =~ ^[1-3]$ ]] || (( 10#$1 > limit )); then
    echo "SLOT must be in 1..$limit on $host" >&2
    exit 2
  fi
}

slot_name() { printf 'robot-sf-ci-%s-%s' "$host" "$1"; }
pid_file() { printf '%s/%s.pid' "$state_dir" "$(slot_name "$1")"; }
log_file() { printf '%s/%s.log' "$state_dir" "$(slot_name "$1")"; }

lock_slot() {
  install -d -m 700 "$state_dir"
  chmod 700 "$state_dir"
  exec {slot_lock_fd}>"$state_dir/$(slot_name "$1").lock"
  flock -x "$slot_lock_fd"
}

unlock_slot() {
  flock -u "$slot_lock_fd"
  exec {slot_lock_fd}>&-
}

build_image() {
  # The official image includes passwordless sudo and docker-group membership.
  # Remove both, preload the CI system packages, then run it with no privileges,
  # no socket, and a read-only root filesystem.
  docker build --tag "$image" --file - "$script_dir" <<'DOCKERFILE'
FROM ghcr.io/actions/actions-runner@sha256:0cfdcc701ce933c6d243c6b0b2da767366dc9f2e99961d4c3754b0b78084cdda
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ffmpeg gh \
    libglib2.0-0t64 libgl1 fonts-dejavu-core jq poppler-utils iputils-ping curl \
    && rm -rf /var/lib/apt/lists/* \
    && usermod -G '' runner \
    && rm -f /etc/sudoers \
    && mkdir -p /opt/robot-sf-runner \
    && cp -a /home/runner/. /opt/robot-sf-runner/ \
    && chown -R runner:runner /opt/robot-sf-runner \
    && install -d -o runner -g runner /home/runner/_work
COPY --chown=runner:runner setup.sh /usr/local/bin/robot-sf-runner
COPY --chown=runner:runner job_started_hook.sh /usr/local/libexec/robot-sf-job-started.sh
COPY --chown=runner:runner network_probe.sh /usr/local/libexec/robot-sf-network-probe
ENV ACTIONS_RUNNER_HOOK_JOB_STARTED=/usr/local/libexec/robot-sf-job-started.sh
USER 1001:1001
ENTRYPOINT ["/usr/local/bin/robot-sf-runner", "container"]
DOCKERFILE
}

run_container() {
  local runner_name="$1"
  local token
  IFS= read -r token || true
  if [[ -z "$token" ]]; then
    echo "Registration token was unavailable" >&2
    exit 1
  fi
  cd /home/runner
  cp -a /opt/robot-sf-runner/. /home/runner/
  install -d -m 700 /home/runner/_work/_tmp /home/runner/_work/_uv_cache \
    /home/runner/_work/_tool
  ./config.sh --unattended --ephemeral --disableupdate --replace \
    --url "https://github.com/$repo" --token "$token" \
    --name "$runner_name" --labels "$label" --work _work
  unset token
  exec nice -n 10 ./run.sh
}

supervise() {
  local name
  name="$(slot_name "$1")"
  while true; do
    if ! ensure_network || ! probe_network; then
      echo "Runner $name network isolation check failed; retrying after 60 seconds" >&2
      sleep 60
      continue
    fi
    # The API response goes directly through the pipe to the container's
    # config step. Neither a token file nor a token-bearing Docker argument is
    # created. --rm removes the container and its anonymous workspace volume.
    if ! check_docker_disk; then
      echo "Runner $name Docker disk capacity check failed; retrying after 60 seconds" >&2
      sleep 60
      continue
    fi
    if gh api -X POST "repos/$repo/actions/runners/registration-token" --jq .token |
      docker run --rm --interactive --name "$name" \
        --user 1001:1001 --read-only --network "$network" \
        --dns 1.1.1.1 --dns 9.9.9.9 \
        --tmpfs /home/runner:rw,exec,nosuid,nodev,uid=1001,gid=1001,size=1g \
        --mount type=volume,dst=/home/runner/_work \
        --tmpfs /tmp:rw,exec,nosuid,nodev,uid=1001,gid=1001,size=512m \
        --cap-drop ALL --security-opt no-new-privileges \
        --pids-limit 512 --cpus 4 --memory 8g --memory-swap 8g \
        --env HOME=/home/runner \
        --env RUNNER_TOOL_CACHE=/home/runner/_work/_tool \
        --env UV_CACHE_DIR=/home/runner/_work/_uv_cache \
        --env TMPDIR=/home/runner/_work/_tmp \
        --env PYTEST_NUM_WORKERS=2 --env OPENBLAS_NUM_THREADS=1 \
        --env OMP_NUM_THREADS=1 \
        "$image" "$name"; then
      echo "Runner $name finished its job; replacing its container"
    else
      echo "Runner $name exited or failed to register; retrying after 15 seconds" >&2
    fi
    sleep 15
  done
}

check_docker_disk() {
  local docker_root available_kib
  docker_root="$(docker info -f '{{.DockerRootDir}}')" || return 1
  [[ -n "$docker_root" ]] || return 1
  available_kib="$(df -Pk -- "$docker_root" | awk 'NR == 2 {print $4}')" || return 1
  if [[ ! "$available_kib" =~ ^[0-9]+$ ]]; then
    echo "Could not determine free space on Docker root: $docker_root" >&2
    return 1
  fi
  if (( available_kib < 20 * 1024 * 1024 )); then
    echo "Docker root has $available_kib KiB free; 20 GiB required: $docker_root" >&2
    return 1
  fi
}

ensure_network() {
  local settings
  if ! docker network inspect "$network" >/dev/null 2>&1; then
    docker network create --driver bridge --subnet "$network_subnet" \
      --gateway "$network_gateway" \
      --opt "com.docker.network.bridge.name=$network_bridge" "$network" >/dev/null
  fi
  settings="$(docker network inspect --format '{{json .}}' "$network")"
  jq -e --arg subnet "$network_subnet" --arg gateway "$network_gateway" \
    --arg bridge "$network_bridge" \
    '.Driver == "bridge" and .EnableIPv6 == false and
     .IPAM.Config[0].Subnet == $subnet and .IPAM.Config[0].Gateway == $gateway and
     .Options["com.docker.network.bridge.name"] == $bridge' \
    >/dev/null <<<"$settings" || {
      echo "Dedicated Docker network has unexpected settings: $network" >&2
      return 1
    }
}

probe_network() {
  docker run --rm --network "$network" --dns 1.1.1.1 --dns 9.9.9.9 --user 0:0 \
    --entrypoint /usr/local/libexec/robot-sf-network-probe \
    "$image" "$network_gateway" 137.250.1.254 || {
      echo "Network isolation probe failed; refusing to start a runner" >&2
      return 1
    }
}

is_supervisor() {
  local pid="$1" slot="$2" args
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  args="$(ps -o args= -p "$pid" 2>/dev/null || true)"
  [[ "$args" == *"$script_dir/setup.sh supervise $slot"* ]]
}

start_slot() {
  local slot="$1" pid_path pid
  pid_path="$(pid_file "$slot")"
  lock_slot "$slot"
  if [[ -f "$pid_path" ]]; then
    pid="$(<"$pid_path")"
    if is_supervisor "$pid" "$slot"; then
      echo "Slot $slot is already running (pid $pid)" >&2
      exit 1
    fi
    rm -f "$pid_path"
  fi
  docker image inspect "$image" >/dev/null
  ensure_network
  probe_network
  nohup "$script_dir/setup.sh" supervise "$slot" >"$(log_file "$slot")" 2>&1 </dev/null &
  pid=$!
  printf '%s\n' "$pid" >"$pid_path"
  unlock_slot
  echo "Started slot $slot (pid $pid); log: $(log_file "$slot")"
}

stop_slot() {
  local slot="$1" pid_path pid
  pid_path="$(pid_file "$slot")"
  lock_slot "$slot"
  if [[ -f "$pid_path" ]]; then
    pid="$(<"$pid_path")"
    if is_supervisor "$pid" "$slot"; then
      kill "$pid"
    fi
    rm -f "$pid_path"
  fi
  docker stop --time 30 "$(slot_name "$slot")" >/dev/null 2>&1 || true
  unlock_slot
  echo "Stopped slot $slot"
}

case "${1:-}" in
  build) [[ $# -eq 1 ]] || { usage; exit 2; }; build_image ;;
  network) [[ $# -eq 1 ]] || { usage; exit 2; }; ensure_network ;;
  supervise) [[ $# -eq 2 ]] || { usage; exit 2; }; require_slot "$2"; supervise "$2" ;;
  start|stop|status)
    [[ $# -eq 2 ]] || { usage; exit 2; }
    require_slot "$2"
    case "$1" in
      start) start_slot "$2" ;;
      stop) stop_slot "$2" ;;
      status)
        if [[ -f "$(pid_file "$2")" ]] &&
          is_supervisor "$(<"$(pid_file "$2")")" "$2"; then
          echo "Slot $2 running"
        else
          echo "Slot $2 stopped"
        fi
        ;;
    esac
    ;;
  container) [[ $# -eq 2 ]] || exit 2; run_container "$2" ;;
  -h|--help) usage ;;
  *) usage >&2; exit 2 ;;
esac
