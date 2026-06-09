#!/usr/bin/env bash
# Resolve optional private operations overlays shared by all local worktrees.

set -euo pipefail

private_ops_repo_root() {
  if [[ -n "${ROBOT_SF_PRIVATE_OPS:-}" ]]; then
    printf '%s\n' "${ROBOT_SF_PRIVATE_OPS}"
    return 0
  fi

  local project_root
  project_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

  local machine_file
  for machine_file in \
    "${project_root}/local.machine.md" \
    "${project_root}"/local.machine.*.md; do
    [[ -f "${machine_file}" ]] || continue
    local configured
    configured="$(
      sed -n 's/^[[:space:]]*-[[:space:]]*private_ops_repo:[[:space:]]*//p' "${machine_file}" \
        | tail -n 1
    )"
    if [[ -n "${configured}" ]]; then
      configured="${configured/#~/${HOME}}"
      printf '%s\n' "${configured}"
      return 0
    fi
  done

  local sibling_root
  local common_dir
  common_dir="$(git -C "${project_root}" rev-parse --git-common-dir 2>/dev/null || echo ".git")"
  if [[ "${common_dir}" != /* ]]; then
    common_dir="${project_root}/${common_dir}"
  fi
  sibling_root="$(cd "${common_dir}/../.." && pwd)/robot_sf_ll7-private-ops"
  printf '%s\n' "${sibling_root}"
}

private_ops_require_file() {
  if [[ $# -ne 1 ]]; then
    echo "private_ops_require_file expects one repository-relative private path." >&2
    return 2
  fi

  local rel_path="$1"
  local ops_root
  ops_root="$(private_ops_repo_root)"
  local target="${ops_root}/${rel_path}"

  if [[ -f "${target}" ]]; then
    printf '%s\n' "${target}"
    return 0
  fi

  cat >&2 <<EOF
Private operations helper not found: ${target}

Configure the private Slurm/Auxme overlay with one of:
  export ROBOT_SF_PRIVATE_OPS=/path/to/robot_sf_ll7-private-ops
  - private_ops_repo: /path/to/robot_sf_ll7-private-ops

The public repository keeps only portable Slurm contracts; cluster-specific
wrappers, host policy, and machine notes live in the private overlay.
EOF
  return 127
}
