#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -eq 1 && ( "$1" == "--help" || "$1" == "-h" ) ]]; then
  echo "Usage: scripts/dev/ci_install_headless_packages.sh <package> [package...]"
  echo "Installs only packages that are missing from the runner."
  echo "Each apt phase is bounded by CI_HEADLESS_APT_PHASE_TIMEOUT_SECONDS (default: 300; maximum: 600)."
  exit 0
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/dev/ci_install_headless_packages.sh <package> [package...]" >&2
  exit 2
fi

phase_timeout_seconds="${CI_HEADLESS_APT_PHASE_TIMEOUT_SECONDS:-300}"
if ! [[ "${phase_timeout_seconds}" =~ ^[1-9][0-9]*$ ]] || (( phase_timeout_seconds > 600 )); then
  echo "ci_install_headless_packages error=invalid_phase_timeout value=${phase_timeout_seconds} expected=integer_1_to_600" >&2
  exit 2
fi

missing=()
for package_name in "$@"; do
  if dpkg-query -W -f='${Status}' "${package_name}" 2>/dev/null | grep -q "install ok installed"; then
    echo "ci_install_headless_packages present package=${package_name}"
  else
    echo "ci_install_headless_packages missing package=${package_name}"
    missing+=("${package_name}")
  fi
done

if [[ ${#missing[@]} -eq 0 ]]; then
  echo "ci_install_headless_packages all requested packages already installed"
  exit 0
fi

apt_options=(
  -o Acquire::Retries=2
  -o Acquire::http::Timeout=20
  -o Acquire::https::Timeout=20
  -o Dpkg::Use-Pty=0
)

export DEBIAN_FRONTEND=noninteractive
export APT_LISTCHANGES_FRONTEND=none

apt_source_hosts() {
  local apt_output="$1"
  local line host
  local -a hosts=()

  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" =~ https?://([^/[:space:]]+) ]]; then
      host="${BASH_REMATCH[1]%%:*}"
      if [[ -n "$host" && " ${hosts[*]} " != *" ${host} "* ]]; then
        hosts+=("$host")
      fi
    fi
  done <<< "$apt_output"

  if [[ ${#hosts[@]} -eq 0 ]]; then
    printf 'unknown'
  else
    local IFS=,
    printf '%s' "${hosts[*]}"
  fi
}

emit_apt_failure() {
  local phase="$1"
  local status="$2"
  local elapsed_seconds="$3"
  local apt_output="$4"
  local failure_class="$5"
  local sources
  sources="$(apt_source_hosts "$apt_output")"

  echo "ci_install_headless_packages error=${failure_class} rc=${status} phase=${phase} timeout_seconds=${phase_timeout_seconds} elapsed_seconds=${elapsed_seconds} packages=${missing[*]} sources=${sources}" >&2
  echo "::error title=Headless package ${phase} failed::class=${failure_class} rc=${status} timeout_seconds=${phase_timeout_seconds} elapsed_seconds=${elapsed_seconds} packages=${missing[*]} sources=${sources}" >&2
}

apt_update_output=""
apt_update_rc=0
apt_update_started_seconds=$SECONDS
if apt_update_output=$(CI_STEP_TIMEOUT_SECONDS="${phase_timeout_seconds}" \
  bash "${script_dir}/ci_step_timer.sh" "Headless package apt update" \
  sudo apt-get "${apt_options[@]}" update 2>&1); then
  apt_update_rc=0
else
  apt_update_rc=$?
fi
printf '%s\n' "$apt_update_output"
apt_update_elapsed_seconds=$((SECONDS - apt_update_started_seconds))

if [[ "$apt_update_rc" -ne 0 ]]; then
  if [[ "$apt_update_rc" -eq 124 ]]; then
    emit_apt_failure "update" "$apt_update_rc" "$apt_update_elapsed_seconds" "$apt_update_output" "apt_update_timeout"
    exit "$apt_update_rc"
  fi

  is_apt_403_text() {
    local text="${1,,}"
    [[ "$text" =~ (403([^0-9]|$)|forbidden) ]]
  }

  is_official_apt_host() {
    case "$1" in
      ubuntu.com|*.ubuntu.com|debian.org|*.debian.org|canonical.com|*.canonical.com)
        return 0
        ;;
      *)
        return 1
        ;;
    esac
  }

  apt_update_other_errors=()
  third_party_403_hosts=()
  pending_host=""
  pending_status=""
  record_pending_apt_error() {
    [[ -z "$pending_host" ]] && return
    if is_apt_403_text "$pending_status" && ! is_official_apt_host "$pending_host"; then
      third_party_403_hosts+=("$pending_host")
    else
      apt_update_other_errors+=("$pending_host")
    fi
    pending_host=""
    pending_status=""
  }

  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" == Err:* ]]; then
      record_pending_apt_error
      if [[ "$line" =~ https?://([^/[:space:]]+) ]]; then
        pending_host="${BASH_REMATCH[1]%%:*}"
        pending_status="$line"
        if is_apt_403_text "$line"; then
          record_pending_apt_error
        fi
      else
        apt_update_other_errors+=("unknown")
      fi
      continue
    fi

    if [[ "$line" == "W: Failed to fetch"* ]]; then
      record_pending_apt_error
      if [[ "$line" =~ https?://([^/[:space:]]+) ]]; then
        pending_host="${BASH_REMATCH[1]%%:*}"
        pending_status="$line"
        if is_apt_403_text "$line"; then
          record_pending_apt_error
        fi
      else
        apt_update_other_errors+=("unknown")
      fi
      continue
    fi

    if [[ "$line" == Get:* || "$line" == Hit:* || "$line" == Ign:* ]]; then
      record_pending_apt_error
      continue
    fi

    if [[ -n "$pending_host" ]]; then
      pending_status+=" $line"
      if is_apt_403_text "$line"; then
        record_pending_apt_error
      fi
    fi
  done <<< "$apt_update_output"
  record_pending_apt_error

  if [[ ${#apt_update_other_errors[@]} -gt 0 || ${#third_party_403_hosts[@]} -eq 0 ]]; then
    IFS=,
    echo "ci_install_headless_packages apt_update_classification=failed_sources sources=${apt_update_other_errors[*]:-unknown}" >&2
    unset IFS
    emit_apt_failure "update" "$apt_update_rc" "$apt_update_elapsed_seconds" "$apt_update_output" "apt_update_failed"
    exit "$apt_update_rc"
  fi

  IFS=,
  echo "ci_install_headless_packages warning=ignored_third_party_apt_403 hosts=${third_party_403_hosts[*]}"
  unset IFS
fi

echo "ci_install_headless_packages installing packages=${missing[*]}"
apt_install_output=""
apt_install_rc=0
apt_install_started_seconds=$SECONDS
if apt_install_output=$(CI_STEP_TIMEOUT_SECONDS="${phase_timeout_seconds}" \
  bash "${script_dir}/ci_step_timer.sh" "Headless package apt install" \
  sudo apt-get "${apt_options[@]}" install -y --no-install-recommends "${missing[@]}" 2>&1); then
  apt_install_rc=0
else
  apt_install_rc=$?
fi
printf '%s\n' "$apt_install_output"
apt_install_elapsed_seconds=$((SECONDS - apt_install_started_seconds))

if [[ "$apt_install_rc" -ne 0 ]]; then
  if [[ "$apt_install_rc" -eq 124 ]]; then
    emit_apt_failure "install" "$apt_install_rc" "$apt_install_elapsed_seconds" "$apt_install_output" "apt_install_timeout"
  else
    emit_apt_failure "install" "$apt_install_rc" "$apt_install_elapsed_seconds" "$apt_install_output" "apt_install_failed"
  fi
  exit "$apt_install_rc"
fi
