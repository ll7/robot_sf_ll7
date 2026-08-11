#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 1 && ( "$1" == "--help" || "$1" == "-h" ) ]]; then
  echo "Usage: scripts/dev/ci_install_headless_packages.sh <package> [package...]"
  echo "Installs only packages that are missing from the runner."
  exit 0
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/dev/ci_install_headless_packages.sh <package> [package...]" >&2
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

apt_update_output=""
apt_update_rc=0
if apt_update_output=$(sudo apt-get "${apt_options[@]}" update 2>&1); then
  apt_update_rc=0
else
  apt_update_rc=$?
fi
printf '%s\n' "$apt_update_output"

if [[ "$apt_update_rc" -ne 0 ]]; then
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
        pending_host="${BASH_REMATCH[1]}"
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
        pending_host="${BASH_REMATCH[1]}"
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
    echo "ci_install_headless_packages error=apt_update_failed rc=${apt_update_rc} sources=${apt_update_other_errors[*]:-unknown}" >&2
    unset IFS
    exit "$apt_update_rc"
  fi

  IFS=,
  echo "ci_install_headless_packages warning=ignored_third_party_apt_403 hosts=${third_party_403_hosts[*]}"
  unset IFS
fi

echo "ci_install_headless_packages installing packages=${missing[*]}"
sudo apt-get "${apt_options[@]}" install -y --no-install-recommends "${missing[@]}"
