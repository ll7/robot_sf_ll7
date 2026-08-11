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

apt_update_log=$(mktemp)
cleanup_apt_update_log() {
  rm -f "$apt_update_log"
}
trap cleanup_apt_update_log EXIT

apt_update_rc=0
if sudo apt-get "${apt_options[@]}" update >"$apt_update_log" 2>&1; then
  apt_update_rc=0
else
  apt_update_rc=$?
fi
cat "$apt_update_log"

if [[ "$apt_update_rc" -ne 0 ]]; then
  apt_update_other_errors=()
  third_party_403_hosts=()
  while IFS=$'\t' read -r error_kind host; do
    [[ -z "$error_kind" ]] && continue
    if [[ "$error_kind" != "403" ]]; then
      apt_update_other_errors+=("${host:-unknown}")
      continue
    fi
    case "${host:-}" in
      ubuntu.com|*.ubuntu.com|debian.org|*.debian.org|canonical.com|*.canonical.com)
        apt_update_other_errors+=("${host:-unknown}")
        ;;
      *)
        third_party_403_hosts+=("${host:-unknown}")
        ;;
    esac
  done < <(
    awk '
      function host_from_url(url) {
        sub(/^https?:\/\//, "", url)
        sub(/\/.*$/, "", url)
        sub(/:.*/, "", url)
        return url
      }
      function emit_pending() {
        if (pending_host == "") {
          return
        }
        if (tolower(pending_status) ~ /(403([^0-9]|$)|forbidden)/) {
          print "403\t" pending_host
        } else {
          print "other\t" pending_host
        }
        pending_host = ""
        pending_status = ""
      }
      {
        line = $0
        if (line ~ /^(Err:|W: Failed to fetch)/) {
          emit_pending()
          if (match(line, /https?:\/\/[^[:space:]]+/)) {
            pending_host = host_from_url(substr(line, RSTART, RLENGTH))
            pending_status = line
          } else {
            print "other\tunknown"
          }
          next
        }
        if (line ~ /^(Get:|Hit:|Ign:)/) {
          emit_pending()
          next
        }
        if (pending_host != "") {
          pending_status = pending_status " " line
          if (tolower(line) ~ /(403([^0-9]|$)|forbidden)/) {
            emit_pending()
          }
        }
      }
      END { emit_pending() }
    ' "$apt_update_log" | sort -u
  )

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
