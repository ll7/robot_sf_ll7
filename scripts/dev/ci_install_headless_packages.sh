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

echo "ci_install_headless_packages installing packages=${missing[*]}"
sudo apt-get "${apt_options[@]}" update
sudo apt-get "${apt_options[@]}" install -y --no-install-recommends "${missing[@]}"
