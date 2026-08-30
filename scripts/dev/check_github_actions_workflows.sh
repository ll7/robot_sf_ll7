#!/usr/bin/env bash
# Canonical entry point to lint GitHub Actions workflows with a pinned actionlint release.
#
# Supported platforms: Linux (x86_64, arm64) and macOS (x86_64, arm64).
# Integrity: verifies the release checksums manifest against the pinned manifest digest,
# then verifies the downloaded archive against the matching manifest entry.
#
# Usage:
#   bash scripts/dev/check_github_actions_workflows.sh [workflow_files...]

set -euo pipefail

ACTIONLINT_VERSION="1.7.12"
EXPECTED_CHECKSUMS_SHA256="433028cf0ba3c42163ea1a668dedce30fcdbe84fe912b1a5e288c006eab8a4f5"

compute_sha256() {
  local file="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$file" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$file" | awk '{print $1}'
  elif command -v python3 >/dev/null 2>&1; then
    python3 -c "import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], 'rb').read()).hexdigest())" "$file"
  elif command -v python >/dev/null 2>&1; then
    python -c "import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], 'rb').read()).hexdigest())" "$file"
  else
    echo "check_github_actions_workflows: cannot compute sha256: no sha256sum, shasum, or python available" >&2
    exit 1
  fi
}

download_file() {
  local url="$1"
  local dest="$2"
  if command -v curl >/dev/null 2>&1; then
    if ! curl -sSfL "$url" -o "$dest"; then
      echo "check_github_actions_workflows: failed to download $url via curl" >&2
      exit 1
    fi
  elif command -v python3 >/dev/null 2>&1; then
    if ! python3 -c "import urllib.request, sys; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" "$url" "$dest"; then
      echo "check_github_actions_workflows: failed to download $url via python3" >&2
      exit 1
    fi
  elif command -v python >/dev/null 2>&1; then
    if ! python -c "import urllib.request, sys; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" "$url" "$dest"; then
      echo "check_github_actions_workflows: failed to download $url via python" >&2
      exit 1
    fi
  else
    echo "check_github_actions_workflows: cannot download file: neither curl nor python available" >&2
    exit 1
  fi
}

detect_platform() {
  local raw_os
  local raw_arch
  raw_os="$(uname -s)"
  raw_arch="$(uname -m)"

  local os=""
  local arch=""

  case "$raw_os" in
    Linux) os="linux" ;;
    Darwin) os="darwin" ;;
    *)
      echo "check_github_actions_workflows: unsupported OS '${raw_os}'; actionlint gate supports Linux and Darwin (macOS)" >&2
      exit 1
      ;;
  esac

  case "$raw_arch" in
    x86_64|amd64) arch="amd64" ;;
    arm64|aarch64) arch="arm64" ;;
    *)
      echo "check_github_actions_workflows: unsupported architecture '${raw_arch}'; actionlint gate supports x86_64/amd64 and arm64/aarch64" >&2
      exit 1
      ;;
  esac

  echo "${os}_${arch}"
}

find_or_install_actionlint() {
  # 1. Respect explicit binary override if provided and executable
  if [[ -n "${ACTIONLINT_BIN:-}" ]] && [[ -x "${ACTIONLINT_BIN}" ]]; then
    echo "${ACTIONLINT_BIN}"
    return 0
  fi

  # 2. Check cache directory
  local cache_dir="${ROBOT_SF_ACTIONLINT_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/robot_sf/actionlint}/${ACTIONLINT_VERSION}"
  local cached_bin="${cache_dir}/actionlint"
  if [[ -x "$cached_bin" ]]; then
    echo "$cached_bin"
    return 0
  fi

  # 3. Check system PATH if version matches
  if command -v actionlint >/dev/null 2>&1; then
    local path_ver
    path_ver="$(actionlint -version 2>/dev/null | head -n 1 || true)"
    if [[ "$path_ver" == *"${ACTIONLINT_VERSION}"* ]]; then
      command -v actionlint
      return 0
    fi
  fi

  # 4. Download and verify release asset
  local platform
  platform="$(detect_platform)"
  local archive_name="actionlint_${ACTIONLINT_VERSION}_${platform}.tar.gz"
  local release_base_url="https://github.com/rhysd/actionlint/releases/download/v${ACTIONLINT_VERSION}"
  local checksums_url="${release_base_url}/actionlint_${ACTIONLINT_VERSION}_checksums.txt"
  local archive_url="${release_base_url}/${archive_name}"

  local tmpdir
  tmpdir="$(mktemp -d 2>/dev/null || mktemp -d -t 'actionlint_tmp')"

  local checksums_file="${tmpdir}/checksums.txt"
  download_file "$checksums_url" "$checksums_file"

  local actual_manifest_sha
  actual_manifest_sha="$(compute_sha256 "$checksums_file")"
  if [[ "$actual_manifest_sha" != "$EXPECTED_CHECKSUMS_SHA256" ]]; then
    echo "check_github_actions_workflows: checksums manifest digest mismatch!" >&2
    echo "  expected: ${EXPECTED_CHECKSUMS_SHA256}" >&2
    echo "  actual:   ${actual_manifest_sha}" >&2
    rm -rf "$tmpdir"
    exit 1
  fi

  local expected_archive_sha
  expected_archive_sha="$(grep "${archive_name}" "$checksums_file" | awk '{print $1}' || true)"
  if [[ -z "$expected_archive_sha" ]]; then
    echo "check_github_actions_workflows: archive '${archive_name}' not found in verified checksums manifest" >&2
    rm -rf "$tmpdir"
    exit 1
  fi

  local archive_file="${tmpdir}/${archive_name}"
  download_file "$archive_url" "$archive_file"

  local actual_archive_sha
  actual_archive_sha="$(compute_sha256 "$archive_file")"
  if [[ "$actual_archive_sha" != "$expected_archive_sha" ]]; then
    echo "check_github_actions_workflows: archive digest mismatch for ${archive_name}!" >&2
    echo "  expected: ${expected_archive_sha}" >&2
    echo "  actual:   ${actual_archive_sha}" >&2
    rm -rf "$tmpdir"
    exit 1
  fi

  mkdir -p "$cache_dir"
  tar -xzf "$archive_file" -C "$cache_dir" actionlint
  chmod +x "$cached_bin"
  rm -rf "$tmpdir"
  echo "$cached_bin"
}

main() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  local repo_root
  repo_root="$(cd "${script_dir}/../.." && pwd)"

  local actionlint_bin
  actionlint_bin="$(find_or_install_actionlint)"

  local targets=()
  if [[ $# -gt 0 ]]; then
    targets=("$@")
  else
    local workflows_dir="${repo_root}/.github/workflows"
    if [[ ! -d "$workflows_dir" ]]; then
      echo "check_github_actions_workflows: workflows directory not found at ${workflows_dir}" >&2
      exit 1
    fi
    while IFS= read -r -d '' wf_file; do
      targets+=("$wf_file")
    done < <(find "$workflows_dir" -maxdepth 1 -type f \( -name "*.yml" -o -name "*.yaml" \) -print0 | sort -z)
  fi

  if [[ ${#targets[@]} -eq 0 ]]; then
    echo "check_github_actions_workflows: no workflow files found to check" >&2
    exit 1
  fi

  cd "$repo_root"
  "$actionlint_bin" "${targets[@]}"
}

main "$@"
