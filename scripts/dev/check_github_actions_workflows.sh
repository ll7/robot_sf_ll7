#!/usr/bin/env bash
# Canonical entry point to lint GitHub Actions workflows with a pinned actionlint release.
#
# Supported platforms: Linux (x86_64, arm64) and macOS (x86_64, arm64).
# Integrity: verifies the pinned release manifest and archive, then verifies every selected
# executable's exact version and platform-specific SHA-256 before use.
#
# Usage:
#   bash scripts/dev/check_github_actions_workflows.sh [workflow_files...]

set -euo pipefail

ACTIONLINT_VERSION="1.7.12"
EXPECTED_CHECKSUMS_SHA256="433028cf0ba3c42163ea1a668dedce30fcdbe84fe912b1a5e288c006eab8a4f5"
EXPECTED_BINARY_SHA256_LINUX_AMD64="c872d6db8c6bf83a8eaa704fc93999f027d55dffbc63b8a6abdccb47df5f4cd4"
EXPECTED_BINARY_SHA256_LINUX_ARM64="ac0323433c2853ec3fb978c611430c5b3dc5d43c58d1a1ec031b00ab572beb60"
EXPECTED_BINARY_SHA256_DARWIN_AMD64="d1f7cee75ae2873609bd9567b4600bebc5315a5e733e73202987a44fafdd53b2"
EXPECTED_BINARY_SHA256_DARWIN_ARM64="8db11704dc296f096216db4db65d86cd7f0ebfdf4c38453a1da276b137b88388"

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

expected_binary_sha256() {
  local platform="$1"
  case "$platform" in
    linux_amd64) echo "$EXPECTED_BINARY_SHA256_LINUX_AMD64" ;;
    linux_arm64) echo "$EXPECTED_BINARY_SHA256_LINUX_ARM64" ;;
    darwin_amd64) echo "$EXPECTED_BINARY_SHA256_DARWIN_AMD64" ;;
    darwin_arm64) echo "$EXPECTED_BINARY_SHA256_DARWIN_ARM64" ;;
    *)
      echo "check_github_actions_workflows: no trusted actionlint binary digest for platform '${platform}'" >&2
      return 1
      ;;
  esac
}

validate_actionlint_binary() {
  local candidate="$1"
  local platform="$2"
  local source="$3"
  local failed=0

  if [[ ! -f "$candidate" ]] || [[ ! -x "$candidate" ]]; then
    echo "check_github_actions_workflows: ${source} candidate is not an executable file: ${candidate}" >&2
    return 1
  fi

  local expected_binary_sha
  expected_binary_sha="$(expected_binary_sha256 "$platform")"

  local actual_binary_sha=""
  if actual_binary_sha="$(compute_sha256 "$candidate")"; then
    if [[ "$actual_binary_sha" != "$expected_binary_sha" ]]; then
      echo "check_github_actions_workflows: ${source} binary digest mismatch!" >&2
      echo "  expected: ${expected_binary_sha}" >&2
      echo "  actual:   ${actual_binary_sha}" >&2
      return 1
    fi
  else
    echo "check_github_actions_workflows: could not hash ${source} candidate: ${candidate}" >&2
    return 1
  fi

  local version_output=""
  local actual_version=""
  if version_output="$("$candidate" -version 2>/dev/null)"; then
    actual_version="$(printf '%s\n' "$version_output" | sed -n '1p')"
  fi
  if [[ "$actual_version" != "$ACTIONLINT_VERSION" ]]; then
    echo "check_github_actions_workflows: ${source} version mismatch!" >&2
    echo "  expected: ${ACTIONLINT_VERSION}" >&2
    echo "  actual:   ${actual_version:-<empty>}" >&2
    failed=1
  fi

  [[ "$failed" -eq 0 ]]
}

find_or_install_actionlint() {
  local platform
  platform="$(detect_platform)"

  # 1. Respect an explicit binary override only when its version and bytes are trusted.
  if [[ -n "${ACTIONLINT_BIN:-}" ]]; then
    if ! validate_actionlint_binary "$ACTIONLINT_BIN" "$platform" "ACTIONLINT_BIN"; then
      exit 1
    fi
    echo "${ACTIONLINT_BIN}"
    return 0
  fi

  # 2. Revalidate cached bytes before every use.
  local cache_dir="${ROBOT_SF_ACTIONLINT_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/robot_sf/actionlint}/${ACTIONLINT_VERSION}"
  local cached_bin="${cache_dir}/actionlint"
  if [[ -e "$cached_bin" ]] || [[ -L "$cached_bin" ]]; then
    if ! validate_actionlint_binary "$cached_bin" "$platform" "cached actionlint"; then
      exit 1
    fi
    echo "$cached_bin"
    return 0
  fi

  # 3. Accept a PATH candidate only when its exact version and bytes are trusted.
  if command -v actionlint >/dev/null 2>&1; then
    local path_bin
    path_bin="$(command -v actionlint)"
    if ! validate_actionlint_binary "$path_bin" "$platform" "PATH actionlint"; then
      exit 1
    fi
    echo "$path_bin"
    return 0
  fi

  # 4. Download and verify release asset
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
  expected_archive_sha="$(awk -v archive_name="$archive_name" '$2 == archive_name { print $1 }' "$checksums_file")"
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

  local extracted_bin="${tmpdir}/actionlint"
  tar -xzf "$archive_file" -C "$tmpdir" actionlint
  chmod +x "$extracted_bin"
  if ! validate_actionlint_binary "$extracted_bin" "$platform" "downloaded actionlint"; then
    rm -rf "$tmpdir"
    exit 1
  fi

  mkdir -p "$cache_dir"
  local staged_bin="${cache_dir}/.actionlint.$$"
  cp "$extracted_bin" "$staged_bin"
  chmod +x "$staged_bin"
  mv -f "$staged_bin" "$cached_bin"
  if ! validate_actionlint_binary "$cached_bin" "$platform" "installed actionlint"; then
    rm -rf "$tmpdir"
    exit 1
  fi
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
