#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Usage: scripts/dev/run_worktree_shared_venv.sh [options] -- <uv-run-command> [args...]

Run a targeted validation command from the current checkout while reusing a shared virtualenv.
The helper pins imports to this worktree by prepending PYTHONPATH=$PWD:$PWD/fast-pysf and sets UV_NO_SYNC=1 so
`uv run` does not silently resync or rewrite the shared environment.
Do not start a fresh linked-worktree command with bare `uv run`: it can materialize a partial local
`.venv` that then shadows the shared environment. Route commands through this wrapper; intentionally
local environments must be created with `scripts/dev/bootstrap_worktree.sh` first.
For linked worktrees, the helper also derives a per-worktree COVERAGE_FILE unless one is already
set, preventing parallel focused pytest runs from sharing output/coverage/.coverage state.

Because the shared env is reused without resync (UV_NO_SYNC=1), a stale owning-checkout .venv can
lag the current worktree source. Before interpreter or pytest commands, the helper checks the
installed vendored `pysocialforce` package against this checkout and fails closed with the exact
`uv sync --all-extras --reinstall-package robot-sf` repair command when it is stale. After that
check, PYTHONPATH=$PWD:$PWD/fast-pysf makes the checkout source authoritative for the command. Use
--standalone for commands that do not import project code, or --no-freshness-check only after
confirming the environment matches.
Pinned tool binaries are a separate boundary (issue #8250): `uv run` executes the requested tool
from the selected venv, so a stale venv would silently run a drifted binary. Before proceeding,
the freshness preflight compares the resolved `<venv>/bin/<tool>` version against the exact `==`
pin in the active checkout's pyproject; on mismatch it fails closed with the exact `--venv`
remedy instead of running. One preflight log line (with elapsed ms) is always emitted.

Standalone commands with a verified boundary that does not import project packages can use
--standalone. That mode skips the project-source freshness check and does not add the worktree root
to PYTHONPATH, while still reusing the shared environment for third-party dependencies.

Options:
  --venv PATH            Shared virtualenv path exported as UV_PROJECT_ENVIRONMENT. Defaults to an
                         initialized current-worktree .venv, otherwise the main checkout .venv.
  --profile NAME         Dependency import profile checked before the command. Defaults to core;
                         use all-extras (or a named pyproject extra) when the command needs it.
  --scratch-dir PATH     Use PATH for temporary files and default uv/XDG caches for this run.
  --standalone           Run a command that is verified not to import project packages. This skips
                         the dependency-profile and project-source checks, but still applies the
                         pinned-tool freshness gate; it does not prepend the worktree root to
                         PYTHONPATH.
  --no-freshness-check   Retained for compatibility; checkout-local fast-pysf source takes
                          precedence after the interpreter package-coherence check. Also accepted
                          via ROBOT_SF_VENV_FRESHNESS_CHECK=skip. Bypasses all freshness gates; use
                          only after confirming the environment matches.
  The wrapped command must begin with an executable after `--`. Nested `uv run` overlay or
  isolated-environment options (`--with*`, `--isolated`, `--python`) are rejected because this
  helper cannot verify freshness for the resulting environment.
  Absolute or relative paths to `uv` are parsed like the plain `uv run` form; other explicit
  tool paths retain their compatibility skip boundary.
  -h, --help             Show this help message.

Environment:
  ROBOT_SF_CI_MIN_FREE_BYTES
                         Minimum free bytes required for the effective temporary directory
                         (default: 1073741824).

Examples:
  scripts/dev/run_worktree_shared_venv.sh -- pytest tests/test_ci_script_contract.py -q
  scripts/dev/run_worktree_shared_venv.sh --venv ../robot_sf_ll7/.venv -- ruff check scripts/dev
  scripts/dev/run_worktree_shared_venv.sh --standalone -- \
    python scripts/dev/check_docs_evidence_integrity.py --files docs/dev_guide.md

Use a full local .venv plus PR_READY_MODE=final for final PR proof; this helper is for quick,
targeted validation in sibling worktrees.
EOF
}

check_scratch_capacity() {
  local space_path="$1"
  local minimum_bytes="${ROBOT_SF_CI_MIN_FREE_BYTES:-1073741824}"
  local minimum_kib
  local df_output
  local available_kib

  if ! [[ "$minimum_bytes" =~ ^[0-9]+$ ]]; then
    echo "ERROR: ROBOT_SF_CI_MIN_FREE_BYTES must be a non-negative integer (got '$minimum_bytes')." >&2
    return 2
  fi
  minimum_kib=$(( (10#$minimum_bytes + 1023) / 1024 ))

  if [[ ! -d "$space_path" || ! -w "$space_path" ]]; then
    echo "ERROR: shared-venv scratch path is missing or not writable: $space_path" >&2
    echo "Use --scratch-dir /path/on/disk or set TMPDIR to a writable disk-backed directory." >&2
    return 2
  fi
  if ! command -v df >/dev/null 2>&1 || ! command -v awk >/dev/null 2>&1; then
    echo "ERROR: shared-venv scratch preflight requires both 'df' and 'awk'." >&2
    return 2
  fi
  if ! df_output="$(df -Pk "$space_path" 2>/dev/null)"; then
    echo "ERROR: shared-venv scratch preflight could not inspect filesystem capacity: $space_path" >&2
    echo "Use --scratch-dir /path/on/disk and retry." >&2
    return 2
  fi
  available_kib="$(awk 'NR > 1 && $4 ~ /^[0-9]+$/ { print $4; exit }' <<<"$df_output")"
  if ! [[ "$available_kib" =~ ^[0-9]+$ ]]; then
    echo "ERROR: shared-venv scratch preflight could not parse available space for: $space_path" >&2
    echo "Use --scratch-dir /path/on/disk and retry." >&2
    return 2
  fi
  if (( 10#$available_kib < minimum_kib )); then
    echo "ERROR: shared-venv scratch preflight failed: ${available_kib} KiB available at $space_path; ${minimum_kib} KiB required." >&2
    echo "The uv command was not started. Free space or retry with --scratch-dir /path/on/disk." >&2
    echo "For a deliberately bounded run only, lower ROBOT_SF_CI_MIN_FREE_BYTES explicitly." >&2
    return 2
  fi
  echo "Shared-venv scratch preflight passed: path=$space_path available=${available_kib}KiB required=${minimum_kib}KiB" >&2
}

configure_scratch_dir() {
  local requested_path="$1"
  local scratch_root

  if ! mkdir -p "$requested_path"; then
    echo "ERROR: could not create shared-venv scratch directory: $requested_path" >&2
    echo "Choose a writable disk-backed path and retry." >&2
    return 2
  fi
  if ! scratch_root="$(cd "$requested_path" 2>/dev/null && pwd -P)"; then
    echo "ERROR: could not resolve shared-venv scratch directory: $requested_path" >&2
    return 2
  fi
  if ! mkdir -p "$scratch_root/tmp" "$scratch_root/uv-cache" "$scratch_root/xdg-cache" "$scratch_root/mplconfig"; then
    echo "ERROR: could not create shared-venv scratch subdirectories under: $scratch_root" >&2
    echo "Choose a writable disk-backed path and retry." >&2
    return 2
  fi

  export ROBOT_SF_CI_SCRATCH_DIR="$scratch_root"
  export TMPDIR="$scratch_root/tmp"
  export UV_CACHE_DIR="$scratch_root/uv-cache"
  export XDG_CACHE_HOME="$scratch_root/xdg-cache"
  export MPLCONFIGDIR="$scratch_root/mplconfig"
  echo "Using shared-venv scratch directory: $scratch_root" >&2
}

venv_override=""
dependency_profile="core"
skip_freshness=""
standalone=""
scratch_dir="${ROBOT_SF_CI_SCRATCH_DIR:-}"
cmd=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)
      if [[ $# -lt 2 ]]; then
        echo "--venv requires a path." >&2
        exit 2
      fi
      venv_override="$2"
      shift 2
      ;;
    --profile)
      if [[ $# -lt 2 || -z "${2:-}" ]]; then
        echo "--profile requires a dependency profile name." >&2
        exit 2
      fi
      dependency_profile="$2"
      shift 2
      ;;
    --scratch-dir)
      if [[ $# -lt 2 || -z "${2:-}" ]]; then
        echo "--scratch-dir requires a path." >&2
        exit 2
      fi
      scratch_dir="$2"
      shift 2
      ;;
    --standalone)
      standalone=1
      shift
      ;;
    --no-freshness-check)
      skip_freshness=1
      shift
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    --)
      shift
      cmd=("$@")
      break
      ;;
    *)
      cmd=("$@")
      break
      ;;
  esac
done

if [[ ${#cmd[@]} -eq 0 ]]; then
  show_help >&2
  exit 2
fi
if [[ "${cmd[0]}" == -* ]]; then
  echo "ERROR: the wrapped command must start with an executable, not an option: ${cmd[0]}" >&2
  echo "Put wrapper options before '--' and provide the command after it." >&2
  exit 2
fi

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

git_common_dir="$(git rev-parse --git-common-dir)"
if [[ "$git_common_dir" != /* ]]; then
  git_common_dir="$(cd "$repo_root/$git_common_dir" && pwd)"
fi
main_repo_root="$(cd "$git_common_dir/.." && pwd)"

if [[ -n "$scratch_dir" ]]; then
  configure_scratch_dir "$scratch_dir"
fi
check_scratch_capacity "${TMPDIR:-/tmp}"

if [[ -n "$venv_override" ]]; then
  venv_path="$venv_override"
elif [[ -x "$repo_root/.venv/bin/python" ]]; then
  venv_path="$repo_root/.venv"
else
  venv_path="$main_repo_root/.venv"
fi
if [[ "$venv_path" != /* ]]; then
  venv_path="$repo_root/$venv_path"
fi

if [[ ! -x "$venv_path/bin/python" ]]; then
  echo "Shared virtualenv not found or incomplete: $venv_path" >&2
  echo "Create it with 'uv sync --all-extras' in the owning checkout, or use a local .venv." >&2
  exit 2
fi

check_dependency_profile() {
  local report
  if ! report="$("$venv_path/bin/python" \
    "$repo_root/scripts/dev/check_worktree_optional_deps.py" \
    --profile "$dependency_profile" 2>&1)"; then
    echo "ERROR: shared-venv dependency profile '$dependency_profile' is incomplete in $venv_path." >&2
    printf '%s\n' "$report" >&2
    echo "Run 'cd $repo_root && scripts/dev/bootstrap_worktree.sh', then rerun this command." >&2
    return 2
  fi
}

if [[ -z "$standalone" ]]; then
  check_dependency_profile
fi

is_project_interpreter_command() {
  local tool_name="${1##*/}"
  case "$tool_name" in
    python|python[0-9]*|pytest|py.test)
      return 0
      ;;
  esac
  return 1
}

check_project_package_freshness() {
  local venv_path="$1"
  local checker="$repo_root/scripts/dev/check_fast_pysf_runtime.py"
  if [[ ! -f "$checker" ]]; then
    echo "ERROR: shared-venv project package freshness checker is missing: $checker" >&2
    echo "Restore the checkout's scripts/dev/check_fast_pysf_runtime.py, then retry." >&2
    return 2
  fi

  local report
  if ! report="$(env -u PYTHONPATH "$venv_path/bin/python" "$checker" 2>&1)"; then
    echo "ERROR: shared-venv project package freshness preflight failed in $venv_path." >&2
    printf '%s\n' "$report" >&2
    echo "Remedy: run 'uv sync --all-extras --reinstall-package robot-sf' in the owning checkout, then retry." >&2
    return 2
  fi
  echo "Shared-venv project package freshness preflight passed: package=pysocialforce venv=$venv_path" >&2
}

check_shared_venv_freshness() {
  local venv_path="$1"
  local src_pkg="$repo_root/fast-pysf/pysocialforce"

  # PYTHONPATH makes the checkout source authoritative after the interpreter
  # package-coherence check below. Pinned tool binaries (issue #8250) are
  # checked below: the requested tool runs from the selected venv, so its
  # version is compared against the active checkout pin.
  local start_ms=""
  start_ms="$(date +%s%3N 2>/dev/null)" || start_ms=""
  local tool="${cmd[0]:-}"
  local freshness_parse_error=""

  # The helper's documented examples also use ``-- uv run <tool>``. Walk the
  # uv-run options so flags or option values cannot hide the tool whose
  # selected-venv version will actually be used. Unknown options fail closed:
  # guessing where the nested command starts would make the freshness gate
  # appear to pass while checking the wrong binary.
  if [[ "${tool##*/}" == "uv" && "${cmd[1]:-}" == "run" ]]; then
    local uv_run_index=2
    local uv_run_arg=""
    tool=""
    while (( uv_run_index < ${#cmd[@]} )); do
      uv_run_arg="${cmd[$uv_run_index]}"
      case "$uv_run_arg" in
        --)
          ((uv_run_index++))
          tool="${cmd[$uv_run_index]:-}"
          break
          ;;
        -m|--module|-s|--script|--gui-script)
          # These modes execute through Python rather than a selected-venv
          # tool entry point; preserve the interpreter skip boundary.
          tool="python"
          break
          ;;
        --isolated|--with|--with-editable|--with-requirements|-w|-p|--python)
          freshness_parse_error="unsupported environment-changing uv run option '$uv_run_arg'"
          break
          ;;
        --with=*|--with-editable=*|--with-requirements=*|--isolated=*|-w=*|-p=*|--python=*)
          freshness_parse_error="unsupported environment-changing uv run option '$uv_run_arg'"
          break
          ;;
        --extra|--no-extra|--group|--no-group|--only-group|--no-editable-package|--env-file|--package|--python-platform|--index|--default-index|-i|--index-url|--extra-index-url|-f|--find-links|--index-strategy|--keyring-provider|-P|--upgrade-package|--upgrade-group|--resolution|--prerelease|--fork-strategy|--exclude-newer|--exclude-newer-package|--no-sources-package|--reinstall-package|--refresh-package|--link-mode|-C|--config-setting|--config-settings-package|--no-build-isolation-package|--no-build-package|--no-binary-package|--allow-insecure-host|--cache-dir|--color|--directory|--project|--config-file)
          if (( uv_run_index + 1 >= ${#cmd[@]} )); then
            freshness_parse_error="uv run option '$uv_run_arg' is missing its value"
            break
          fi
          ((uv_run_index+=2))
          ;;
        --extra=*|--no-extra=*|--group=*|--no-group=*|--only-group=*|--no-editable-package=*|--env-file=*|-w=*|--with=*|--with-editable=*|--with-requirements=*|--package=*|--python-platform=*|--index=*|--default-index=*|-i=*|--index-url=*|--extra-index-url=*|-f=*|--find-links=*|--index-strategy=*|--keyring-provider=*|-P=*|--upgrade-package=*|--upgrade-group=*|--resolution=*|--prerelease=*|--fork-strategy=*|--exclude-newer=*|--exclude-newer-package=*|--no-sources-package=*|--reinstall-package=*|--refresh-package=*|--link-mode=*|-C=*|--config-setting=*|--config-settings-package=*|--no-build-isolation-package=*|--no-build-package=*|--no-binary-package=*|-p=*|--python=*|--allow-insecure-host=*|--cache-dir=*|--color=*|--directory=*|--project=*|--config-file=*)
          ((uv_run_index++))
          ;;
        --all-extras|--no-dev|--no-default-groups|--all-groups|--only-dev|--no-editable|--exact|--no-env-file|--isolated|--active|--no-sync|--locked|--frozen|--all-packages|--no-project|--no-index|-U|--no-cache|--refresh|--reinstall|--compile-bytecode|--no-build-isolation|--no-build|--no-binary|--upgrade|--no-sources|--managed-python|--no-managed-python|--no-python-downloads|-n|--quiet|-q|--verbose|-v|--system-certs|--offline|--no-progress|--no-config|-h|--help)
          ((uv_run_index++))
          ;;
        -*)
          freshness_parse_error="unrecognized uv run option '$uv_run_arg'"
          break
          ;;
        *)
          tool="$uv_run_arg"
          break
          ;;
      esac
    done
  fi

  local skip_reason=""
  local pin=""

  freshness_elapsed_ms() {
    local end_ms=""
    end_ms="$(date +%s%3N 2>/dev/null)" || end_ms=""
    if [[ "$start_ms" =~ ^[0-9]+$ && "$end_ms" =~ ^[0-9]+$ ]]; then
      printf '%s' "$((end_ms - start_ms))"
    else
      printf 'unknown'
    fi
  }

  if [[ -n "$freshness_parse_error" ]]; then
    echo "ERROR: Shared-venv tool freshness preflight could not identify the nested uv tool: $freshness_parse_error elapsed_ms=$(freshness_elapsed_ms) venv=$venv_path" >&2
    echo "Use a supported 'uv run' option form or rerun with --no-freshness-check only after confirming the environment matches." >&2
    return 2
  fi

  if [[ -z "$tool" ]]; then
    skip_reason="empty-command"
  elif [[ "$tool" == *"/"* ]]; then
    skip_reason="explicit-path"
  elif [[ ! "$tool" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    skip_reason="unsafe-tool-name"
  fi
  if [[ -z "$skip_reason" ]]; then
    case "$tool" in
      python|python[0-9]*|pytest|py.test|bash|sh|dash|uv|git)
        skip_reason="interpreter-or-shell"
        ;;
    esac
  fi
  if [[ -z "$skip_reason" && ! -f "$repo_root/pyproject.toml" ]]; then
    skip_reason="no-pin-manifest"
  fi
  if [[ -z "$skip_reason" ]]; then
    local tool_esc="${tool//./\\.}"
    pin="$(grep -E "^[[:space:]]*\"${tool_esc}==[0-9A-Za-z._+-]+\"" "$repo_root/pyproject.toml" 2>/dev/null | head -n 1 | sed -E 's/^[^=]*==([0-9A-Za-z._+-]+).*/\1/' || true)"
    if [[ -z "$pin" ]]; then
      skip_reason="unpinned"
    fi
  fi
  if [[ -z "$skip_reason" && ! -x "$venv_path/bin/$tool" ]]; then
    echo "ERROR: Shared-venv pinned tool is absent from the selected environment: $venv_path/bin/$tool (active checkout pins $tool==$pin)." >&2
    echo "Selected environment: $venv_path (active checkout: $repo_root)." >&2
    echo "Remedy: install the pinned tool into the selected environment, re-sync the owning checkout, or pass an explicit --venv containing it." >&2
    echo "To bypass after confirming the environment matches, rerun with --no-freshness-check." >&2
    return 2
  fi
  if [[ -n "$skip_reason" ]]; then
    if [[ -z "$standalone" && "$skip_reason" == "interpreter-or-shell" ]] \
      && is_project_interpreter_command "$tool"; then
      if ! check_project_package_freshness "$venv_path"; then
        return 2
      fi
    fi
    echo "Shared-venv tool freshness preflight skipped: tool=${tool:-none} reason=$skip_reason elapsed_ms=$(freshness_elapsed_ms) venv=$venv_path" >&2
    return 0
  fi

  local resolved=""
  resolved="$("$venv_path/bin/$tool" --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -n 1 || true)"
  if [[ -z "$resolved" ]]; then
    echo "Shared-venv tool freshness preflight skipped: tool=$tool reason=unparsable-version elapsed_ms=$(freshness_elapsed_ms) venv=$venv_path" >&2
    return 0
  fi
  if [[ "$resolved" == "$pin" ]]; then
    echo "Shared-venv tool freshness preflight passed: tool=$tool resolved=$resolved pin==$pin elapsed_ms=$(freshness_elapsed_ms) venv=$venv_path" >&2
    return 0
  fi

  echo "ERROR: Shared-venv tool freshness preflight failed: tool '$tool' resolved to $resolved but the active checkout pins $tool==$pin." >&2
  echo "Selected environment: $venv_path (active checkout: $repo_root)." >&2
  if [[ "$venv_path" == "$repo_root/.venv" && "$main_repo_root/.venv" != "$venv_path" ]]; then
    echo "Remedy: rerun with --venv $main_repo_root/.venv, or re-sync the owning checkout and retry." >&2
  else
    echo "Remedy: re-sync the owning checkout ('uv sync --all-extras' where this venv lives), or pass an explicit --venv." >&2
  fi
  echo "To bypass after confirming the environment matches, rerun with --no-freshness-check." >&2
  return 2
}

if [[ -z "$skip_freshness" && "${ROBOT_SF_VENV_FRESHNESS_CHECK:-}" != "skip" ]]; then
  if ! check_shared_venv_freshness "$venv_path"; then
    exit 2
  fi
fi

export UV_PROJECT_ENVIRONMENT="$venv_path"
export UV_NO_SYNC=1
# An explicit shared --venv override must stay authoritative across nested
# common_setup.sh consumers: pin VIRTUAL_ENV so an incomplete worktree-local
# .venv cannot shadow the shared environment (issue #7823).
if [[ -n "$venv_override" ]]; then
  export VIRTUAL_ENV="$venv_path"
  export ROBOT_SF_EXPLICIT_VENV_OVERRIDE="$venv_path"
fi
if [[ -z "$standalone" ]]; then
  export PYTHONPATH="$repo_root:$repo_root/fast-pysf${PYTHONPATH:+:$PYTHONPATH}"
fi

if [[ -z "${COVERAGE_FILE:-}" && "$git_common_dir" != "$repo_root/.git" ]]; then
  worktree_id="$(printf '%s' "$repo_root" | git hash-object --stdin | cut -c1-12)"
  export COVERAGE_FILE="$repo_root/output/coverage/.coverage.${worktree_id}"
fi

exec uv run "${cmd[@]}"
