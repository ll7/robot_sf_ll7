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
installed vendored `pysocialforce` package against this checkout. If it is stale in a linked
worktree using the default environment, the helper automatically creates or refreshes only that
worktree's `.venv`, checks capacity, serializes recovery, and verifies freshness before the command
starts. It never repairs the owning checkout implicitly. Explicit `--recover-stale-fast-pysf`
remains available to force the same worktree-local recovery. Use --standalone for commands that do
not import project packages, or --no-freshness-check only after confirming the environment matches.
Pinned tool binaries are a separate boundary (issue #8250): `uv run` executes the requested tool
from the selected venv, so a stale venv would silently run a drifted binary. Before proceeding,
the freshness preflight compares the resolved `<venv>/bin/<tool>` version against the exact `==`
pin in the active checkout's pyproject; on mismatch it fails closed with the exact `--venv`
remedy instead of running. One preflight log line (with elapsed ms) is always emitted.
Exact development pins are parsed structurally using host Python 3.11+ (stdlib only).
Malformed TOML or ambiguous/unsupported exact declarations fail closed; unpinned tools still skip.

Standalone commands with a verified boundary that does not import project packages can use
--standalone. That mode skips the project-source freshness check and does not add the worktree root
to PYTHONPATH, while still reusing the shared environment for third-party dependencies.

Options:
  --venv PATH            Shared virtualenv path exported as UV_PROJECT_ENVIRONMENT. Defaults to an
                         initialized current-worktree .venv, otherwise the main checkout .venv.
  --profile NAME         Dependency import profile checked before the command. Defaults to core;
                         use all-extras (or a named pyproject extra) when the command needs it.
  --scratch-dir PATH     Use PATH for temporary files and default uv/XDG caches for this run.
  --recover-stale-fast-pysf
                         Explicitly create or refresh a worktree-local .venv before the command.
                         Requires a linked worktree; cannot be combined with --venv, --standalone,
                         or a freshness bypass. Default linked-worktree runs recover automatically
                         after a stale fast-pysf detection.
  --standalone           Run a command that is verified not to import project packages. This skips
                         the dependency-profile and project-source checks, but still applies the
                         pinned-tool freshness gate; it does not prepend the worktree root to
                         PYTHONPATH.
  --isolated-ruff        Validate with the active checkout's exact Ruff pin, without a project
                         environment. Accepts only `ruff check` or `ruff format --check` with
                         safe selectors and file paths. Uses the root pyproject configuration,
                         an ephemeral /tmp tool cache, and no shared/project environment writes.
                         Cannot be combined with other wrapper options. May download Ruff.
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
  scripts/dev/run_worktree_shared_venv.sh --recover-stale-fast-pysf -- \
    uv run python scripts/dev/issue_audit_core.py plan --mode autonomous
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
recover_stale_fast_pysf=""
standalone=""
isolated_ruff=""
isolated_conflict=""
command_separator=""
scratch_dir="${ROBOT_SF_CI_SCRATCH_DIR:-}"
cmd=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)
      isolated_conflict=1
      if [[ $# -lt 2 ]]; then
        echo "--venv requires a path." >&2
        exit 2
      fi
      venv_override="$2"
      shift 2
      ;;
    --profile)
      isolated_conflict=1
      if [[ $# -lt 2 || -z "${2:-}" ]]; then
        echo "--profile requires a dependency profile name." >&2
        exit 2
      fi
      dependency_profile="$2"
      shift 2
      ;;
    --scratch-dir)
      isolated_conflict=1
      if [[ $# -lt 2 || -z "${2:-}" ]]; then
        echo "--scratch-dir requires a path." >&2
        exit 2
      fi
      scratch_dir="$2"
      shift 2
      ;;
    --recover-stale-fast-pysf)
      isolated_conflict=1
      recover_stale_fast_pysf=1
      shift
      ;;
    --standalone)
      isolated_conflict=1
      standalone=1
      shift
      ;;
    --no-freshness-check)
      isolated_conflict=1
      skip_freshness=1
      shift
      ;;
    --isolated-ruff)
      isolated_ruff=1
      shift
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    --)
      command_separator=1
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
if [[ -n "$isolated_ruff" ]]; then
  if [[ -n "$isolated_conflict" || -n "$scratch_dir" || -z "$command_separator" \
    || "${ROBOT_SF_VENV_FRESHNESS_CHECK:-}" == "skip" ]]; then
    echo "ERROR: --isolated-ruff requires '--' and cannot be combined with other wrapper options or a freshness bypass." >&2
    exit 2
  fi
  # Host-only stdlib parsing/execution: never select a project interpreter or load site hooks.
  host_python=""
  for candidate in /usr/bin/python3 /usr/local/bin/python3; do
    if [[ -x "$candidate" ]] && "$candidate" -I -S -B -c 'import tomllib' 2>/dev/null; then
      host_python="$candidate"
      break
    fi
  done
  if [[ -z "$host_python" ]]; then
    echo "ERROR: --isolated-ruff requires host Python 3.11+ with stdlib tomllib." >&2
    exit 2
  fi
  # Preserve caller cwd; the default shared-env branch below retains its root normalization.
  exec "$host_python" -I -S -B - "$repo_root" "${cmd[@]}" <<'PY'
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import tomllib


def fail(message):
    raise ValueError(message)


def admit(root, argv):
    if len(argv) < 2 or argv[0] != "ruff" or argv[1] not in {"check", "format"}:
        fail("expected bare 'ruff check' or 'ruff format --check'")
    mode = argv[1]
    flags = {"--no-cache", "--quiet", "-q", "--verbose", "-v", "--respect-gitignore",
             "--no-respect-gitignore", "--force-exclude", "--no-force-exclude"}
    values = {"--exclude", "--extend-exclude", "--line-length", "--target-version", "--extension"}
    if mode == "check":
        flags |= {"--no-fix", "--no-fix-only", "--no-unsafe-fixes", "--preview", "--no-preview"}
        values |= {"--select", "--extend-select", "--ignore", "--extend-ignore",
                   "--per-file-ignores", "--extend-per-file-ignores", "--output-format"}
    else:
        flags |= {"--check", "--preview", "--no-preview"}
    options = argv[2:]
    index = 0
    format_check = False
    while index < len(options):
        arg = options[index]
        if arg == "--":
            if "-" in options[index + 1:]:
                fail("stdin is not supported; supply file paths")
            break
        if arg == "-":
            fail("stdin is not supported; supply file paths")
        if arg.startswith("-"):
            name, equals, value = arg.partition("=")
            if name in flags and not equals:
                format_check |= name == "--check"
            elif name in values:
                if not equals:
                    index += 1
                    if index >= len(options) or options[index].startswith("-"):
                        fail(f"missing value for {name}")
                    value = options[index]
                if not value:
                    fail(f"missing value for {name}")
            else:
                fail(f"unsupported validation option: {name}")
        index += 1
    if mode == "format" and not format_check:
        fail("format requires --check before the filename separator")
    manifest = root / "pyproject.toml"
    data = tomllib.loads(manifest.read_text(encoding="utf-8"))
    dev = data.get("dependency-groups", {}).get("dev", [])
    if not isinstance(dev, list):
        fail("expected one exact Ruff dependency in dependency-groups.dev")
    # Include malformed/extras/URL/marker forms in the candidate count, never first-match grep.
    declarations = [item for item in dev if isinstance(item, str)
                    and re.match(r"(?i)^\s*ruff(?=[^a-z0-9_.-]|$)", item)]
    if len(declarations) != 1 or not re.fullmatch(r"ruff==[0-9]+\.[0-9]+\.[0-9]+", declarations[0]):
        fail("expected one exact unqualified ruff==X.Y.Z dev dependency")
    if any(not isinstance(item, str) for item in dev):
        fail("included/ambiguous dev dependency groups are unsupported")
    pin = declarations[0].split("==")[1]
    config = data.get("tool", {}).get("ruff", {})
    if config.get("required-version") != "==" + pin:
        fail("Ruff dev pin and tool.ruff.required-version must be exactly equal")
    if any(key in config for key in ("extend", "cache-dir", "output-file")):
        fail("root Ruff configuration cannot extend or redirect cache/output")
    if config.get("fix", False) is not False or config.get("fix-only", False) is not False:
        fail("root Ruff configuration cannot enable fix or fix-only")
    return pin, manifest, mode


def run():
    root = Path(sys.argv[1]).resolve(strict=True)
    argv = sys.argv[2:]
    pin, manifest, mode = admit(root, argv)
    common = Path(subprocess.check_output(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], text=True).strip())
    owner = common.parent.resolve(strict=True)
    protected = [root, owner, (root / ".venv").resolve(), (owner / ".venv").resolve()]
    for key in ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"):
        if os.environ.get(key):
            protected.append(Path(os.environ[key]).resolve())

    def check_location(path):
        resolved = path.resolve()
        if any(resolved == item or item in resolved.parents for item in protected):
            fail("isolated cache/temp overlaps a project or environment (including symlinks)")
        return resolved

    # These variables are never used, but reject dangerous aliases explicitly before any writes.
    for key in ("UV_CACHE_DIR", "UV_TOOL_DIR", "TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME",
                "RUFF_CACHE_DIR"):
        if os.environ.get(key):
            check_location(Path(os.environ[key]))
    temp_base = check_location(Path("/tmp"))
    uv = shutil.which("uv")
    if not uv:
        fail("uv executable is unavailable; no unpinned fallback")
    uv = str(Path(uv).resolve(strict=True))
    child_env = {key: value for key, value in os.environ.items()
                 if not key.startswith(("UV_", "PYTHON", "RUFF_", "VIRTUAL_ENV", "XDG_"))
                 and key not in {"TMPDIR", "TMP", "TEMP", "__PYVENV_LAUNCHER__"}}
    # The sole supported inherited uv control is offline operation. No fallback on a cache miss.
    offline = os.environ.get("UV_OFFLINE")
    if offline is not None:
        if offline not in {"0", "1", "true", "false"}:
            fail("UV_OFFLINE must be 0, 1, true, or false")
        child_env["UV_OFFLINE"] = offline
    child = None
    interrupted = 0
    task_dir = None
    cleanup_complete = False

    def on_signal(signum, _frame):
        nonlocal interrupted
        interrupted = signum
        if cleanup_complete:
            # Keep the exit boundary signal-aware even after run() computes its return value.
            raise SystemExit(128 + signum)
        if child is not None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, on_signal)

    def invoke(command, capture=False):
        nonlocal child
        if interrupted:
            return None
        child = subprocess.Popen(command, env=child_env, start_new_session=True,
                                 stdout=subprocess.PIPE if capture else None,
                                 stderr=subprocess.PIPE if capture else None, text=True)
        if interrupted:
            on_signal(interrupted, None)
        deadline = None
        while True:
            try:
                stdout, stderr = child.communicate(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                if interrupted:
                    if deadline is None:
                        deadline = time.monotonic() + 2
                    if time.monotonic() >= deadline:
                        try:
                            os.killpg(child.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
        result = (child.returncode, stdout, stderr)
        # Stop any surviving descendants before deleting this invocation's temporary state.
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child = None
        return result

    try:
        task_dir = Path(tempfile.mkdtemp(prefix="robot-sf-isolated-ruff-", dir=temp_base))
        check_location(task_dir)
        for name in ("uv-cache", "tmp", "xdg-cache", "tools"):
            (task_dir / name).mkdir()
        child_env.update(UV_CACHE_DIR=str(task_dir / "uv-cache"), TMPDIR=str(task_dir / "tmp"),
                         UV_TOOL_DIR=str(task_dir / "tools"),
                         XDG_CACHE_HOME=str(task_dir / "xdg-cache"))
        base = [uv, "tool", "run", "--isolated", "--no-config", "--no-env-file",
                "--no-python-downloads", "--python", sys.executable,
                "--from", "ruff==" + pin, "ruff"]
        version = invoke([*base, "--version"], capture=True)
        if interrupted:
            return 128 + interrupted
        if version[0] != 0 or version[1].strip() != "ruff " + pin:
            fail("isolated Ruff provisioning/version verification failed; "
                 + (version[2] or version[1] or "no version output")[:2000])
        print(f"Isolated Ruff verified: ruff=={pin}; temporary cache={task_dir}", file=sys.stderr)
        guards = ["--config", str(manifest), "--no-cache"]
        if mode == "check":
            guards += ["--no-fix", "--no-fix-only", "--no-unsafe-fixes"]
        result = invoke([*base, mode, *guards, *argv[2:]])
        if interrupted:
            return 128 + interrupted
        return result[0] if result[0] >= 0 else 128 - result[0]
    finally:
        if child is not None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
        if task_dir is not None:
            shutil.rmtree(task_dir)
        cleanup_complete = True
        if interrupted:
            # A signal during successful cleanup must override an already evaluated Ruff status.
            raise SystemExit(128 + interrupted)


try:
    status = run()
except (OSError, ValueError, TypeError, KeyError, AttributeError, subprocess.SubprocessError) as error:
    print(f"ERROR: isolated Ruff validation refused: {error}", file=sys.stderr)
    status = 2
sys.exit(status)
PY
fi
cd "$repo_root"
# Do not let an ambient UV_PROJECT redirect dependency resolution to another
# checkout. The effective project is always the current worktree.
unset UV_PROJECT

git_common_dir="$(git rev-parse --git-common-dir)"
if [[ "$git_common_dir" != /* ]]; then
  git_common_dir="$(cd "$repo_root/$git_common_dir" && pwd)"
fi
main_repo_root="$(cd "$git_common_dir/.." && pwd)"
is_linked_worktree=0
if [[ "$git_common_dir" != "$repo_root/.git" ]]; then
  is_linked_worktree=1
fi

if [[ -n "$scratch_dir" ]]; then
  configure_scratch_dir "$scratch_dir"
fi
check_scratch_capacity "${TMPDIR:-/tmp}"

if [[ -n "$recover_stale_fast_pysf" ]]; then
  if [[ -n "$venv_override" ]]; then
    echo "ERROR: --recover-stale-fast-pysf cannot be combined with --venv." >&2
    echo "Recovery is limited to the current linked worktree's .venv." >&2
    exit 2
  fi
  if [[ -n "$standalone" ]]; then
    echo "ERROR: --recover-stale-fast-pysf cannot be combined with --standalone." >&2
    echo "Recovery must verify the project fast-pysf package before the command starts." >&2
    exit 2
  fi
  if [[ -n "$skip_freshness" || "${ROBOT_SF_VENV_FRESHNESS_CHECK:-}" == "skip" ]]; then
    echo "ERROR: --recover-stale-fast-pysf cannot be combined with a freshness bypass." >&2
    echo "Recovery always verifies the refreshed package before running the command." >&2
    exit 2
  fi

  recovery_script="$repo_root/scripts/dev/recover_fast_pysf_worktree.sh"
  if [[ ! -x "$recovery_script" ]]; then
    echo "ERROR: worktree fast-pysf recovery helper is missing or not executable: $recovery_script" >&2
    exit 2
  fi
  if "$recovery_script"; then
    :
  else
    recovery_rc=$?
    exit "$recovery_rc"
  fi
  venv_path="$repo_root/.venv"
else
  if [[ -n "$venv_override" ]]; then
    venv_path="$venv_override"
  elif [[ -x "$repo_root/.venv/bin/python" ]]; then
    venv_path="$repo_root/.venv"
  else
    venv_path="$main_repo_root/.venv"
  fi
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
    local stale_diagnostic="installed pysocialforce package is stale relative to this checkout"
    if [[ "$report" != *"$stale_diagnostic"* ]]; then
      echo "The checker failure is not the supported stale-package condition; refusing automatic recovery." >&2
      echo "Remedy: inspect the checker failure and repair the selected environment explicitly." >&2
      return 2
    fi
    echo "Remedy (linked worktree): the wrapper will retry with a worktree-local environment automatically." >&2
    echo "To force that recovery before the freshness check, use --recover-stale-fast-pysf." >&2
    echo "Remedy: run 'uv sync --all-extras --reinstall-package robot-sf' in the owning checkout, then retry." >&2
    return 3
  fi
  echo "Shared-venv project package freshness preflight passed: package=pysocialforce venv=$venv_path" >&2
}

recover_stale_fast_pysf_automatically() {
  if [[ "$is_linked_worktree" -ne 1 || -n "$venv_override" ]]; then
    return 1
  fi

  recovery_script="$repo_root/scripts/dev/recover_fast_pysf_worktree.sh"
  if [[ ! -x "$recovery_script" ]]; then
    echo "ERROR: automatic worktree fast-pysf recovery helper is missing or not executable: $recovery_script" >&2
    return 2
  fi
  echo "Recovering stale fast-pysf in the linked worktree: $repo_root/.venv" >&2
  "$recovery_script" || return $?
  venv_path="$repo_root/.venv"
  if [[ ! -x "$venv_path/bin/python" ]]; then
    echo "ERROR: automatic fast-pysf recovery did not create a usable worktree environment: $venv_path" >&2
    return 2
  fi
  echo "Automatic fast-pysf recovery selected worktree environment: $venv_path" >&2
}

read_default_dev_tool_pin() {
  # Match the isolated runner's trusted host boundary, without its stricter Ruff-only policy.
  # Keep this lazy so interpreter, explicit-path and no-manifest skips need no TOML runtime.
  local host_python="" candidate
  for candidate in /usr/bin/python3 /usr/local/bin/python3; do
    if [[ -x "$candidate" ]] && "$candidate" -I -S -B -c 'import tomllib' 2>/dev/null; then
      host_python="$candidate"
      break
    fi
  done
  if [[ -z "$host_python" ]]; then
    echo "ERROR: Shared-venv pin parser requires host Python 3.11+ with stdlib tomllib." >&2
    return 2
  fi
  "$host_python" -I -S -B - "$1" "$2" <<'PY'
import re
import sys
import tomllib

try:
    with open(sys.argv[1], "rb") as stream:
        manifest = tomllib.load(stream)
    groups = manifest.get("dependency-groups", {})
    if not isinstance(groups, dict):
        raise ValueError("dependency-groups must be a table")
    dev = groups.get("dev", [])
    if not isinstance(dev, list) or any(not isinstance(item, str) for item in dev):
        raise ValueError("dev must be an array of strings; include groups are unsupported")
    tool = sys.argv[2]
    declarations = []
    pins = set()
    for item in dev:
        name = re.match(r"[A-Za-z0-9_.-]+", item.strip())
        if name is None or name.group() != tool:
            continue
        declarations.append(item)
        exact = re.fullmatch(re.escape(tool) + r"==([0-9A-Za-z._+-]+)", item.strip())
        if exact:
            pins.add(exact.group(1))
        else:
            # Marker/URL equality is not a tool pin. Preserve plain wildcard-only ranges,
            # but never disguise an unsupported exact-looking declaration as unpinned.
            requirement = item.partition(";")[0].partition("@")[0].strip()
            wildcard = re.fullmatch(
                re.escape(tool) + r"\s*==\s*[0-9]+(?:\.[0-9]+)*\.\*", requirement
            )
            if "==" in requirement and wildcard is None:
                raise ValueError(f"unsupported exact development declaration for {tool}")
    if len(pins) > 1:
        raise ValueError(f"conflicting exact development pins for {tool}")
    if pins and any(item.strip() != f"{tool}=={next(iter(pins))}" for item in declarations):
        raise ValueError(f"ambiguous development declarations for {tool}")
    if pins:
        print(next(iter(pins)))
except (OSError, ValueError, TypeError) as exc:
    print(f"ERROR: Shared-venv pin parser failed: {exc}", file=sys.stderr)
    sys.exit(2)
PY
}

check_shared_venv_freshness() {
  # Do not declare venv_path as local: automatic recovery in a linked worktree
  # updates venv_path to the worktree-local environment, which must persist to
  # UV_PROJECT_ENVIRONMENT for the final uv-run execution boundary (issue #8772).
  if [[ "$#" -gt 0 ]]; then
    venv_path="$1"
  fi
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
    if ! pin="$(read_default_dev_tool_pin "$repo_root/pyproject.toml" "$tool")"; then
      echo "ERROR: Shared-venv tool freshness preflight failed: tool=$tool reason=pin-parser-error elapsed_ms=$(freshness_elapsed_ms) venv=$venv_path" >&2
      return 2
    fi
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
      if check_project_package_freshness "$venv_path"; then
        :
      else
        freshness_rc=$?
        if [[ "$freshness_rc" -eq 3 ]] && recover_stale_fast_pysf_automatically; then
          if ! check_dependency_profile; then
            return 2
          fi
          if ! check_project_package_freshness "$venv_path"; then
            return 2
          fi
        else
          return 2
        fi
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
  if [[ "$tool" == "ruff" ]]; then
    local owner_manifest="" owner_pin="unknown"
    if [[ "$venv_path" == "$repo_root/.venv" ]]; then
      owner_manifest="$repo_root/pyproject.toml"
    elif [[ "$venv_path" == "$main_repo_root/.venv" ]]; then
      owner_manifest="$main_repo_root/pyproject.toml"
    fi
    if [[ -n "$owner_manifest" && -x /usr/bin/python3 ]]; then
      owner_pin="$(/usr/bin/python3 -I -S -B - "$owner_manifest" <<'PY'
import re
import sys
import tomllib
try:
    with open(sys.argv[1], "rb") as stream:
        dev = tomllib.load(stream).get("dependency-groups", {}).get("dev", [])
    pins = [item for item in dev if isinstance(item, str) and item.startswith("ruff")]
    print(pins[0][6:] if len(pins) == 1 and re.fullmatch(r"ruff==\d+\.\d+\.\d+", pins[0]) else "unknown")
except (OSError, ValueError, AttributeError, TypeError):
    print("unknown")
PY
      )" || owner_pin="unknown"
    fi
    echo "Selected executable: $venv_path/bin/ruff; owning manifest pin: ruff==$owner_pin." >&2
    if [[ "$owner_pin" == "unknown" ]]; then
      echo "Owning manifest identity/pin is unknown; owner re-sync has not been verified as a remedy." >&2
    elif [[ "$owner_pin" != "$pin" ]]; then
      echo "Owning manifest differs from the active checkout; owner re-sync alone reinstalls ruff==$owner_pin, not ruff==$pin." >&2
    else
      echo "Owning manifest matches the active pin; the selected installation is stale. Owner-managed re-sync or a matching explicit --venv can repair it." >&2
    fi
    echo "Remedy: for focused validation use --isolated-ruff -- ruff check <paths> or --isolated-ruff -- ruff format --check <paths>." >&2
    return 2
  fi
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
# When the worktree-local .venv is selected (explicitly or via automatic
# recovery), keep VIRTUAL_ENV aligned so uv run and subprocesses execute
# with the verified worktree interpreter (issue #8772).
if [[ -n "$venv_override" ]]; then
  export VIRTUAL_ENV="$venv_path"
  export ROBOT_SF_EXPLICIT_VENV_OVERRIDE="$venv_path"
elif [[ "$venv_path" == "$repo_root/.venv" ]]; then
  export VIRTUAL_ENV="$venv_path"
fi
if [[ -z "$standalone" ]]; then
  export PYTHONPATH="$repo_root:$repo_root/fast-pysf${PYTHONPATH:+:$PYTHONPATH}"
fi

if [[ -z "${COVERAGE_FILE:-}" && "$git_common_dir" != "$repo_root/.git" ]]; then
  worktree_id="$(printf '%s' "$repo_root" | git hash-object --stdin | cut -c1-12)"
  export COVERAGE_FILE="$repo_root/output/coverage/.coverage.${worktree_id}"
fi

exec uv run "${cmd[@]}"
