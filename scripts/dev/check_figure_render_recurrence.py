#!/usr/bin/env python3
"""Isolated recurrence guard for recurrence-eligible figure-render commands (issue #6770).

Second child of #6616. Loads and structurally validates the v1 figure-render registry produced
by ``scripts/dev/build_figure_render_registry.py`` (issue #6769), then executes ONLY the entries
marked ``recurrence_eligible: true`` in isolated temporary output roots and reports whether each
reproduced its declared contract.

Per-entry recurrence contract (all must hold for ``passed``):

* every committed input exists with its declared SHA-256 (no input drift);
* the command is safe to re-run (no shell metacharacters, network, scheduler submission, absolute
  output path, or shell environment-assignment prefix);
* every declared expected output appears as a literal command token so it can be redirected into an
  isolated temporary output root (the committed evidence path is never written);
* the command is invoked as an argument vector without ``shell=True``, from its declared working
  directory, with a minimal inherited environment (no network access, no scheduler credentials);
* the command terminates within its declared timeout;
* the command exits with its expected exit code (``0`` by default; an explicit, auditable negative
  control may declare an expected non-zero exit code);
* every declared expected output exists under the temporary output root;
* no undeclared output appears inside the temporary output root; and
* no write lands outside the temporary output root and the benign runtime cache/venv/bytecode set.

Excluded entries are reported with their controlled exclusion reason codes and are NEVER executed.
The guard fails closed (non-zero exit) on a missing/invalid registry, a missing ``strace`` binary
(required for deterministic write containment), any structural registry violation, an empty eligible
set unless an explicit zero-entry policy is recorded, or any per-command contract violation.

The compact machine-readable recurrence report records the registry/source commit, command ids,
statuses, exclusion reason codes, and per-command check outcomes. It contains NO figure content.
Generated figure output is disposable local scratch and is never committed: outputs are written into
an ephemeral temporary root that is removed at the end of the run.

Claim boundary: passing the guard proves only local workflow recurrence and write containment for the
pinned commands at the pinned commit. It does NOT validate figure semantics, scientific
interpretation, benchmark correctness, evidence admission, or publication suitability.
"""

# evidence-writer-exempt: the recurrence report JSON at
# docs/context/evidence/issue_6770_figure_render_recurrence_report.json is written through the shared
# ``robot_sf.evidence.writers.write_json`` helper (which applies the review marker and registers the
# bundle in docs/context/catalog.yaml). Non-evidence --report targets (e.g. /tmp validation outputs)
# are written without catalog registration.

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from robot_sf.evidence.writers import write_json as write_evidence_json

REPO_ROOT = Path(__file__).resolve().parents[2]
ISSUE_NUMBER = 6770
REPORT_SCHEMA = "issue_6770_figure_render_recurrence_report.v1"
REGISTRY_DEFAULT = REPO_ROOT / "docs" / "context" / "figure_render_registry.v1.yaml"
REPORT_DEFAULT = (
    REPO_ROOT / "docs" / "context" / "evidence" / "issue_6770_figure_render_recurrence_report.json"
)
REGISTRY_VERSION = 1

ALLOWED_EXCLUSION_REASONS = frozenset(
    {
        "external_input",
        "requires_slurm",
        "requires_network",
        "missing_committed_fixture",
        "non_deterministic_contract",
        "unsafe_command",
        "historical_only",
        "superseded",
    }
)

# Minimal environment inherited by every executed command. Only operational variables are allowlisted;
# every credential-bearing variable (SLURM_*, cloud tokens, secrets) is deliberately excluded. The
# forced UV_* flags keep uv from reaching the network (the CI job provides the hard network block).
ENV_ALLOWLIST = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "TZ",
    "SHELL",
    "TERM",
)
ENV_UV_ALLOWLIST = (
    "UV_CACHE_DIR",
    "UV_PYTHON_INSTALL_DIR",
    "UV_PROJECT_ENVIRONMENT",
    "UV_PYTHON",
    "VIRTUAL_ENV",
)
ENV_FORCE = {
    "UV_OFFLINE": "1",
    "UV_NO_SYNC": "1",
    "PYTHONUNBUFFERED": "1",
}

# strace write-tracking. ``-s 4096`` is mandatory: strace truncates string arguments to 32 bytes by
# default, which would silently corrupt the long temporary output paths we are checking against.
STRACE_BIN = "strace"
STRACE_SYSCALLS = (
    "openat,open,creat,unlink,unlinkat,rename,renameat,renameat2,link,linkat,"
    "symlink,symlinkat,mkdir,mkdirat,rmdir,truncate"
)
OPEN_SYSCALLS = frozenset({"open", "openat", "creat"})
MULTI_PATH_SYSCALLS = frozenset(
    {"rename", "renameat", "renameat2", "link", "linkat", "symlink", "symlinkat"}
)
SINGLE_PATH_SYSCALLS = frozenset({"unlink", "unlinkat", "mkdir", "mkdirat", "rmdir", "truncate"})
ALL_WRITE_SYSCALLS = OPEN_SYSCALLS | MULTI_PATH_SYSCALLS | SINGLE_PATH_SYSCALLS
WRITE_FLAGS = ("O_WRONLY", "O_RDWR", "O_CREAT", "O_TRUNC", "O_APPEND")

# Command-string safety re-validation (mirrors build_figure_render_registry._detect_trigger so the
# guard never trusts the registry's eligibility flag alone).
SHELL_METACHAR_RE = re.compile(r"(&&|\|\||;|\||>|<|`|\$\()")
NETWORK_RE = re.compile(r"(https?://|(^|\s)(curl|wget|scp|rsync|ssh)(\s|$))", re.IGNORECASE)
SLURM_RE = re.compile(r"(^|\s)(sbatch|srun|salloc|squeue)(\s|$)", re.IGNORECASE)
ENV_ASSIGN_RE = re.compile(r"^[A-Z_][A-Z0-9_]*=")
ABSOLUTE_ARG_RE = re.compile(r"^/(?!/)")
FAIL_ON_FLAG_RE = re.compile(r"^--fail-on")

# Explicit, auditable negative controls. A negative control is a command whose successful recurrence
# is contracted to PRODUCE its declared output AND exit with a specific non-zero code (it is a
# violation/guard detector whose non-zero exit IS the detection signal). Keyed on the registry's
# stable entry id; if the command string changes the id changes and the guard fails closed on the
# resulting unexpected non-zero exit, prompting an explicit allowlist review.
NEGATIVE_CONTROLS: dict[str, dict[str, Any]] = {
    "docs_context_evidence_issue_3482_event_ledger_reconciliation_guard_README_md__cmd2_fccbfb79": {
        "reason": (
            "violation-detection negative control: export_event_ledger_reconciliation.py invoked with "
            "--fail-on-violations on a deliberately-violating fixture is contracted to write its "
            "reconciliation report and exit 1; reproducing means the declared output is present, "
            "containment holds, and the exit code is 1"
        ),
        "expected_exit_code": 1,
    },
}

_STRACE_LINE_RE = re.compile(
    r"^(?P<pid>\d+)\s+(?P<sc>[A-Za-z0-9_]+)\((?P<args>.*)\)\s+=\s+(?P<rest>.*)$"
)
_OPENAT_ARG_RE = re.compile(
    r'^(?:AT_FDCWD|-?\d+),\s*"(?P<path>(?:[^"\\]|\\.)*)"\s*,\s*(?P<flags>[^,)]+)'
)
_OPEN_ARG_RE = re.compile(r'^"(?P<path>(?:[^"\\]|\\.)*)"\s*,\s*(?P<flags>[^,)]+)')
_QUOTED_RE = re.compile(r'"(?P<path>(?:[^"\\]|\\.)*)"')


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def now_iso() -> str:
    """Return the current UTC time as a second-precision ISO-8601 string."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def git_head() -> str | None:
    """Return the current repository HEAD commit SHA, or ``None`` if unavailable."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, OSError):
        return None


def sha256_of(path: Path) -> str | None:
    """Return the SHA-256 hex digest of a file, or ``None`` if it is not a readable file."""
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _norm_abs(path: Path, cwd: Path) -> Path:
    """Resolve ``path`` to an absolute, lexically-normalized Path for prefix matching.

    Relative paths are resolved against ``cwd``. ``Path.resolve()`` performs lexical normalization
    for non-existent paths (Python 3.6+) without requiring the target to exist, which is essential
    because traced writes may create or delete paths that are gone by parse time.
    """
    p = path if path.is_absolute() else (cwd / path)
    return p.resolve()


def _is_under(path: Path, root: Path) -> bool:
    """Return True when ``path`` is ``root`` or a descendant of ``root``."""
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _is_bytecode(path: Path) -> bool:
    """Return True for benign Python bytecode artifacts (``__pycache__`` / ``*.pyc``)."""
    return "__pycache__" in path.parts or path.name.endswith(".pyc")


# ---------------------------------------------------------------------------
# Registry loading and structural validation
# ---------------------------------------------------------------------------


def load_registry(path: Path) -> dict[str, Any]:
    """Load the registry YAML, raising ``ValueError`` on a parse failure."""
    if not path.is_file():
        raise ValueError(f"registry file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"registry is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("registry top level must be a mapping")
    return data


_REQUIRED_ENTRY_FIELDS = (
    "id",
    "command",
    "working_dir",
    "inputs",
    "expected_outputs",
    "timeout_seconds",
    "recurrence_eligible",
    "exclusion_reason",
)


def _validate_entry(entry: dict[str, Any], idx: int, seen_ids: set[str]) -> list[str]:
    """Return structural errors for a single registry entry."""
    errors: list[str] = []
    for key in _REQUIRED_ENTRY_FIELDS:
        if key not in entry:
            errors.append(f"entry[{idx}] missing required field '{key}'")
    errors.extend(_entry_identity_errors(entry, idx, seen_ids))
    errors.extend(_entry_shape_errors(entry, idx))
    errors.extend(_entry_eligibility_errors(entry, idx))
    return errors


def _entry_identity_errors(entry: dict[str, Any], idx: int, seen_ids: set[str]) -> list[str]:
    """Validate the entry id is a non-empty unique string."""
    errors: list[str] = []
    entry_id = entry.get("id")
    if not isinstance(entry_id, str) or not entry_id:
        errors.append(f"entry[{idx}] has a non-string id")
    elif entry_id in seen_ids:
        errors.append(f"duplicate entry id '{entry_id}'")
    else:
        seen_ids.add(entry_id)
    return errors


def _entry_shape_errors(entry: dict[str, Any], idx: int) -> list[str]:
    """Validate the structural shape (types) of a single registry entry."""
    errors: list[str] = []
    command = entry.get("command")
    if not isinstance(command, list) or not command or not all(isinstance(t, str) for t in command):
        errors.append(f"entry[{idx}] command must be a non-empty list of strings")
    if not isinstance(entry.get("working_dir"), str):
        errors.append(f"entry[{idx}] working_dir must be a string")
    inputs = entry.get("inputs")
    if not isinstance(inputs, list):
        errors.append(f"entry[{idx}] inputs must be a list")
    else:
        for inp in inputs:
            if not isinstance(inp, dict) or "path" not in inp or "sha256" not in inp:
                errors.append(f"entry[{idx}] has a malformed input entry")
    if not isinstance(entry.get("expected_outputs"), list):
        errors.append(f"entry[{idx}] expected_outputs must be a list")
    if not isinstance(entry.get("timeout_seconds"), int) or entry["timeout_seconds"] <= 0:
        errors.append(f"entry[{idx}] timeout_seconds must be a positive integer")
    return errors


def _entry_eligibility_errors(entry: dict[str, Any], idx: int) -> list[str]:
    """Validate the recurrence_eligible / exclusion_reason contract."""
    errors: list[str] = []
    eligible = entry.get("recurrence_eligible")
    if not isinstance(eligible, bool):
        errors.append(f"entry[{idx}] recurrence_eligible must be a boolean")
        return errors
    reason = entry.get("exclusion_reason")
    if eligible:
        if reason is not None:
            errors.append(f"entry[{idx}] is eligible but carries an exclusion_reason")
        if not entry.get("expected_outputs"):
            errors.append(f"entry[{idx}] is eligible but declares no expected outputs")
    elif reason not in ALLOWED_EXCLUSION_REASONS:
        errors.append(f"entry[{idx}] has non-canonical exclusion_reason {reason!r}")
    return errors


def validate_registry_structure(registry: dict[str, Any]) -> list[str]:
    """Return a list of structural validation errors (empty when the registry is well-formed)."""
    errors: list[str] = []
    if registry.get("version") != REGISTRY_VERSION:
        errors.append(f"registry version must be {REGISTRY_VERSION}")
    entries = registry.get("entries")
    if not isinstance(entries, list) or not entries:
        errors.append("registry must contain a non-empty 'entries' list")
        return errors
    seen_ids: set[str] = set()
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            errors.append(f"entry[{idx}] is not a mapping")
            continue
        errors.extend(_validate_entry(entry, idx, seen_ids))
    return errors


def _resolve_working_dir(working_dir: str) -> Path:
    """Resolve a registry ``working_dir`` to an absolute path inside the repository."""
    wd = Path(working_dir)
    return (REPO_ROOT / wd).resolve() if not wd.is_absolute() else wd.resolve()


# ---------------------------------------------------------------------------
# Per-command pre-checks
# ---------------------------------------------------------------------------


def re_safety_check(command: list[str]) -> str | None:
    """Re-validate command safety; return a controlled reason string, or ``None`` when safe."""
    if command and ENV_ASSIGN_RE.match(command[0]):
        return "unsafe_command"
    joined = " ".join(command)
    if SHELL_METACHAR_RE.search(joined):
        return "unsafe_command"
    if SLURM_RE.search(joined):
        return "requires_slurm"
    if NETWORK_RE.search(joined):
        return "requires_network"
    for token in command:
        if ABSOLUTE_ARG_RE.match(token):
            return "unsafe_command"
    return None


def verify_input_hashes(entry: dict[str, Any]) -> str | None:
    """Verify every committed input exists with its declared SHA-256; return a drift reason or None."""
    for inp in entry.get("inputs", []):
        rel = Path(inp["path"])
        full = rel if rel.is_absolute() else REPO_ROOT / rel
        if not full.is_file():
            return f"input_drift:{inp['path']}:missing"
        digest = sha256_of(full)
        if digest is None or digest != inp["sha256"]:
            return f"input_drift:{inp['path']}:hash_mismatch"
    return None


def rewrite_outputs(
    command: list[str], expected_outputs: list[str], temp_root: Path
) -> tuple[list[str], str | None]:
    """Rewrite declared-output tokens into the temporary output root.

    Returns ``(rewritten_command, error)`` where ``error`` is set when a declared output does not
    appear as a literal command token (so the command would write the committed evidence path and
    cannot be safely redirected).
    """
    rewritten = list(command)
    normalized_outputs = {Path(o).as_posix(): o for o in expected_outputs}
    # Track which declared outputs we actually redirected so we can fail closed on unredirectable ones.
    redirected: set[str] = set()
    for idx, token in enumerate(rewritten):
        token_path = Path(token).as_posix()
        for norm, original in normalized_outputs.items():
            target = (temp_root / Path(original)).as_posix()
            if token_path == norm:
                rewritten[idx] = target
                redirected.add(original)
            elif norm and token_path.startswith(norm + "/"):
                # Token is a descendant of a declared output (a file inside a declared output dir).
                suffix = token_path[len(norm) + 1 :]
                rewritten[idx] = f"{target}/{suffix}"
                redirected.add(original)
    unredirectable = [o for o in expected_outputs if o not in redirected]
    if unredirectable:
        return rewritten, f"output_not_redirectable:{unredirectable[0]}"
    return rewritten, None


# ---------------------------------------------------------------------------
# Minimal environment and benign write roots
# ---------------------------------------------------------------------------


def build_minimal_env() -> dict[str, str]:
    """Build a minimal credential-free environment for executed commands."""
    env: dict[str, str] = {}
    for key in ENV_ALLOWLIST:
        if key in os.environ:
            env[key] = os.environ[key]
    for key in ENV_UV_ALLOWLIST:
        if key in os.environ:
            env[key] = os.environ[key]
    env.setdefault("VIRTUAL_ENV", str((REPO_ROOT / ".venv").resolve()))
    env.update(ENV_FORCE)
    return env


# Benign write patterns outside the temp root and the benign roots. ``uv-<hash>.lock`` is uv's
# project lock under TMPDIR (issue #6770 containment note); it carries no figure content.
BENIGN_WRITE_PATTERNS = (re.compile(r"^.*/uv-[0-9a-f]+\.lock$"),)


def _benign_write_roots(temp_root: Path, trace_file: Path, repo_root: Path) -> list[Path]:
    """Return absolute roots outside the temporary output root that are benign runtime writes."""
    roots: list[Path] = [trace_file.resolve()]
    home = Path(os.environ.get("HOME", "/")).resolve()
    candidates = [
        REPO_ROOT / ".venv",
        home / ".cache",
        home / ".config",
        home / ".local",
    ]
    for var in (
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "UV_CACHE_DIR",
        "UV_PYTHON_INSTALL_DIR",
    ):
        value = os.environ.get(var)
        if value:
            candidates.append(Path(value))
    for candidate in candidates:
        try:
            roots.append(candidate.resolve())
        except OSError:
            roots.append(candidate)
    for dev in ("null", "tty", "urandom", "random", "zero", "stdout", "stderr", "stdin"):
        roots.append(Path("/dev") / dev)
    for virt in ("/proc", "/sys"):
        roots.append(Path(virt))
    return roots


# ---------------------------------------------------------------------------
# strace write tracking
# ---------------------------------------------------------------------------


def _extract_write_paths(args: str, syscall: str) -> list[str]:
    """Extract the path argument(s) touched by a write syscall's raw argument string."""
    if syscall == "openat":
        match = _OPENAT_ARG_RE.match(args)
        return [match.group("path")] if match else []
    if syscall == "open":
        match = _OPEN_ARG_RE.match(args)
        return [match.group("path")] if match else []
    if syscall == "creat":
        match = _QUOTED_RE.search(args)
        return [match.group("path")] if match else []
    # rename/link/symlink/mkdir/unlink/truncate: collect every quoted path argument.
    return [m.group("path") for m in _QUOTED_RE.finditer(args)]


def _write_flags_present(args: str, syscall: str) -> bool:
    """Return True when an open/openat call carries a write/create/truncate flag."""
    if syscall == "creat":
        return True
    flags_match = _OPENAT_ARG_RE.match(args) if syscall == "openat" else _OPEN_ARG_RE.match(args)
    if not flags_match:
        return True  # conservative: treat an unparseable open as a potential write
    flags = flags_match.group("flags")
    return any(flag in flags for flag in WRITE_FLAGS)


def _parse_retval(rest: str) -> int | None:
    """Parse the leading numeric return value from a strace line's trailing segment."""
    token = rest.strip().split(None, 1)[0] if rest.strip() else ""
    try:
        return int(token)
    except ValueError:
        return None


def collect_write_targets(trace_file: Path, cwd: Path) -> set[Path]:
    """Parse a merged strace trace and return the set of successfully modified absolute paths."""
    targets: set[Path] = set()
    if not trace_file.is_file():
        return targets
    for line in trace_file.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _STRACE_LINE_RE.match(line)
        if not match:
            continue
        syscall = match.group("sc")
        if syscall not in ALL_WRITE_SYSCALLS:
            continue
        args = match.group("args")
        rest = match.group("rest")
        if syscall in OPEN_SYSCALLS and not _write_flags_present(args, syscall):
            continue
        retval = _parse_retval(rest)
        if retval is None:
            continue
        success = retval >= 0 if syscall in OPEN_SYSCALLS else retval == 0
        if not success:
            continue
        for raw_path in _extract_write_paths(args, syscall):
            if not raw_path:
                continue
            targets.add(_norm_abs(Path(raw_path), cwd))
    return targets


def classify_writes(
    write_targets: set[Path],
    temp_root: Path,
    declared_temp_paths: list[Path],
    benign_roots: list[Path],
    benign_patterns: tuple[re.Pattern[str], ...],
) -> tuple[list[str], list[str]]:
    """Split traced writes into (extra_outputs_inside_temp, escapes_outside_temp_and_benign)."""
    extra: list[str] = []
    escapes: list[str] = []
    for path in sorted(write_targets):
        if _is_under(path, temp_root):
            if not any(
                path == declared or _is_under(path, declared) for declared in declared_temp_paths
            ):
                extra.append(str(path))
            continue
        if any(_is_under(path, root) for root in benign_roots):
            continue
        if any(pattern.match(str(path)) for pattern in benign_patterns):
            continue
        if _is_bytecode(path):
            continue
        escapes.append(str(path))
    return extra, escapes


# ---------------------------------------------------------------------------
# Command execution
# ---------------------------------------------------------------------------


@dataclass
class CommandOutcome:
    """The full recurrence outcome for a single registry entry."""

    id: str
    status: str  # "passed" | "failed"
    negative_control: bool
    expected_exit_code: int
    exit_code: int | None
    timed_out: bool
    duration_seconds: float
    timeout_seconds: int
    input_hash_check: str
    safety_check: str
    output_check: str
    containment_check: str
    error_reason: str | None
    declared_outputs: list[str] = field(default_factory=list)

    def to_report(self) -> dict[str, Any]:
        """Serialize the outcome to its compact machine-readable report shape."""
        return {
            "id": self.id,
            "status": self.status,
            "negative_control": self.negative_control,
            "expected_exit_code": self.expected_exit_code,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "duration_seconds": round(self.duration_seconds, 3),
            "timeout_seconds": self.timeout_seconds,
            "input_hash_check": self.input_hash_check,
            "safety_check": self.safety_check,
            "output_check": self.output_check,
            "containment_check": self.containment_check,
            "error_reason": self.error_reason,
            "declared_outputs": list(self.declared_outputs),
        }


def _run_under_strace(
    argv: list[str], cwd: Path, env: dict[str, str], timeout: int, trace_file: Path
) -> tuple[int | None, bool]:
    """Run ``argv`` under strace, returning ``(exit_code, timed_out)``.

    strace is started in a new session so a timeout can kill the whole traced process group.
    strace's own exit status mirrors the traced command's, so the returned code is the command's.
    """
    strace_path = shutil.which(STRACE_BIN)
    if strace_path is None:
        raise RuntimeError("strace binary not found on PATH; required for write containment")
    strace_argv = [
        strace_path,
        "-f",
        "-e",
        f"trace={STRACE_SYSCALLS}",
        "-s",
        "4096",
        "-o",
        str(trace_file),
        "--",
        *argv,
    ]
    try:
        proc = subprocess.Popen(
            strace_argv,
            cwd=str(cwd),
            env=env,
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        raise RuntimeError(f"failed to launch strace: {exc}") from exc
    try:
        exit_code = proc.wait(timeout=timeout)
        return exit_code, False
    except subprocess.TimeoutExpired:
        _kill_process_group(proc.pid)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        return None, True


def _kill_process_group(pgid: int) -> None:
    """Best-effort SIGKILL of an entire process group (strace + traced children)."""
    try:
        os.killpg(os.getpgid(pgid), signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass


def run_one_command(
    entry: dict[str, Any],
    repo_root: Path,
    scratch: Path,
    run_index: int,
) -> CommandOutcome:
    """Execute a single eligible entry under isolation and return its recurrence outcome."""
    entry_id = entry["id"]
    timeout_seconds = int(entry["timeout_seconds"])
    expected_outputs = list(entry["expected_outputs"])
    negative_control_spec = NEGATIVE_CONTROLS.get(entry_id)
    is_negative_control = negative_control_spec is not None
    expected_exit_code = (
        int(negative_control_spec["expected_exit_code"]) if negative_control_spec else 0
    )
    declared_outputs = list(expected_outputs)

    def _base(**overrides: Any) -> CommandOutcome:
        defaults: dict[str, Any] = {
            "id": entry_id,
            "status": "failed",
            "negative_control": is_negative_control,
            "expected_exit_code": expected_exit_code,
            "exit_code": None,
            "timed_out": False,
            "duration_seconds": 0.0,
            "timeout_seconds": timeout_seconds,
            "input_hash_check": "ok",
            "safety_check": "ok",
            "output_check": "not_run",
            "containment_check": "not_run",
            "error_reason": None,
            "declared_outputs": declared_outputs,
        }
        defaults.update(overrides)
        return CommandOutcome(**defaults)

    # Pre-check: input hashes.
    drift = verify_input_hashes(entry)
    if drift is not None:
        return _base(
            input_hash_check=drift,
            error_reason="input_drift",
        )

    # Pre-check: command safety.
    unsafe = re_safety_check(entry["command"])
    if unsafe is not None:
        return _base(safety_check=unsafe, error_reason="unsafe_command")

    # Isolated temporary output root for this command. ``command_index`` is per-source and is NOT
    # globally unique, so the run position disambiguates commands that share an index.
    slug = f"cmd_{run_index:02d}"
    temp_root = (scratch / slug).resolve()
    temp_root.mkdir(parents=True, exist_ok=True)

    # Pre-check: every declared output must be redirectable.
    rewritten, redirect_error = rewrite_outputs(entry["command"], expected_outputs, temp_root)
    if redirect_error is not None:
        return _base(
            output_check=redirect_error,
            containment_check="not_run",
            error_reason="output_not_redirectable",
        )
    # Pre-create the parent of each declared output in the temp root so commands that do not mkdir
    # their output parent can still write the redirected path.
    for declared in expected_outputs:
        try:
            (temp_root / declared).resolve().parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

    trace_file = scratch / f"{slug}.strace"
    benign_roots = _benign_write_roots(temp_root, trace_file, repo_root)
    declared_temp_paths = [(temp_root / Path(o)).resolve() for o in expected_outputs]
    cwd = _resolve_working_dir(entry["working_dir"])
    env = build_minimal_env()

    start = time.monotonic()
    exit_code, timed_out = _run_under_strace(rewritten, cwd, env, timeout_seconds, trace_file)
    duration = time.monotonic() - start

    if timed_out:
        return _base(
            exit_code=exit_code,
            timed_out=True,
            duration_seconds=duration,
            output_check="not_run",
            containment_check="not_run",
            error_reason="timeout",
        )

    # Exit-code contract: zero for ordinary commands; the declared negative-control code
    # (e.g. a violation detector) otherwise.
    if exit_code != expected_exit_code:
        return _base(
            exit_code=exit_code,
            duration_seconds=duration,
            output_check="not_run",
            containment_check="not_run",
            error_reason="nonzero_exit",
        )

    # Declared output presence + extra-output detection (walk the temp root) + write containment.
    output_check, containment_check, error_reason = _output_and_containment_outcome(
        expected_outputs, temp_root, trace_file, cwd, declared_temp_paths, benign_roots
    )
    status = "passed" if error_reason is None else "failed"
    return _base(
        status=status,
        exit_code=exit_code,
        duration_seconds=duration,
        output_check=output_check,
        containment_check=containment_check,
        error_reason=error_reason,
    )


def _output_and_containment_outcome(
    expected_outputs: list[str],
    temp_root: Path,
    trace_file: Path,
    cwd: Path,
    declared_temp_paths: list[Path],
    benign_roots: list[Path],
) -> tuple[str, str, str | None]:
    """Classify declared-output presence, extra outputs, and write escapes after a run."""
    missing = [declared for declared in expected_outputs if not (temp_root / declared).exists()]
    extra_outputs, escapes = _containment(
        trace_file, cwd, temp_root, declared_temp_paths, benign_roots, BENIGN_WRITE_PATTERNS
    )
    output_check = "ok"
    error_reason: str | None = None
    if missing:
        output_check = f"missing:{missing[0]}"
        error_reason = "missing_output"
    elif extra_outputs:
        output_check = f"extra:{extra_outputs[0]}"
        error_reason = "extra_output"
    containment_check = "ok" if not escapes else f"escape:{escapes[0]}"
    if escapes and error_reason is None:
        error_reason = "undeclared_write"
    return output_check, containment_check, error_reason


def _containment(
    trace_file: Path,
    cwd: Path,
    temp_root: Path,
    declared_temp_paths: list[Path],
    benign_roots: list[Path],
    benign_patterns: tuple[re.Pattern[str], ...] = (),
) -> tuple[list[str], list[str]]:
    """Combine the temp-root walk with strace write tracking for containment classification."""
    # Walk the temp root: every file present that is not a declared output (or inside one) is extra.
    extra: list[str] = []
    if temp_root.is_dir():
        for path in sorted(temp_root.rglob("*")):
            if path.is_dir():
                continue
            if not any(
                path.resolve() == declared or _is_under(path.resolve(), declared)
                for declared in declared_temp_paths
            ):
                extra.append(str(path.resolve()))
    # strace: every successful write outside the temp root and the benign set is an escape.
    write_targets = collect_write_targets(trace_file, cwd)
    strace_extra, escapes = classify_writes(
        write_targets, temp_root, declared_temp_paths, benign_roots, benign_patterns
    )
    extra.extend(strace_extra)
    # De-duplicate extras while preserving order.
    seen: set[str] = set()
    deduped_extra = [e for e in extra if not (e in seen or seen.add(e))]
    return deduped_extra, escapes


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _eligible(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the recurrence-eligible entries in registry order."""
    return [e for e in entries if e.get("recurrence_eligible") is True]


def _excluded(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the excluded entries in registry order."""
    return [e for e in entries if e.get("recurrence_eligible") is not True]


def _exclusions_from_registry(registry: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the controlled exclusion records derived from a registry's excluded entries."""
    entries = registry.get("entries", []) if isinstance(registry, dict) else []
    return [
        {"id": entry["id"], "exclusion_reason": entry["exclusion_reason"]}
        for entry in _excluded(entries)
    ]


def run_recurrence(
    registry_path: Path,
    report_path: Path,
    zero_entry_policy: str | None,
) -> tuple[dict[str, Any], int]:
    """Run the recurrence guard and return ``(report, exit_code)``."""
    head = git_head()
    registry: dict[str, Any] = {}
    command_outcomes: list[CommandOutcome] = []
    negative_control_warnings: list[str] = []

    def _report(err: str | None) -> dict[str, Any]:
        return _build_report(
            registry_path=registry_path,
            report_path=report_path,
            head=head,
            registry=registry,
            command_outcomes=command_outcomes,
            negative_control_warnings=negative_control_warnings,
            zero_entry_policy=zero_entry_policy,
            structural_error=err,
        )

    try:
        registry = load_registry(registry_path)
    except ValueError as exc:
        return _report(f"registry_load_error:{exc}"), 2

    errors = validate_registry_structure(registry)
    if errors:
        return _report(f"registry_structure_error:{errors[0]}"), 2

    if shutil.which(STRACE_BIN) is None:
        return _report("strace_missing"), 2

    entries = registry.get("entries", [])
    eligible_entries = _eligible(entries)

    # Stale negative-control allowlist detection: warn (do not fail) when a declared negative
    # control id is no longer present in the eligible set.
    eligible_ids = {entry["id"] for entry in eligible_entries}
    for nc_id in NEGATIVE_CONTROLS:
        if nc_id not in eligible_ids:
            negative_control_warnings.append(
                f"stale_negative_control:{nc_id} (registry no longer has this eligible entry)"
            )

    if not eligible_entries:
        if zero_entry_policy:
            return _report(None), 0
        return _report("empty_eligible_set"), 2

    scratch = Path(tempfile.mkdtemp(prefix="figure_recurrence_"))
    try:
        for run_index, entry in enumerate(eligible_entries):
            command_outcomes.append(run_one_command(entry, REPO_ROOT, scratch, run_index))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    report = _report(None)
    passed = all(o.status == "passed" for o in command_outcomes) and bool(command_outcomes)
    exit_code = 0 if passed else 1
    return report, exit_code


def _build_report(
    *,
    registry_path: Path,
    report_path: Path,
    head: str | None,
    registry: dict[str, Any],
    command_outcomes: list[CommandOutcome],
    negative_control_warnings: list[str],
    zero_entry_policy: str | None,
    structural_error: str | None,
) -> dict[str, Any]:
    """Assemble the compact machine-readable recurrence report payload."""
    provenance = registry.get("provenance") if isinstance(registry, dict) else None
    registry_source_commit = (
        provenance.get("source_commit") if isinstance(provenance, dict) else None
    )
    entries = registry.get("entries", []) if isinstance(registry, dict) else []
    passed_count = sum(1 for o in command_outcomes if o.status == "passed")
    return {
        "schema": REPORT_SCHEMA,
        "generated_at": now_iso(),
        "source_commit": head,
        "registry_source_commit": registry_source_commit,
        "registry_path": (
            registry_path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
            if _is_under(registry_path.resolve(), REPO_ROOT.resolve())
            else str(registry_path)
        ),
        "issue": ISSUE_NUMBER,
        "route_evidence_only": True,
        "claim_boundary": (
            "Local workflow recurrence and write containment only. No figure, benchmark, "
            "metric, schema, or publication claim."
        ),
        "strace_used": True,
        "total_entries": len(entries),
        "eligible_count": sum(1 for e in entries if e.get("recurrence_eligible") is True),
        "executed_count": len(command_outcomes),
        "passed_count": passed_count,
        "failed_count": len(command_outcomes) - passed_count,
        "negative_control_count": sum(1 for o in command_outcomes if o.negative_control),
        "zero_entry_policy": zero_entry_policy,
        "structural_error": structural_error,
        "negative_control_warnings": negative_control_warnings,
        "commands": [o.to_report() for o in command_outcomes],
        "exclusions": _exclusions_from_registry(registry),
        "report_path": str(report_path),
    }


def write_report(report: dict[str, Any], report_path: Path) -> None:
    """Write the report through the shared evidence writer (registers in the catalog when tracked)."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_evidence_json(report_path, report, catalog_area="workflow_evidence")


def main(argv: list[str] | None = None) -> int:
    """Entry point for the isolated figure-render recurrence guard."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=REGISTRY_DEFAULT,
        help=f"path to the v1 figure-render registry (default: {REGISTRY_DEFAULT.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPORT_DEFAULT,
        help=(
            "path to write the compact recurrence report (default: "
            f"{REPORT_DEFAULT.relative_to(REPO_ROOT)}; use a /tmp path for ad-hoc validation)"
        ),
    )
    parser.add_argument(
        "--zero-entry-policy",
        default=None,
        help=(
            "record an explicit policy string and exit 0 when the eligible set is empty "
            "(omit to fail closed on an empty eligible set)"
        ),
    )
    args = parser.parse_args(argv)

    report, exit_code = run_recurrence(args.registry, args.report, args.zero_entry_policy)
    write_report(report, args.report)
    # Human-readable summary on stderr (never stdout; stdout stays clean for piping).
    summary = (
        f"figure-render recurrence: executed={report['executed_count']} "
        f"passed={report['passed_count']} failed={report['failed_count']} "
        f"excluded={len(report['exclusions'])} exit={exit_code} "
        f"report={args.report}"
    )
    if report["structural_error"]:
        summary += f" structural_error={report['structural_error']}"
    print(summary, file=sys.stderr)
    for outcome in report["commands"]:
        if outcome["status"] != "passed":
            print(
                f"  failed: {outcome['id']} reason={outcome['error_reason']} "
                f"exit={outcome['exit_code']} output={outcome['output_check']} "
                f"containment={outcome['containment_check']}",
                file=sys.stderr,
            )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
